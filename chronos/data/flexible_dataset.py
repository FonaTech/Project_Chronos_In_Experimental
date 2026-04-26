"""
chronos/data/flexible_dataset.py

Streaming JSONL loaders with byte-offset indexing. Three variants cover
the six Chronos training stages without ever loading the file into RAM:

  FlexibleDataset       — pretrain / language modeling. Auto-detects
                           common text-field schemas (text, content,
                           instruction+output, conversations, messages).
  StreamingSFTDataset   — supervised fine-tuning. Uses the tokenizer's
                           chat template; only assistant tokens contribute
                           to the loss (label = -100 elsewhere). Drop-in
                           for minimind's SFTDataset.
  StreamingDPODataset   — preference pairs. Returns the dict shape DPO /
                           ORPO trainers expect.

Each dataset stores only `offsets: list[int]` (8 bytes/sample) plus the
file path. `__getitem__` does seek+readline+json.loads on demand, so a
multi-GB corpus uses < 1 GB of resident memory.

The file handle is opened lazily per process (re-opens after fork so
DataLoader num_workers > 0 still works) and protected by a thread lock
so num_workers = 0 paths with concurrent access are safe.
"""
from __future__ import annotations

import json
import os
import threading

import torch
from torch.utils.data import Dataset


# ── Field-name auto-detection (FlexibleDataset only) ──────────────────

_TEXT_KEYS = ("text", "content", "story", "passage", "document", "article")
_PAIR_KEYS = (
    ("instruction", "output"),
    ("instruction", "response"),
    ("prompt", "response"),
    ("prompt", "completion"),
    ("input", "output"),
    ("question", "answer"),
    ("query", "answer"),
)


def _extract_text(record: dict) -> str:
    for k in _TEXT_KEYS:
        if k in record and record[k]:
            return str(record[k])
    for k1, k2 in _PAIR_KEYS:
        if k1 in record and k2 in record:
            parts = [str(record[k1]).strip(), str(record[k2]).strip()]
            return "\n".join(p for p in parts if p)
    if "conversations" in record and isinstance(record["conversations"], list):
        return " ".join(
            str(c.get("value", c.get("content", "")))
            for c in record["conversations"]
        )
    if "messages" in record and isinstance(record["messages"], list):
        return " ".join(str(m.get("content", "")) for m in record["messages"])
    parts = [str(v) for v in record.values()
             if isinstance(v, (str, int, float)) and str(v).strip()]
    return " ".join(parts)


# ── Shared streaming infrastructure ───────────────────────────────────

class _StreamingJSONLBase(Dataset):
    """Base class providing byte-offset index + lazy per-process file handle."""

    def __init__(self, data_path: str):
        self.data_path = os.path.abspath(data_path)
        self.offsets: list[int] = []
        with open(self.data_path, "rb") as f:
            pos = 0
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(pos)
                pos = f.tell()
        if not self.offsets:
            raise ValueError(f"No JSONL records in {data_path}")
        self._fh = None
        self._fh_lock = threading.Lock()

    def _get_fh(self):
        pid = os.getpid()
        if self._fh is None or getattr(self._fh, "_chronos_pid", None) != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.data_path, "rb")
            self._fh._chronos_pid = pid
        return self._fh

    def _read_record(self, index: int) -> dict:
        offset = self.offsets[index]
        with self._fh_lock:
            fh = self._get_fh()
            fh.seek(offset)
            raw = fh.readline()
        return json.loads(raw.decode("utf-8"))

    def __len__(self):
        return len(self.offsets)

    def __getstate__(self):
        """Make streaming datasets safe for macOS/Windows spawn workers.

        DataLoader with num_workers > 0 pickles the dataset. File handles and
        thread locks cannot be pickled, and each worker must open its own file
        descriptor anyway so seeks are process-local.
        """
        state = self.__dict__.copy()
        state["_fh"] = None
        state["_fh_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fh = None
        self._fh_lock = threading.Lock()

    def __del__(self):
        try:
            if self._fh is not None:
                self._fh.close()
        except Exception:
            pass


# ── Pretrain / generic LM ────────────────────────────────────────────

class FlexibleDataset(_StreamingJSONLBase):
    """Pretrain-style streaming dataset; auto-detects the text field."""

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        first = self._read_record(0)
        detected = _extract_text(first)
        field_hint = (
            next((k for k in _TEXT_KEYS if k in first), None)
            or next((f"{k1}+{k2}" for k1, k2 in _PAIR_KEYS
                     if k1 in first and k2 in first), None)
            or ("conversations" if "conversations" in first else "auto")
        )
        print(f"[FlexibleDataset] {len(self.offsets)} records (streaming), "
              f"detected format: '{field_hint}', "
              f"sample preview: {detected[:80]!r}")

    def __getitem__(self, index):
        record = self._read_record(index)
        text = _extract_text(record)
        tok = self.tokenizer(
            text, add_special_tokens=False,
            max_length=self.max_length - 2, truncation=True,
        )
        tokens = (
            [self.tokenizer.bos_token_id]
            + tok.input_ids
            + [self.tokenizer.eos_token_id]
        )
        pad_id = self.tokenizer.pad_token_id or 0
        tokens = tokens + [pad_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == pad_id] = -100
        return input_ids, labels


# ── SFT (conversations → assistant-token loss mask) ─────────────────

class StreamingSFTDataset(_StreamingJSONLBase):
    """Streaming counterpart to minimind's SFTDataset.

    Reads `conversations: [{role, content, ...}, ...]` records, applies
    the tokenizer's chat template, and masks every position outside an
    assistant turn with -100 so the loss only fires on assistant tokens.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Sentinels marking the assistant turn boundary in encoded form.
        # Same formulation minimind uses (`bos`+'assistant\n' / `eos`+'\n').
        self.bos_id = tokenizer(
            f'{tokenizer.bos_token}assistant\n', add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f'{tokenizer.eos_token}\n', add_special_tokens=False
        ).input_ids
        print(f"[StreamingSFTDataset] {len(self.offsets)} records (streaming)")

    def _build_prompt(self, conversations) -> str:
        messages = []
        tools = None
        for m in conversations:
            m = dict(m)
            if m.get("role") == "system" and m.get("tools"):
                tools = json.loads(m["tools"]) if isinstance(m["tools"], str) else m["tools"]
            if m.get("tool_calls") and isinstance(m["tool_calls"], str):
                m["tool_calls"] = json.loads(m["tool_calls"])
            messages.append(m)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools,
        )

    def _label_mask(self, input_ids: list[int]) -> list[int]:
        """-100 outside every <bos>assistant\\n ... <eos>\\n span."""
        labels = [-100] * len(input_ids)
        i = 0
        nb = len(self.bos_id)
        ne = len(self.eos_id)
        while i < len(input_ids):
            if input_ids[i:i + nb] == self.bos_id:
                start = i + nb
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + ne] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + ne, self.max_length)):
                    labels[j] = input_ids[j]
                i = end + ne if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        record = self._read_record(index)
        convs = record.get("conversations")
        if not convs:
            # Graceful fall-through for mixed corpora — treat as pretrain.
            text = _extract_text(record)
            ids = self.tokenizer(text, truncation=True,
                                 max_length=self.max_length).input_ids
        else:
            prompt = self._build_prompt(convs)
            ids = self.tokenizer(prompt).input_ids[:self.max_length]
        pad_id = self.tokenizer.pad_token_id or 0
        ids = ids + [pad_id] * (self.max_length - len(ids))
        labels = self._label_mask(ids)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# ── DPO / ORPO (chosen + rejected pair) ─────────────────────────────

class StreamingDPODataset(_StreamingJSONLBase):
    """Streaming counterpart to minimind's DPODataset.

    Returns the dict shape the DPO/ORPO trainers expect:
      {x_chosen, y_chosen, mask_chosen, x_rejected, y_rejected, mask_rejected}
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        super().__init__(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(
            f'{tokenizer.bos_token}assistant\n', add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer(
            f'{tokenizer.eos_token}\n', add_special_tokens=False
        ).input_ids
        print(f"[StreamingDPODataset] {len(self.offsets)} pairs (streaming)")

    def _loss_mask(self, ids: list[int]) -> list[int]:
        mask = [0] * len(ids)
        i = 0
        nb = len(self.bos_id); ne = len(self.eos_id)
        while i < len(ids):
            if ids[i:i + nb] == self.bos_id:
                start = i + nb
                end = start
                while end < len(ids):
                    if ids[end:end + ne] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + ne, self.max_length)):
                    mask[j] = 1
                i = end + ne if end < len(ids) else len(ids)
            else:
                i += 1
        return mask

    def __getitem__(self, index):
        rec = self._read_record(index)
        chosen = rec["chosen"]; rejected = rec["rejected"]

        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False,
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False,
        )

        c_enc = self.tokenizer(chosen_prompt, truncation=True,
                               max_length=self.max_length, padding="max_length")
        r_enc = self.tokenizer(rejected_prompt, truncation=True,
                               max_length=self.max_length, padding="max_length")
        c_ids = c_enc["input_ids"]
        r_ids = r_enc["input_ids"]
        c_mask = self._loss_mask(c_ids)
        r_mask = self._loss_mask(r_ids)
        return {
            "x_chosen":     torch.tensor(c_ids[:-1], dtype=torch.long),
            "y_chosen":     torch.tensor(c_ids[1:],  dtype=torch.long),
            "mask_chosen":  torch.tensor(c_mask[1:], dtype=torch.long),
            "x_rejected":   torch.tensor(r_ids[:-1], dtype=torch.long),
            "y_rejected":   torch.tensor(r_ids[1:],  dtype=torch.long),
            "mask_rejected":torch.tensor(r_mask[1:], dtype=torch.long),
        }
