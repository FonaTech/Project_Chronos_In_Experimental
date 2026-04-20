"""
chronos/data/flexible_dataset.py

Robust dataset loader that accepts any JSONL format without requiring a
specific field name. Tries common text field names in order, then falls
back to concatenating all string values in the record.

Supported formats (auto-detected):
  {"text": "..."}                    ← minimind pretrain
  {"content": "..."}                 ← common HuggingFace datasets
  {"instruction": "...", "output": "..."}  ← Alpaca-style
  {"conversations": [...]}           ← ShareGPT-style (SFT)
  {"prompt": "...", "response": "..."}
  {"input": "...", "output": "..."}
  {"question": "...", "answer": "..."}
  any JSON with string values        ← last-resort: join all strings
"""
import json
import torch
from torch.utils.data import Dataset

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
    # 1. Direct text field
    for k in _TEXT_KEYS:
        if k in record and record[k]:
            return str(record[k])

    # 2. Instruction/response pairs
    for k1, k2 in _PAIR_KEYS:
        if k1 in record and k2 in record:
            parts = [str(record[k1]).strip(), str(record[k2]).strip()]
            return "\n".join(p for p in parts if p)

    # 3. ShareGPT conversations
    if "conversations" in record:
        convs = record["conversations"]
        if isinstance(convs, list):
            return " ".join(
                str(c.get("value", c.get("content", "")))
                for c in convs
            )

    # 4. messages (OpenAI format)
    if "messages" in record:
        msgs = record["messages"]
        if isinstance(msgs, list):
            return " ".join(
                str(m.get("content", "")) for m in msgs
            )

    # 5. Last resort: join all non-empty string values
    parts = [str(v) for v in record.values()
             if isinstance(v, (str, int, float)) and str(v).strip()]
    return " ".join(parts)


class FlexibleDataset(Dataset):
    """
    Loads any JSONL file regardless of field names.
    Tokenises with bos/eos, pads to max_length.
    Returns (input_ids, labels) tensors compatible with minimind's training loop.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not self.samples:
            raise ValueError(f"No valid JSON records found in {data_path}")

        # Detect format by inspecting first record
        first = self.samples[0]
        detected = _extract_text(first)
        field_hint = (
            next((k for k in _TEXT_KEYS if k in first), None)
            or next((f"{k1}+{k2}" for k1, k2 in _PAIR_KEYS
                     if k1 in first and k2 in first), None)
            or ("conversations" if "conversations" in first else "auto")
        )
        print(f"[FlexibleDataset] {len(self.samples)} records, "
              f"detected format: '{field_hint}', "
              f"sample preview: {detected[:80]!r}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text = _extract_text(self.samples[index])
        tok = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length - 2,
            truncation=True,
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
