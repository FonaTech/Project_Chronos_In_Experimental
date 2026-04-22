import sys
import chronos.deps  # ensure minimind on sys.path

from model.model_minimind import MiniMindConfig


class ChronosConfig(MiniMindConfig):
    """
    Extends MiniMindConfig with Project Chronos-specific fields:
    - lookahead_steps: how many future token steps to predict expert routing for
    - num_shared_experts: experts always resident in VRAM as fallback
    - lambda_balance / lambda_temporal: loss coefficients
    - vram_budget_gb: VRAM cap for expert cache
    - prefetch_depth: async prefetch queue depth
    """
    model_type = "chronos"

    def __init__(self, **kwargs):
        self.lookahead_steps: int = kwargs.pop("lookahead_steps", 2)
        self.num_shared_experts: int = kwargs.pop("num_shared_experts", 1)
        self.lambda_balance: float = kwargs.pop("lambda_balance", 5e-4)
        self.lambda_temporal: float = kwargs.pop("lambda_temporal", 1e-3)
        self.vram_budget_gb: float = kwargs.pop("vram_budget_gb", 4.0)
        self.prefetch_depth: int = kwargs.pop("prefetch_depth", 2)
        # Hybrid attention params
        self.use_hybrid_attention: bool = kwargs.pop("use_hybrid_attention", True)
        self.kv_latent_dim: int = kwargs.pop("kv_latent_dim", 64)   # MLA compression dim
        self.rope_dim: int = kwargs.pop("rope_dim", 32)              # RoPE portion of head_dim
        self.sliding_window_size: int = kwargs.pop("sliding_window_size", 2048)
        # Pinned memory safety: max RAM for expert buffers (fraction of physical RAM)
        self.pinned_memory_max_fraction: float = kwargs.pop("pinned_memory_max_fraction", 0.25)
        # Storage backend for expert weights on SSD.
        # "safetensors" packs each Louvain cluster into one .ctsr file +
        # cluster_manifest.json (mmap-friendly, sequential reads).
        # "pt" keeps the legacy one-pickle-per-expert layout.
        self.storage_format: str = kwargs.pop("storage_format", "safetensors")
        # Optional path to a precomputed cluster_manifest.json. When set,
        # CacheManager will attach this layout instead of the one in ssd_dir.
        self.cluster_manifest_path = kwargs.pop("cluster_manifest_path", None)
        # M2: weight on the lookahead supervision loss
        self.lambda_lookahead: float = kwargs.pop("lambda_lookahead", 0.1)
        # M4: weight on the router KL anchor (alignment stages only).
        # Prevents SFT/DPO/ORPO/GRPO gradients from drifting routing away
        # from the pretrained distribution that the cluster layout was
        # optimized for. Typical values:
        #   pretrain: 0.0,  SFT: 0.01,  DPO/ORPO/GRPO: 0.1
        self.lambda_router_anchor: float = kwargs.pop("lambda_router_anchor", 0.0)
        # M7: regularization defaults. MiniMindConfig defaults dropout=0.0
        # and weight_decay isn't even a config field — both major causes
        # of overfitting on small corpora. Override here.
        kwargs.setdefault("dropout", 0.1)
        # Force MoE on
        kwargs.setdefault("use_moe", True)
        super().__init__(**kwargs)
