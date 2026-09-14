from dataclasses import dataclass


@dataclass
class EngineConfig:
    cache_backend: str = "naive"       # "naive" | "mla"
    kv_quant_bits: int = 16            # 16 | 8 | 4
    weight_quant_bits: int = 16        # 16 | 8
    eviction: str = "none"             # "none" | "h2o" | "snapkv"
    prefix_cache: str = "off"          # "off" | "hash" | "radix"
    batching: str = "static"           # "static" | "continuous"
    decoding: str = "standard"         # "standard" | "speculative"
    spec_method: str | None = None     # "draft_verify" | "medusa" | "lookahead" | "eagle"
    use_lora: bool = False
    compiled: bool = False
    disaggregated: bool = False
