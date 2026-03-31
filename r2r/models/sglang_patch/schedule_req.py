from typing import Optional, Tuple, Union, List, Dict
from dataclasses import dataclass
from sglang.srt.sampling.sampling_params import SamplingParams


@dataclass
class EntropyLookaheadRpc:
    """SLM -> LLM: score token log p_LLM(token_k | context_k) for each pair."""

    query_id: int
    contexts: List[List[int]]
    tokens: List[int]


@dataclass
class EntropyLookaheadResp:
    query_id: int
    logprobs: List[float]
    ok: bool = True
    error: Optional[str] = None


@dataclass
class SlidingWindowJsRpc:
    """SLM -> LLM: next-token logits at each sliding window position.

    The LLM runs one short forward per position j on prefix full_ids[: base_len + L - n + j]
    (SGLang prefill only returns logits at the last token, not a full [seq_len, vocab] matrix).
    """

    query_id: int
    full_ids: List[int]
    base_len: int
    window_size: int


@dataclass
class SlidingWindowJsResp:
    query_id: int
    llm_logits: List  # list of numpy.ndarray or torch.Tensor, shape [vocab] each
    ok: bool = True
    error: Optional[str] = None


@dataclass
class NextTokenJsRpc:
    """SLM -> LLM: next-token logits p_LLM(· | context_ids) (one short forward).

    When ``rid`` is set, KV is kept for a follow-up ``WaitingReq`` with status
    ``EVJS_CONTINUE`` (same rid) so the formal LLM step does not prefill again.
    """

    query_id: int
    context_ids: List[int]
    rid: Optional[str] = None


@dataclass
class NextTokenJsAbortRpc:
    """Release KV from a prior fused NextTokenJsRpc(rid=...) when JS gate fails."""

    rid: str


@dataclass
class NextTokenJsResp:
    query_id: int
    llm_logits: Optional[object] = None  # numpy.ndarray [vocab] or None on failure
    ok: bool = True
    error: Optional[str] = None


class SimpleSamplingParams:
    def __init__(self, temperature: float = 1.0, top_k: int = -1, top_p: float = 1.0, max_new_tokens: int = 128):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
    
    def derive_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

class WaitingReq:
    def __init__(
        self,
        rid: str,
        new_token_ids: List[int],
        sampling_params: Optional[SimpleSamplingParams] = None,
        status: str = "need",
    ):
        self.rid = rid
        self.new_token_ids = new_token_ids
        self.sampling_params = sampling_params
        self.status = status

