import torch
import torch.nn.functional as F
from typing import Tuple, Union

def compute_entropy(logits: torch.Tensor) -> Union[float, torch.Tensor]:
    """
    Calculate entropy of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        
    Returns:
        Entropy values as a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size]
    
    return entropy.item() if is_single_input else entropy

def compute_js_divergence_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Symmetric Jensen–Shannon divergence JS(P, Q) with P,Q = softmax(logits).

    Args:
        logits_p: 1D tensor [vocab] or 2D [1, vocab]
        logits_q: same shape as logits_p

    Returns:
        Scalar JS divergence in natural log units (nats).
    """
    if logits_p.dim() == 1:
        logits_p = logits_p.unsqueeze(0)
    if logits_q.dim() == 1:
        logits_q = logits_q.unsqueeze(0)
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    m = 0.5 * (p + q)
    log_p = torch.log(p + eps)
    log_q = torch.log(q + eps)
    log_m = torch.log(m + eps)
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return float((0.5 * (kl_pm + kl_qm)).item())


def compute_variance(logits: torch.Tensor) -> Union[float, torch.Tensor]:
    """
    计算 softmax 概率分布的方差 Var(p)。
    方差小 -> 概率均匀分散（模型困惑）；方差大 -> 概率集中在少数 token。
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        
    Returns:
        Variance values as a scalar (if single input) or tensor of shape [batch_size]
    """
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    probs = F.softmax(logits, dim=-1)
    mean_probs = probs.mean(dim=-1, keepdim=True)
    variance = ((probs - mean_probs) ** 2).mean(dim=-1)  # [batch_size]
    
    return variance.item() if is_single_input else variance

def compute_logu(logits: torch.Tensor, topk: int = 10) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    """
    Calculate log-u score of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider
        
    Returns:
        Tuple of (aleatoric_uncertainty, epistemic_uncertainty)
        Each is a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    # Get top-k logits and their indices
    topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)  # [batch_size, topk]
    
    # Calculate sum of logits (S)
    alpha = torch.sum(topk_logits, dim=-1, keepdim=True)  # [batch_size, 1]
    
    # Calculate normalized probabilities (p_i = x_i/S)
    probs = topk_logits / alpha  # [batch_size, topk]
    
    # Calculate digamma terms
    digamma_xi = torch.digamma(topk_logits + 1)  # ψ(x_i + 1)
    digamma_sum = torch.digamma(alpha + 1)  # ψ(S + 1)
    
    # Calculate aleatoric uncertainty efficiently
    # AU = -∑(p_i * (ψ(x_i + 1) - ψ(S + 1)))
    aleatoric_uncertainty = -torch.sum(probs * (digamma_xi - digamma_sum), dim=-1)  # [batch_size]
    
    # Calculate epistemic uncertainty
    # EU = K / (S + K)
    epistemic_uncertainty = topk / (alpha.squeeze(-1) + topk)  # [batch_size]
    
    if is_single_input:
        return aleatoric_uncertainty.item(), epistemic_uncertainty.item()
    else:
        return aleatoric_uncertainty, epistemic_uncertainty


def log_prob_of_token(logits: torch.Tensor, token_id: int) -> float:
    """log softmax probability of a single token id. logits: [vocab] or [1, vocab]."""
    if logits.dim() == 2:
        logits = logits.squeeze(0)
    return F.log_softmax(logits, dim=-1)[token_id].item()


def compute_reliability(logits: torch.Tensor, topk: int = 10) -> Union[float, torch.Tensor]:
    """
    Calculate reliability of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider
        
    Returns:
        Reliability values as a scalar (if single input) or tensor of shape [batch_size]
    """
    aleatoric_uncertainty, epistemic_uncertainty = compute_logu(logits, topk)
    
    # Handle both scalar and tensor inputs
    if isinstance(aleatoric_uncertainty, float):
        return 1 / (aleatoric_uncertainty * epistemic_uncertainty)
    else:
        return 1 / (aleatoric_uncertainty * epistemic_uncertainty)
