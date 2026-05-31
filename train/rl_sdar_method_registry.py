"""
Unified registration for RL-SDAR training methods: dataset construction + train step.

Add a new method:
  1) Subclass RLSDARMethodHandler (or compose from helpers below).
  2) register_rl_sdar_method("MyMethod", MyHandler()).

Optionally split data vs forward later by delegating inside train_step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple
from abc import ABC, abstractmethod
from functools import partial

import torch
from torch.export.graph_signature import TensorArgument
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
    """
    Perform a single "round" on one sample b:
    - For each block, take the minimum non -1 value in step_map.
    - Create pmask (positions equal to the block minimum).
    - Create a noise mask for the extended segment (positions >= block minimum).
    - Mark the chosen minimum positions in step_map as -1 for the next round.
    Returns:
    extended_input_ids_b : Tensor with duplicated + masked response segment
    pmask_b              : Boolean mask for tokens selected in this round
    new_step_map_b       : Updated step_map (selected positions set to -1)
    has_any              : Whether any position was selected in this round
    """
    device = input_ids_b.device
    NB = (L1 + block_size - 1) // block_size
    pad_len = NB * block_size - L1
    # Reshape step_map into [NB, block_size], fill last incomplete block with -1
    step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
    step_pad[:L1] = step_map_b
    step_blk = step_pad.view(NB, block_size)                      # [NB, Bk]
    valid = step_blk.ge(0)                                        # Valid positions (not -1)
    big = torch.iinfo(step_blk.dtype).max
    tmp = step_blk.masked_fill(~valid, big)                       # Fill invalid positions with a large value
    min_vals, _ = tmp.min(dim=1, keepdim=True)                    # Current minimum for each block
    # Select positions equal to block minimum (only valid positions)
    pmask_blk = step_blk.eq(min_vals) & valid                     
    if not pmask_blk.any():
        # No positions left to select in this round
        return None, None, step_map_b, False
    # Noise mask for extended segment: mark positions >= block minimum
    ge_mask_blk = step_blk.ge(min_vals) & valid                   # [NB, Bk]
    # Flatten back to length L1 (discard padding)
    pmask_tail = pmask_blk.view(-1)[:L1]                          # [L1]
    ge_mask_tail = ge_mask_blk.view(-1)[:L1]                      # [L1]
    # Construct pmask_b: [0:L0] = False, [L0:] = pmask_tail
    pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
    pmask_b[L0:] = pmask_tail
    # Build extended segment: duplicate response and replace noise positions with mask_id
    tail = input_ids_b[L0:L0+L1].clone()
    tail[ge_mask_tail] = mask_id
    
    extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
    extended_input_ids_b[:L0+L1] = input_ids_b
    extended_input_ids_b[L0+L1:] = tail
    # Update step_map: mark selected minimum positions as -1 for the next round
    new_step_map_b = step_map_b.clone()
    new_step_map_b[pmask_tail] = -1
    return extended_input_ids_b, pmask_b, new_step_map_b, True


def one_round_vectorized_strict_block_serial(
    input_ids_b: torch.Tensor,
    step_map_b: torch.Tensor,
    L0: int,
    L1: int,
    block_size: int,
    mask_id: int,
):
    """
    Strict block-serial variant of `one_round_vectorized`.

    Semantics:
    - At each "round", only the earliest (leftmost) block that still has any valid (>=0) step entries
      is allowed to contribute prediction targets (`p_mask`).
    - Tokens in later blocks are fully masked in the duplicated tail, so the model cannot condition on
      future blocks while predicting the current one.
    - Within the current block, positions with step_map >= current_min are masked in the tail;
      positions with step_map < current_min are treated as already-confirmed and kept unmasked.

    Returns:
      extended_input_ids_b: [L0 + L1 + L1]
      pmask_b:             [L0 + L1]  (True only on positions predicted this round)
      new_step_map_b:      [L1]       (predicted positions set to -1)
      has_any:             bool
    """
    device = input_ids_b.device
    # [NB, block_size] padded view of step_map
    NB = (L1 + block_size - 1) // block_size
    step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
    step_pad[:L1] = step_map_b.to(device=device, dtype=torch.long)
    step_blk = step_pad.view(NB, block_size)

    valid_blk = step_blk.ge(0).any(dim=1)  # [NB]
    if not valid_blk.any():
        return None, None, step_map_b, False

    cur_blk = int(valid_blk.long().argmax().item())
    cur = step_blk[cur_blk]  # [block_size]
    valid = cur.ge(0)
    big = torch.iinfo(cur.dtype).max
    cur_min = int(cur.masked_fill(~valid, big).min().item())

    # p_mask: only min-step positions in the current block
    pmask_blk = cur.eq(cur_min) & valid  # [block_size]
    if not pmask_blk.any():
        return None, None, step_map_b, False

    pmask_tail = torch.zeros((L1,), dtype=torch.bool, device=device)
    start = cur_blk * block_size
    end = min(L1, (cur_blk + 1) * block_size)
    pmask_tail[start:end] = pmask_blk[: (end - start)]

    pmask_b = torch.zeros((L0 + L1,), dtype=torch.bool, device=device)
    pmask_b[L0:] = pmask_tail

    # Build duplicated tail: only reveal past + already-confirmed part of current block.
    tail = input_ids_b[L0 : L0 + L1].clone()
    if start > 0:
        # earlier blocks: keep as-is (confirmed history)
        pass

    # current block: mask positions with step >= cur_min (includes pmask targets + future steps)
    cur_tail = tail[start:end]
    cur_steps = step_map_b[start:end].to(device=device, dtype=torch.long)
    cur_mask = cur_steps.ge(cur_min) & cur_steps.ge(0)
    cur_tail[cur_mask] = mask_id
    tail[start:end] = cur_tail

    # later blocks: fully mask (strict serial across blocks)
    if end < L1:
        tail[end:] = mask_id

    extended_input_ids_b = torch.empty((L0 + L1 + L1,), dtype=input_ids_b.dtype, device=device)
    extended_input_ids_b[: L0 + L1] = input_ids_b
    extended_input_ids_b[L0 + L1 :] = tail

    new_step_map_b = step_map_b.clone()
    new_step_map_b[pmask_tail] = -1
    return extended_input_ids_b, pmask_b, new_step_map_b, True

def collapse_k_unique(lst, k: int):
    if k <= 0:
        raise ValueError("k must be > 0")
    uniq = sorted(set(lst))

    mapping = {}
    n = len(uniq)
    for idx, val in enumerate(uniq):
        group = idx // k
        end_idx = min((group + 1) * k - 1, n - 1)
        rep = uniq[end_idx]
        mapping[val] = rep
    return [mapping[x] for x in lst]

    

def process_pad_fn(attn, input_ids, start_pos, pad_id, L0, L1):
    N = L0 + 2 * L1
    device = input_ids.device
    cols = torch.arange(N, device=device)                  # (N,)
    key_mask = (cols < start_pos).unsqueeze(0) & (input_ids == pad_id)  # (B, N)
    # set -inf
    attn.masked_fill_(key_mask[:, None, None, :], 0)
    # aviod +-inf or none in forward
    A = attn[:, 0]  # (B, N, N)
    bad = (A.sum(dim=-1) == 0) & (torch.arange(A.size(1), device=A.device).unsqueeze(0) < start_pos)
    b, r = bad.nonzero(as_tuple=True)
    A[b, r, :] = 0; A[b, r, r] = 1  
    return attn

class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, reward, mask_ratio=None):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward   = reward
        self.mask_ratio = mask_ratio
        self.logp_old = torch.full(
            (len(extended_input_ids), p_mask.shape[1]), 
            float('-inf')
        )
        # self.logp_old_seq = torch.full(
        #     (len(extended_input_ids),), float('-inf')
        # )

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.reward[idx],
            # self.mask_ratio[idx],
        )
    
    @staticmethod
    def simple_collate(batch):
        idx, extended_input_ids, p_mask, tok_idx_ext, labels, reward = zip(*batch)
        return {
            "ids":        torch.tensor(idx),
            "extended_input_ids":  torch.stack(extended_input_ids),
            "p_mask":  torch.stack(p_mask),
            "tok_idx_ext":  torch.stack(tok_idx_ext),
            "labels":  torch.stack(labels),
            "reward":     reward
        }
    

class DUELTrainDataset(Dataset):
    def __init__(
        self,
        extended_input_ids: Tensor,
        p_mask: Tensor,
        tok_idx_ext: Tensor,
        labels: Tensor,
        reward: Tensor,
        valid: Tensor,
    ):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward = reward
        self.valid = valid
        self.logp_old_tok = torch.full((extended_input_ids.shape[0], extended_input_ids.shape[1], p_mask.shape[-1]), float("-inf"))
        self.logp_old_seq = torch.full((extended_input_ids.shape[0], extended_input_ids.shape[1]), float("-inf"))

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.reward[idx],
            self.valid[idx],
        )

    @staticmethod
    def collate(batch):
        ids, extended_input_ids, p_mask, tok_idx_ext, labels, reward, valid = zip(*batch)
        return {
            "ids": torch.tensor(ids),
            "extended_input_ids": torch.stack(extended_input_ids),
            "p_mask": torch.stack(p_mask),
            "tok_idx_ext": torch.stack(tok_idx_ext),
            "labels": torch.stack(labels),
            "reward": torch.stack(reward),
            "valid": torch.stack(valid),
        }


@dataclass
class RLTrainStepContext:
    """Closures / handles needed for one optimization step."""

    accelerator: Any
    dataset_lm: Dataset
    train_dataloader_lm: DataLoader
    model: Any
    config: Any
    basic_block_attention: Tensor
    start_pos: int
    response_length: int
    pad_id: int


@dataclass
class RLBuildContext:
    """Everything needed to materialize the LM dataset (no model)."""

    input_ids_lm: Tensor
    step_map_list: list
    reward_list: list
    config: Any
    start_pos: int
    post_num: Optional[int]
    mask_id: int
    pad_id: int
    collect_training_data: Callable[..., Tuple[Any, ...]]
    # collapse_k_unique: Callable[..., list]
    # one_round_vectorized: Callable[..., Tuple[Any, ...]]
    # train_dataset_class: type
    # simple_collate_fn: Callable[[list], Dict[str, Any]]

@dataclass
class ForwardProcessContext:
    extended_input_ids: Tensor
    p_mask: Tensor
    tok_idx_ext: Tensor
    labels: Tensor
    adv: Tensor
    logp_old: Tensor
    attention_mask: Tensor
    L0: int
    L1: int


class RLSDARMethodHandler(Protocol):
    """One training.method: how to build data and how to run train_step."""

    def build(self, ctx: RLBuildContext) -> Tuple[Dataset, Callable[[list], Dict[str, Any]]]:
        ...

    @property
    def logp_old_is_seq(self) -> bool:
        ...

    def train_step(self, tctx: RLTrainStepContext, batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        ...

    def compute_logp_old(self, tctx: RLTrainStepContext, train_dataloader_lm: Any) -> None:
        ...

@dataclass
class TrainStepMetrics:
    tot_loss: Tensor
    policy_loss: Tensor
    kl_loss: Tensor
    kl_mean: Tensor
    kl_std: Tensor
    log_ratio_mean: Tensor
    log_ratio_stsd: Tensor
    adv_mean: Tensor
    adv_std: Tensor
    clip_count: Tensor
    clip_total: Tensor



def _forward_process_token(
    tctx: RLTrainStepContext,
    *,
    extended_input_ids: Tensor,
    p_mask: Tensor,
    tok_idx_ext: Tensor,
    labels: Tensor,
    adv: Tensor,
    logp_old_tok: Tensor,
    mask_ratio: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    config = tctx.config
    model = tctx.model
    start_pos = tctx.start_pos
    basic_block_attention = tctx.basic_block_attention
    process_pad = tctx.process_pad
    get_elbo = tctx.get_elbo
    kl_estimator = tctx.kl_estimator

    adv = torch.as_tensor(adv, device=extended_input_ids.device, dtype=torch.float32).detach()

    B, L = p_mask.shape
    L0 = start_pos
    L1 = L - L0
    device = extended_input_ids.device

    attention_mask = basic_block_attention.clone()
    attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
    attention_mask = process_pad(attention_mask, extended_input_ids)

    logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
    logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)

    elbo = get_elbo(labels[:, L0:], logits[:, L0:], mask_ratio[:, L0:], p_mask[:, L0:])
    log_probs = F.log_softmax(logits, dim=-1)
    logp_new_tok = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    log_ratio = logp_new_tok - logp_old_tok
    ratio = torch.where(p_mask, log_ratio, torch.zeros_like(log_ratio)).clamp(-10.0, 10.0)
    ratio = torch.exp(ratio)
    clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)

    kl_loss = torch.tensor(0.0, device=device)
    kl_mean = torch.zeros((), device=device)
    kl_std = torch.zeros((), device=device)
    kl_estimator_type = config.training.get("kl_estimator", "k1")
    kl_mode = config.training.get("kl_mode", "in_loss")
    use_kl_importance_weight = config.training.get("use_kl_importance_weight", False)
    use_kl_clip = config.training.get("use_kl_clip", False)
    use_kl_length_normalization = config.training.get("use_kl_length_normalization", False)
    if config.training.beta > 0:
        log_ratio = torch.where(p_mask, log_ratio, torch.zeros_like(log_ratio))
        if kl_estimator_type == "rb":
            kl_seq = kl_estimator(log_ratio, kl_estimator_type, log_ratio_old=logp_old_tok)
        else:
            kl_seq = kl_estimator(log_ratio, kl_estimator_type)
        if use_kl_importance_weight and kl_estimator_type != "rb":
            iw_tok = torch.exp(log_ratio)
        else:
            iw_tok = torch.ones_like(log_ratio)

        if use_kl_clip:
            clipped_iw_tok = torch.clamp(iw_tok, 1 - config.training.eps, 1 + config.training.eps)
            kl_seq = torch.min(iw_tok * kl_seq, clipped_iw_tok * kl_seq)
        else:
            kl_seq = iw_tok * kl_seq

        if use_kl_length_normalization:
            kl_seq = kl_seq.sum(dim=1) / L1
        else:
            kl_seq = kl_seq.sum(dim=1)
        kl_mean = kl_seq.mean()
        kl_std = kl_seq.std(unbiased=False) if kl_seq.numel() > 1 else torch.zeros((), device=kl_seq.device)
        kl_loss = config.training.beta * kl_seq.sum() / B

    adv_tok = adv.unsqueeze(1)
    if kl_mode == "in_reward":
        adv_eff_tok = adv_tok - kl_loss.detach()
    elif kl_mode == "in_loss":
        adv_eff_tok = adv_tok
    else:
        raise ValueError(f"Unknown KL mode: {kl_mode}")

    surrogate_tok = torch.min(ratio * adv_eff_tok, clipped * adv_eff_tok)
    surrogate_tok = surrogate_tok * p_mask
    surrogate_tok = surrogate_tok.sum(dim=1) / L1

    policy_loss = -(surrogate_tok.sum() / B)
    total_loss = policy_loss + kl_loss if kl_mode == "in_loss" else policy_loss

    surrogate_tok_abs_mean = surrogate_tok.abs().mean()
    surrogate_tok_pos_frac = (surrogate_tok > 0).float().mean()
    log_ratio_tok_mean = log_ratio.mean()
    log_ratio_tok_std = log_ratio.std(unbiased=False) if log_ratio.numel() > 1 else torch.zeros((), device=log_ratio.device)
    log_ratio_tok_abs_mean = log_ratio.abs().mean()
    log_ratio_tok_per_tok_mean = (log_ratio / L1).mean()
    clip_mask = ((ratio - clipped).abs() > 1e-8) & p_mask
    clip_count = clip_mask.float().sum()
    clip_total = p_mask.float().sum()
    reward_mean = adv.mean()
    reward_std = adv.std(unbiased=False) if adv.numel() > 1 else torch.zeros((), device=adv.device)
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
    entropy_mean = (entropy * p_mask).sum() / p_mask.sum().clamp(min=1)
    elbo_mean = elbo.mean()
    elbo_std = elbo.std(unbiased=False) if elbo.numel() > 1 else torch.zeros((), device=elbo.device)

    metrics = {
        "loss/total": total_loss.detach(),
        "loss/policy": policy_loss.detach(),
        "loss/kl": kl_loss.detach(),
        "policy/surr_abs_mean": surrogate_tok_abs_mean.detach(),
        "policy/surr_pos_frac": surrogate_tok_pos_frac.detach(),
        "kl/mean": kl_mean.detach(),
        "kl/std": kl_std.detach(),
        "policy/entropy": entropy_mean.detach(),
        "log_ratio/tok_mean": log_ratio_tok_mean.detach(),
        "log_ratio/tok_std": log_ratio_tok_std.detach(),
        "log_ratio/tok_abs_mean": log_ratio_tok_abs_mean.detach(),
        "log_ratio/per_tok_mean": log_ratio_tok_per_tok_mean.detach(),
        "adv_eff/mean": adv_eff_tok.mean().detach(),
        "adv_eff/std": adv_eff_tok.std(unbiased=False) if adv_eff_tok.numel() > 1 else torch.zeros((), device=adv_eff_tok.device).detach(),
        "adv/mean": reward_mean.detach(),
        "adv/std": reward_std.detach(),
        "clip_count": clip_count.detach(),
        "clip_total": clip_total.detach(),
        "elbo/mean": elbo_mean.detach(),
        "elbo/std": elbo_std.detach(),
    }
    return total_loss, metrics


def _forward_process_espo(
    tctx: RLTrainStepContext,
    *,
    extended_input_ids: Tensor,
    p_mask: Tensor,
    tok_idx_ext: Tensor,
    labels: Tensor,
    adv: Tensor,
    logp_old_seq: Tensor,
    mask_ratio: Tensor,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    config = tctx.config
    model = tctx.model
    start_pos = tctx.start_pos
    basic_block_attention = tctx.basic_block_attention
    process_pad = tctx.process_pad
    get_elbo = tctx.get_elbo
    kl_estimator = tctx.kl_estimator
    kl_mode = tctx.kl_mode

    adv = torch.as_tensor(adv, device=extended_input_ids.device, dtype=torch.float32).detach()
    B, L = p_mask.shape
    L0 = start_pos
    L1 = L - L0
    device = extended_input_ids.device

    attention_mask = basic_block_attention.clone()
    attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
    attention_mask = process_pad(attention_mask, extended_input_ids)

    logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
    logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)
    logp_new_seq = get_elbo(labels[:, L0:], logits[:, L0:], mask_ratio[:, L0:], p_mask[:, L0:])
    log_ratio_seq = logp_new_seq - logp_old_seq
    ratio = (log_ratio_seq / L1).clamp(-10.0, 10.0)
    ratio = torch.exp(ratio)
    clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)

    beta = config.training.beta
    kl_per_seq = torch.zeros(B, device=device)
    kl_mean = torch.zeros((), device=device)
    kl_std = torch.zeros((), device=device)
    kl_loss = torch.zeros(B, device=device)
    if beta > 0 and config.training.kl_estimator != "none":
        kl_per_seq = kl_estimator(log_ratio_seq, config.training.kl_estimator)
        kl_mean = kl_per_seq.mean()
        kl_std = kl_per_seq.std(unbiased=False) if kl_per_seq.numel() > 1 else torch.zeros((), device=kl_per_seq.device)

    if kl_mode == "in_reward":
        adv_effective = adv - beta * kl_per_seq.detach()
    elif kl_mode == "in_loss":
        adv_effective = adv
        kl_loss = beta * kl_per_seq
    elif kl_mode == "none":
        adv_effective = adv
    else:
        raise ValueError(f"Unknown KL mode: {kl_mode}")

    surrogate_seq = torch.min(ratio * adv_effective, clipped * adv_effective)
    surrogate_abs_mean = surrogate_seq.abs().mean()
    surrogate_pos_frac = (surrogate_seq > 0).float().mean()
    policy_loss = -surrogate_seq
    total_loss = (policy_loss + kl_loss).mean()

    clip_mask = (ratio - clipped).abs() > 1e-8
    clip_count = clip_mask.float().sum()
    reward_mean = adv.mean()
    reward_std = adv.std(unbiased=False) if adv.numel() > 1 else torch.zeros((), device=adv.device)
    adv_effective_mean = adv_effective.mean()
    adv_effective_std = adv_effective.std(unbiased=False) if adv_effective.numel() > 1 else torch.zeros((), device=adv_effective.device)
    log_ratio_mean = log_ratio_seq.mean()
    log_ratio_std = log_ratio_seq.std(unbiased=False) if log_ratio_seq.numel() > 1 else torch.zeros((), device=log_ratio_seq.device)
    log_ratio_abs_mean = log_ratio_seq.abs().mean()
    log_ratio_per_tok_mean = (log_ratio_seq / L1).mean()
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
    entropy_mean = (entropy * p_mask).sum() / p_mask.sum().clamp(min=1)
    elbo_mean = logp_new_seq.mean()
    elbo_std = logp_new_seq.std(unbiased=False) if logp_new_seq.numel() > 1 else torch.zeros((), device=logp_new_seq.device)

    metrics = {
        "loss/total": total_loss.detach(),
        "loss/policy": policy_loss.mean().detach(),
        "loss/kl": kl_loss.mean().detach(),
        "policy/surr_abs_mean": surrogate_abs_mean.detach(),
        "policy/surr_pos_frac": surrogate_pos_frac.detach(),
        "kl/mean": kl_mean.detach(),
        "kl/std": kl_std.detach(),
        "policy/entropy": entropy_mean.detach(),
        "log_ratio/seq_mean": log_ratio_mean.detach(),
        "log_ratio/seq_std": log_ratio_std.detach(),
        "log_ratio/seq_abs_mean": log_ratio_abs_mean.detach(),
        "log_ratio/per_tok_mean": log_ratio_per_tok_mean.detach(),
        "adv_eff/mean": adv_effective_mean.detach(),
        "adv_eff/std": adv_effective_std.detach(),
        "adv/mean": reward_mean.detach(),
        "adv/std": reward_std.detach(),
        "clip_count": clip_count.detach(),
        "clip_total": torch.tensor(float(B), device=device),
        "elbo/mean": elbo_mean.detach(),
        "elbo/std": elbo_std.detach(),
    }
    return total_loss, metrics

def _move_optional_extended_prev(batch: Dict[str, Any], device: torch.device) -> None | Tensor:
    if batch.get("extended_prev") is not None:
       return batch["extended_prev"].to(device)
    return None


def _move_optional_prev_p_mask(batch: Dict[str, Any], device: torch.device) -> None | Tensor:
    if batch.get("prev_p_mask") is not None:
        return batch["prev_p_mask"].to(device)
    return None


class DefaultMethodHandler(ABC):
    """Shared train_step for token-level PPO-style forward + optional extended_prev in batch."""

    def build(self, ctx: RLBuildContext) -> Tuple[Dataset, Callable[[list], Dict[str, Any]]]:
        B, L = ctx.input_ids_lm.shape
        L0    = ctx.start_pos
        L1    = L - L0
        block_size = ctx.config.training.block_size
        for b in range(B):
            step_map_i = ctx.step_map_list[b]
            for j in range(int((L1 - 1) / block_size) + 1):
                start = j * block_size
                end = min(L1, (j + 1) * block_size)
                ctx.step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], ctx.config.training.shrink)
        step_map = torch.as_tensor(ctx.step_map_list, dtype=torch.long)

        assert step_map.shape[1] == L1
        extended_input_ids_list = []
        p_mask_list = []
        reward_list = []
        for b in range(B):
            step_b = step_map[b]
            while True:
                out = one_round_vectorized(
                    input_ids_b=ctx.input_ids_lm[b],
                    step_map_b=step_b,
                    L0=L0,
                    L1=L1,
                    block_size=block_size,
                    mask_id=ctx.mask_id)
                extended_b, pmask_b, step_b, has_any = out
                if not has_any:
                    break
                extended_input_ids_list.append(extended_b)
                p_mask_list.append(pmask_b)
                reward_list.append(ctx.reward_list[b])
        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask = torch.stack(p_mask_list, dim=0).to(torch.bool)
        labels = extended_input_ids[:, :L].clone()
        idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
        valid = (idx >= ctx.start_pos) | extended_input_ids[:, :L].ne(ctx.pad_id)
        tok_idx = valid.long().cumsum(dim=-1) - 1
        tok_idx = tok_idx.masked_fill(~valid, 1)
        tok_idx_resp = tok_idx[:, ctx.start_pos:]
        tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)
        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx  = keep.nonzero(as_tuple=True)[0]          # LongTensor of indices
        extended_input_ids = extended_input_ids[idx]
        p_mask = p_mask[idx]
        tok_idx_ext = tok_idx_ext[idx]
        labels = labels[idx]
        reward_list = [reward_list[i] for i in idx.tolist()]
        dataset = TrainDataset(extended_input_ids, p_mask, tok_idx_ext, labels, reward_list)
        return dataset, TrainDataset.simple_collate

    @property
    def logp_old_is_seq(self) -> bool:
        return False

    def logp(self, ctx: ForwardProcessContext, model) -> Tensor:
        logits = model(input_ids=ctx.extended_input_ids, attention_mask=ctx.attention_mask, position_ids=ctx.tok_idx_ext).logits
        logits = torch.cat([logits[:, :ctx.L0, :], logits[:, ctx.L0 + ctx.L1 :, :]], dim=1)
        log_probs = F.log_softmax(logits, dim=-1)
        logp_tok = log_probs.gather(dim=-1, index=ctx.labels.unsqueeze(-1)).squeeze(-1)
        return logp_tok

    @torch.no_grad()
    def compute_logp_old(self, tctx: RLTrainStepContext) -> None:
        """Default old-logp precompute; method-specific handlers can override."""
        accelerator = tctx.accelerator
        dataset = tctx.dataset_lm
        train_dataloader_lm = tctx.train_dataloader_lm
        model = tctx.model
        L0 = tctx.start_pos
        L1 = tctx.response_length
        basic_block_attention = tctx.basic_block_attention
        process_pad = partial(process_pad_fn, start_pos=L0, pad_id=tctx.pad_id, L0=L0, L1=L1)
        # get_elbo = tctx.get_elbo

        model.eval()
        for batch in train_dataloader_lm:
            ids = batch["ids"]
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            B, L = p_mask.shape
            device = extended_input_ids.device

            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids)

            fwd_ctx = ForwardProcessContext(extended_input_ids=extended_input_ids, 
            p_mask=p_mask, tok_idx_ext=tok_idx_ext, labels=labels, 
            adv=None, logp_old=None, attention_mask=attention_mask, L0=L0, L1=L1)

            # logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
            # logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)
            if self.logp_old_is_seq:
                # elbo = get_elbo(labels[:, L0:], logits[:, L0:], mask_ratio[:, L0:], p_mask[:, L0:])
                # dataset.logp_old[ids] = elbo.float().cpu()
                raise NotImplementedError("Sequence-level logp_old is not implemented for DUEL")
            else:
                # log_probs = F.log_softmax(logits, dim=-1)
                # logp_tok = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                # dataset.logp_old_tok[ids] = logp_tok.float().cpu()
                logp_tok = self.logp(fwd_ctx, model)
                dataset.logp_old[ids] = logp_tok.float().cpu()
        accelerator.wait_for_everyone()
        model.train()

    def train_step(self, tctx: RLTrainStepContext, batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]:
        device = tctx.accelerator.device
        extended_input_ids = batch["extended_input_ids"].to(device)
        p_mask = batch["p_mask"].to(device)
        tok_idx_ext = batch["tok_idx_ext"].to(device)
        labels = batch["labels"].to(device)
        reward = batch["reward"]
        # mask_ratio = batch["mask_ratio"].to(device)
        # extended_prev = _move_optional_extended_prev(batch, device)
        L0 = tctx.start_pos
        L1 = tctx.response_length
        fwd_ctx = ForwardProcessContext(extended_input_ids=extended_input_ids, 
        p_mask=p_mask, tok_idx_ext=tok_idx_ext, labels=labels, 
        adv=reward, logp_old=None, attention_mask=None, L0=L0, L1=L1)
        old_lp = tctx.dataset_lm.logp_old[batch["ids"].cpu()].to(device)
        if torch.isneginf(old_lp).any().item():
            print(old_lp)
        return self.forward_process(fwd_ctx, batch, old_lp)
    

    def forward_process(self, tctx: RLTrainStepContext, batch: Dict[str, Any], old_lp: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        config = tctx.config
        p_mask = tctx.p_mask

        
        adv = torch.as_tensor(adv, device=tctx.extended_input_ids.device).detach()

        B, L = tctx.p_mask.shape
        L0    = tctx.start_pos
        L1    = L - L0
        device = tctx.extended_input_ids.device

        attention_mask = tctx.basic_block_attention.clone()
        attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
        attention_mask = process_pad_fn(attention_mask, tctx.extended_input_ids)

        logits = tctx.model(input_ids = tctx.extended_input_ids, attention_mask = attention_mask, position_ids = tctx.tok_idx_ext).logits
        logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)  # (B, L0+L1, V)

        log_probs = F.log_softmax(logits, dim=-1)
        
        logp_new_tok  = log_probs.gather(dim=-1, index=tctx.labels.unsqueeze(-1)).squeeze(-1)     # (B, T)
        
        logp_old_tok = old_lp
        ratio   = logp_new_tok - logp_old_tok
        ratio = torch.where(p_mask, ratio, torch.zeros_like(ratio)).clamp(-10.0, 10.0) # make it stable, inf * 0 -> none
        ratio   = torch.exp(ratio)          # (B, T)
        clipped = torch.clamp(ratio, 1 - config.training.eps, 1 + config.training.eps)            # (B, T)

        adv_tok = adv.unsqueeze(1)

        surrogate_tok = torch.min(ratio * adv_tok, clipped * adv_tok)  # (B, T)
        surrogate_tok = surrogate_tok * p_mask

        num_mask = torch.clamp(p_mask.sum(dim=1), min=1)
        surrogate_tok = surrogate_tok.sum(dim=1) / L1

        policy_loss = - (surrogate_tok.sum() / B)

        # KL penalty (optional)
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if config.training.beta > 0:
            kl_seq = logp_new_tok - logp_old_tok
            kl_seq = torch.where(p_mask, kl_seq, torch.zeros_like(kl_seq))
            if config.training.use_kl_estimator_k3:
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * p_mask).sum(dim=1) / L1
            kl_loss = config.training.beta * kl_seq.sum() / B
            total_loss = policy_loss + kl_loss
        else:
            total_loss = policy_loss
        metrics = TrainStepMetrics(
            tot_loss=total_loss.detach(),
            policy_loss=policy_loss.detach(),
            kl_loss=kl_loss.detach(),
            kl_mean=kl_seq.mean().detach(),
            kl_std=kl_seq.std(unbiased=False).detach(),
            log_ratio_mean=log_ratio_mean,
            log_ratio_std=log_ratio_std,
        )
        return total_loss, self.metric_to_dict(metrics) 

    def metric_to_dict(self, metrics: TrainStepMetrics) -> Dict[str, float]:
        return {
            "loss/total": metrics.tot_loss.item(),
            "loss/policy": metrics.policy_loss.item(),
            "loss/kl": metrics.kl_loss.item(),
            "kl/mean": metrics.kl_mean.item(),
            "kl/std": metrics.kl_std.item(),
            "log_ratio/seq_mean": metrics.log_ratio_mean.item(),
            "log_ratio/seq_std": metrics.log_ratio_std.item(),
            "log_ratio/seq_abs_mean": metrics.log_ratio_abs_mean.item(),
            "log_ratio/per_tok_mean": metrics.log_ratio_per_tok_mean.item(),
            "adv_eff/mean": metrics.adv_effective_mean.item(),
            "adv_eff/std": metrics.adv_effective_std.item(),
            "adv/mean": metrics.reward_mean.item(),
            "adv/std": metrics.reward_std.item(),
            "clip_count": metrics.clip_count.item(),
            "clip_total": metrics.clip_total.item(),
            "elbo/mean": metrics.elbo_mean.item(),
            "elbo/std": metrics.elbo_std.item(),
            "policy/entropy": metrics.entropy_mean.item(),
            "policy/surr_abs_mean": metrics.surrogate_abs_mean.item(),
            "policy/surr_pos_frac": metrics.surrogate_pos_frac.item(),
            "adv_eff/mean": metrics.adv_effective_mean.item(),
            "adv_eff/std": metrics.adv_effective_std.item(),
            "adv/mean": metrics.reward_mean.item(),
            "adv/std": metrics.reward_std.item(),
            "clip_count": metrics.clip_count.item(),
            "clip_total": metrics.clip_total.item(),
        }


class ESPOForwardHandler(DefaultMethodHandler):
    """Same data path as default; sequence-level forward + logp_old_seq."""

    @property
    def logp_old_is_seq(self) -> bool:
        return True

        

class DUELHandler(DefaultMethodHandler):
    """DUEL + theory mask_ratio; dataset lives in rl_sdar_duel_theory."""

    def build(self, ctx: RLBuildContext) -> Tuple[Dataset, Callable[[list], Dict[str, Any]]]:
        B, L = ctx.input_ids_lm.shape
        L0 = ctx.start_pos
        L1 = L - L0
        block_size = ctx.config.training.block_size
        
        for b in range(B):
            step_map_i = ctx.step_map_list[b]
            for j in range(int((L1 - 1) / block_size) + 1):
                start = j * block_size
                end = min(L1, (j + 1) * block_size)
                ctx.step_map_list[b][start:end] = ctx.collapse_k_unique(step_map_i[start:end], ctx.config.training.shrink)
        step_map = torch.as_tensor(ctx.step_map_list, dtype=torch.long)

        # Collect per-sample multi-step trajectories.
        extended_input_ids_list = []
        p_mask_list = []
        for b in range(B):
            step_b = step_map[b]
            extended_input_ids_b = []
            p_mask_b = []
            while True:
                out = one_round_vectorized_strict_block_serial(
                    input_ids_b=ctx.input_ids_lm[b],
                    step_map_b=step_b,
                    L0=L0,
                    L1=L1,
                    block_size=block_size,
                    mask_id=ctx.mask_id,
                )
                y_beforet_t_b, delta_t_b, step_b, has_any = out
                if not has_any:
                    break
                extended_input_ids_b.append(y_beforet_t_b)
                p_mask_b.append(delta_t_b)
            extended_input_ids_list.append(extended_input_ids_b)
            p_mask_list.append(p_mask_b)

        # Build 3D tensors: (batch, step, position), with step padding.
        max_steps = max((len(x) for x in extended_input_ids_list), default=0)
        ext_pos = L0 + 2 * L1
        p_mask_pos = L0 + L1
        extended_input_ids = torch.full(
            (B, max_steps, ext_pos),
            fill_value=ctx.pad_id,
            dtype=ctx.input_ids_lm.dtype,
        )
        p_mask = torch.zeros((B, max_steps, p_mask_pos), dtype=torch.bool)
        labels = ctx.input_ids_lm.clone()
        idx = torch.arange(L).unsqueeze(0).expand(B, -1)
        valid_idx = (idx >= ctx.start_pos) | ctx.input_ids_lm.ne(ctx.pad_id)
        tok_idx = valid_idx.long().cumsum(dim=-1) - 1
        tok_idx = tok_idx.masked_fill(~valid_idx, 1)
        tok_idx_ext = torch.cat([tok_idx, tok_idx[ctx.start_pos:]], dim=0)

        
        valid = torch.zeros((B, max_steps), dtype=torch.bool)

        for b in range(B):
            n_steps = len(extended_input_ids_list[b])
            if n_steps == 0:
                continue
            extended_input_ids[b, :n_steps] = torch.stack(extended_input_ids_list[b], dim=0)
            p_mask[b, :n_steps] = torch.stack(p_mask_list[b], dim=0)
            valid[b, :n_steps] = True

        reward = torch.as_tensor(ctx.reward_list, dtype=torch.float32)
        dataset = DUELTrainDataset(
            extended_input_ids=extended_input_ids,
            p_mask=p_mask,
            tok_idx_ext=tok_idx_ext,
            labels=labels,
            reward=reward,
            valid=valid,
        )
        return dataset, DUELTrainDataset.collate

    def logp(self, ctx: ForwardProcessContext, model) -> Tensor:
        

    def compute_logp_old(self, tctx: RLTrainStepContext) -> None:
        accelerator = tctx.accelerator
        dataset = tctx.dataset_lm
        train_dataloader_lm = tctx.train_dataloader_lm
        model = tctx.model
        L0 = tctx.start_pos
        L1 = tctx.response_length
        basic_block_attention = tctx.basic_block_attention
        process_pad = partial(process_pad_fn, start_pos=L0, pad_id=tctx.pad_id, L0=L0, L1=L1)
        # get_elbo = tctx.get_elbo

        model.eval()
        for batch in train_dataloader_lm:
            ids = batch["ids"]
            extended_input_ids = batch["extended_input_ids"].to(accelerator.device)
            p_mask = batch["p_mask"].to(accelerator.device)
            tok_idx_ext = batch["tok_idx_ext"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)

            B = p_mask.shape[0]
            device = extended_input_ids.device

            attention_mask = basic_block_attention.clone()
            attention_mask = attention_mask.repeat_interleave(B, dim=0).to(device)
            attention_mask = process_pad(attention_mask, extended_input_ids)

            fwd_ctx = ForwardProcessContext(extended_input_ids=extended_input_ids, 
            p_mask=p_mask, tok_idx_ext=tok_idx_ext, labels=labels, 
            adv=None, logp_old=None, attention_mask=attention_mask, L0=L0, L1=L1)

            # logits = model(input_ids=extended_input_ids, attention_mask=attention_mask, position_ids=tok_idx_ext).logits
            # logits = torch.cat([logits[:, :L0, :], logits[:, L0 + L1 :, :]], dim=1)
            if self.logp_old_is_seq:
                # elbo = get_elbo(labels[:, L0:], logits[:, L0:], mask_ratio[:, L0:], p_mask[:, L0:])
                # dataset.logp_old[ids] = elbo.float().cpu()
                raise NotImplementedError("Sequence-level logp_old is not implemented for DUEL")
            else:
                # log_probs = F.log_softmax(logits, dim=-1)
                # logp_tok = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
                # dataset.logp_old_tok[ids] = logp_tok.float().cpu()
                logp_tok = self.logp(fwd_ctx, model)
                dataset.logp_old[ids] = logp_tok.float().cpu()
        accelerator.wait_for_everyone()
        model.train()



_METHOD_REGISTRY: Dict[str, RLSDARMethodHandler] = {}


def register_rl_sdar_method(name: str, handler: RLSDARMethodHandler) -> None:
    if name in _METHOD_REGISTRY:
        raise KeyError(f"RL-SDAR method {name!r} is already registered")
    _METHOD_REGISTRY[name] = handler


def get_rl_sdar_method_handler(name: str) -> RLSDARMethodHandler:
    try:
        return _METHOD_REGISTRY[name]
    except KeyError as e:
        known = ", ".join(sorted(_METHOD_REGISTRY))
        raise KeyError(f"Unknown training.method={name!r}. Registered: {known}") from e


def registered_rl_sdar_methods() -> Tuple[str, ...]:
    return tuple(sorted(_METHOD_REGISTRY))


def _register_builtin_methods() -> None:
    _tok = DefaultMethodHandler()
    _espo = ESPOForwardHandler()
    _duel = DUELHandler()
    for n in ("random_masking", "coupled", "TraceRL"):
        register_rl_sdar_method(n, _tok)
    register_rl_sdar_method("TraceESPO", _espo)
    register_rl_sdar_method("ESPO", _espo)
    register_rl_sdar_method("DUEL", _duel)


_register_builtin_methods()
