"""
TracePairTheory: optional RL-SDAR data construction (y_t, y_{t-1}, theory mask_ratio).

Kept separate from rl_sdar.py so the original collect / dataset paths stay unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _make_fully_masked_tail_extended(
    input_ids_b: torch.Tensor, L0: int, L1: int, mask_id: int
) -> torch.Tensor:
    ext = torch.empty(L0 + 2 * L1, dtype=input_ids_b.dtype, device=input_ids_b.device)
    ext[: L0 + L1] = input_ids_b[: L0 + L1].clone()
    ext[L0 + L1 :] = mask_id
    return ext


def _theory_mask_ratio_row(
    L0: int,
    L1: int,
    round_i: int,
    total_rounds: int,
    device: torch.device,
    time_eps: float,
    schedule: str,
) -> torch.Tensor:
    row = torch.ones(L0 + L1, device=device, dtype=torch.float32)
    if total_rounds <= 0:
        r = 1.0
    elif schedule == "forward_from_trace":
        tau = float(total_rounds - round_i) / float(max(total_rounds, 1))
        r = time_eps + (1.0 - time_eps) * tau
    elif schedule == "uniform_forward_t":
        j = total_rounds - 1 - round_i
        tau = float(j + 1) / float(total_rounds + 1)
        r = time_eps + (1.0 - time_eps) * tau
    else:
        raise ValueError(f"Unknown training.theory_mask_schedule={schedule}")
    row[L0:] = r
    return row


def finalize_lm_collected_rows(
    extended_input_ids_list: List[torch.Tensor],
    pmask_list: List[torch.Tensor],
    reward_list: List[Any],
    mask_ratio: torch.Tensor,
    L: int,
    L0: int,
    start_pos: int,
    post_num: int | None,
    pad_id: int,
    extended_prev_list: List[torch.Tensor] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor, torch.Tensor | None]:
    """Shared post-process: stack, pad masking, tok_idx_ext, filter empty p_mask rows."""
    extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
    p_mask = torch.stack(pmask_list, dim=0).to(torch.bool)

    pad_resp = (extended_input_ids[:, :L] == pad_id) & p_mask
    if post_num is not None:
        cum_pad = torch.cumsum(pad_resp.int(), dim=1)
        p_mask &= ~(pad_resp & (cum_pad > post_num))

    labels = extended_input_ids[:, :L].clone()

    idx = torch.arange(L).unsqueeze(0).expand(extended_input_ids.shape[0], -1)
    valid = (idx >= start_pos) | extended_input_ids[:, :L].ne(pad_id)
    tok_idx = valid.long().cumsum(dim=-1) - 1
    tok_idx = tok_idx.masked_fill(~valid, 1)
    tok_idx_resp = tok_idx[:, start_pos:]
    tok_idx_ext = torch.cat([tok_idx, tok_idx_resp], dim=1)

    keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
    idx_keep = keep.nonzero(as_tuple=True)[0]

    extended_input_ids = extended_input_ids[idx_keep]
    p_mask = p_mask[idx_keep]
    tok_idx_ext = tok_idx_ext[idx_keep]
    labels = labels[idx_keep]
    mask_ratio = mask_ratio[idx_keep]
    reward_list = [reward_list[i] for i in idx_keep.tolist()]

    extended_prev_out = None
    if extended_prev_list is not None and len(extended_prev_list) > 0:
        extended_prev_out = torch.stack(extended_prev_list, dim=0)[idx_keep]

    return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list, mask_ratio, extended_prev_out


def collect_trace_pair_dataset(
    input_ids: torch.Tensor,
    step_map_list: Sequence[Any],
    reward: Sequence[Any],
    *,
    collapse_k_unique: Callable[[list, int], list],
    one_round_vectorized: Callable[..., tuple],
    config: Any,
    start_pos: int,
    post_num: int | None,
    mask_id: int,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list, torch.Tensor, torch.Tensor]:
    B, L = input_ids.shape
    L0 = start_pos
    L1 = L - L0
    block_size = config.training.block_size
    time_eps = float(config.training.get("time_epsilon", 1e-3))
    theory_sched = config.training.get("theory_mask_schedule", "forward_from_trace")
    shrink = config.training.shrink

    sm_list = [list(row) for row in step_map_list]

    for b in range(B):
        step_map_i = sm_list[b]
        for j in range(int((L1 - 1) / block_size) + 1):
            start = j * block_size
            end = min(L1, (j + 1) * block_size)
            sm_list[b][start:end] = collapse_k_unique(step_map_i[start:end], shrink)

    step_map = torch.as_tensor(sm_list, dtype=torch.long, device=input_ids.device)
    assert step_map.shape[1] == L1

    extended_input_ids_list: List[torch.Tensor] = []
    extended_prev_list: List[torch.Tensor] = []
    pmask_list: List[torch.Tensor] = []
    reward_list_out: List[Any] = []
    mask_ratio_list: List[torch.Tensor] = []

    for b in range(B):
        step_b = step_map[b].clone()
        prev_ext = _make_fully_masked_tail_extended(input_ids[b], L0, L1, mask_id)
        round_ext: List[torch.Tensor] = []
        round_prev: List[torch.Tensor] = []
        round_p: List[torch.Tensor] = []
        round_rew: List[Any] = []
        while True:
            extended_b, pmask_b, step_b, has_any = one_round_vectorized(
                input_ids_b=input_ids[b],
                step_map_b=step_b,
                L0=L0,
                L1=L1,
                block_size=block_size,
                mask_id=mask_id,
            )
            if not has_any:
                break
            round_prev.append(prev_ext.clone())
            round_ext.append(extended_b.clone())
            round_p.append(pmask_b.clone())
            round_rew.append(reward[b])
            prev_ext = extended_b.clone()

        total_r = len(round_ext)
        for i in range(total_r):
            mr = _theory_mask_ratio_row(L0, L1, i, total_r, input_ids.device, time_eps, theory_sched)
            extended_prev_list.append(round_prev[i])
            extended_input_ids_list.append(round_ext[i])
            pmask_list.append(round_p[i])
            reward_list_out.append(round_rew[i])
            mask_ratio_list.append(mr)

    mask_ratio = torch.stack(mask_ratio_list, dim=0)

    extended_input_ids, p_mask, tok_idx_ext, labels, reward_list_out, mask_ratio, extended_prev = (
        finalize_lm_collected_rows(
            extended_input_ids_list,
            pmask_list,
            reward_list_out,
            mask_ratio,
            L,
            L0,
            start_pos,
            post_num,
            pad_id,
            extended_prev_list,
        )
    )
    if extended_prev is None:
        raise RuntimeError("TracePair produced no rows (empty trace); check step_map / input.")
    return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list_out, mask_ratio, extended_prev


class TracePairTrainDataset(Dataset):
    """Same storage as TrainDataset plus extended_prev (y_{t-1}); logp buffers match row count."""

    def __init__(
        self,
        extended_input_ids: torch.Tensor,
        extended_prev: torch.Tensor,
        p_mask: torch.Tensor,
        tok_idx_ext: torch.Tensor,
        labels: torch.Tensor,
        reward: list,
        mask_ratio: torch.Tensor,
    ):
        self.extended_input_ids = extended_input_ids
        self.extended_prev = extended_prev
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward = reward
        self.mask_ratio = mask_ratio
        self.logp_old_tok = torch.full((len(extended_input_ids), p_mask.shape[1]), float("-inf"))
        self.logp_old_seq = torch.full((len(extended_input_ids),), float("-inf"))

    def __len__(self):
        return len(self.extended_input_ids)

    def __getitem__(self, idx):
        return (
            idx,
            self.extended_input_ids[idx],
            self.extended_prev[idx],
            self.p_mask[idx],
            self.tok_idx_ext[idx],
            self.labels[idx],
            self.reward[idx],
            self.mask_ratio[idx],
        )


def simple_collate_trace_pair_theory(batch):
    idx, extended_input_ids, extended_prev, p_mask, tok_idx_ext, labels, reward, mask_ratio = zip(*batch)
    return {
        "ids": torch.tensor(idx),
        "extended_input_ids": torch.stack(extended_input_ids),
        "extended_prev": torch.stack(extended_prev),
        "p_mask": torch.stack(p_mask),
        "tok_idx_ext": torch.stack(tok_idx_ext),
        "labels": torch.stack(labels),
        "reward": reward,
        "mask_ratio": torch.stack(mask_ratio),
    }


def log_trace_pair_enabled() -> None:
    logger.info(
        "TracePairTheory: rows include extended_prev=y_{t-1}, extended_input_ids=y_t, "
        "mask_ratio from training.theory_mask_schedule. Default loss still uses y_t only."
    )
