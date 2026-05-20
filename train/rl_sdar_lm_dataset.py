"""
Shared LM dataset construction for RL-SDAR: JSON → tokenize → method_handler.build.

Used by `rl_sdar.py` training and `collect_data_test.py` debugging.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Optional, Tuple, Type

import torch
from torch.utils.data import Dataset

from train.rl_sdar_method_registry import RLBuildContext, get_rl_sdar_method_handler


def simple_collate(batch):
    idx, extended_input_ids, p_mask, tok_idx_ext, labels, reward, mask_ratio = zip(*batch)
    return {
        "ids": torch.tensor(idx),
        "extended_input_ids": torch.stack(extended_input_ids),
        "p_mask": torch.stack(p_mask),
        "tok_idx_ext": torch.stack(tok_idx_ext),
        "labels": torch.stack(labels),
        "reward": reward,
        "mask_ratio": torch.stack(mask_ratio),
    }


def build_rl_sdar_lm_dataset(
    config: Any,
    *,
    project_name: str,
    uni_prompting: Any,
    mask_id: int,
    pad_id: int,
    train_dataset_class: Type[Dataset],
    max_samples: Optional[int] = None,
) -> Tuple[Dataset, Callable[[list], Dict[str, Any]], Dict[str, Any]]:
    """Load rollout JSON, tokenize, run registered method `build`, return dataset + collate + meta."""

    json_path = "./" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json"
    with open(json_path, "r") as f:
        dataset_load = json.load(f)
    if max_samples is not None:
        dataset_load = dataset_load[: max_samples]

    prompt_list = []
    response_list = []
    step_map_list = []
    reward_list = []
    for x in dataset_load:
        prompt_list.append(x["prompt"])
        response_list.append(x["response"])
        reward_list.append(x["reward"])

    input_ids_lm, _, start_pos, drop_num = uni_prompting((prompt_list, response_list))

    _, L = input_ids_lm.shape
    L0 = start_pos
    L1 = L - L0
    post_num = config.training.get("post_num", None)

    for x in dataset_load:
        if "step_map" not in x.keys():
            step_map_list.append([j for j in range(L1)])
        else:
            step_map_i = x["step_map"]
            if len(step_map_i) > L1:
                step_map_i = step_map_i[:L1]
            else:
                step_map_i = step_map_i + [max(step_map_i) + 1] * (L1 - len(step_map_i))
            step_map_list.append(step_map_i)

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

    def one_round_vectorized(input_ids_b, step_map_b, L0, L1, block_size, mask_id):
        device = input_ids_b.device
        NB = (L1 + block_size - 1) // block_size
        pad_len = NB * block_size - L1

        step_pad = torch.full((NB * block_size,), -1, dtype=torch.long, device=device)
        step_pad[:L1] = step_map_b
        step_blk = step_pad.view(NB, block_size)

        valid = step_blk.ge(0)
        big = torch.iinfo(step_blk.dtype).max
        tmp = step_blk.masked_fill(~valid, big)
        min_vals, _ = tmp.min(dim=1, keepdim=True)

        pmask_blk = step_blk.eq(min_vals) & valid
        if not pmask_blk.any():
            return None, None, step_map_b, False

        ge_mask_blk = step_blk.ge(min_vals) & valid

        pmask_tail = pmask_blk.view(-1)[:L1]
        ge_mask_tail = ge_mask_blk.view(-1)[:L1]

        pmask_b = torch.zeros(L0 + L1, dtype=torch.bool, device=device)
        pmask_b[L0:] = pmask_tail

        tail = input_ids_b[L0 : L0 + L1].clone()
        tail[ge_mask_tail] = mask_id

        extended_input_ids_b = torch.empty(L0 + L1 + L1, dtype=input_ids_b.dtype, device=device)
        extended_input_ids_b[: L0 + L1] = input_ids_b
        extended_input_ids_b[L0 + L1 :] = tail

        new_step_map_b = step_map_b.clone()
        new_step_map_b[pmask_tail] = -1

        return extended_input_ids_b, pmask_b, new_step_map_b, True

    def get_mask(time_epsilon, B, L1, device, seed=2026):
        if not hasattr(get_mask, "_generators"):
            get_mask._generators = {}
        device_key = str(device)
        if device_key not in get_mask._generators:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
            get_mask._generators[device_key] = gen
        gen = get_mask._generators[device_key]

        t = time_epsilon + (1 - time_epsilon) * torch.rand(B, device=device, generator=gen)
        p_mask = 1.0 - t.unsqueeze(1).expand(B, L1)
        is_mask = torch.rand(B, L1, device=device, generator=gen) <= p_mask
        return t, p_mask, is_mask

    def collect_training_data(input_ids, step_map_list, reward):

        B, L = input_ids.shape
        L0 = start_pos
        L1 = L - L0
        block_size = config.training.block_size

        lower = config.training.lower_p
        upper = config.training.upper_p

        if config.training.method == "random_masking":

            extended_input_ids_list, pmask_list, reward_list = [], [], []

            for b in range(B):

                reward_list.append(reward[b])

                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                for j in range(int((L1 - 1) / block_size) + 1):

                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)

                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start) : (L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)

                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)

                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)

            mask_ratio = torch.ones(B, L0 + L1, device=input_ids.device, dtype=torch.float)

        if config.training.method == "coupled":

            extended_input_ids_list, pmask_list, reward_list = [], [], []
            coupled_input_ids_list, coupled_pmask_list, coupled_reward_list = [], [], []

            for b in range(B):

                reward_list.append(reward[b])
                coupled_reward_list.append(reward[b])

                extended_input_ids_b = input_ids[b]
                pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                coupled_input_ids_b = input_ids[b]
                coupled_pmask_b = torch.zeros(start_pos, dtype=torch.bool)

                for j in range(int((L1 - 1) / block_size) + 1):

                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)

                    pmask_b_j = torch.rand(end - start) <= torch.empty(end - start).uniform_(lower, upper)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)
                    coupled_pmask_b = torch.cat([coupled_pmask_b, ~pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start) : (L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)

                    coupled_noise_b_j = input_ids[b, (L0 + start) : (L0 + end)].clone()
                    coupled_noise_b_j = coupled_noise_b_j.masked_fill_(~pmask_b_j, mask_id)

                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                    coupled_input_ids_b = torch.cat([coupled_input_ids_b, coupled_noise_b_j], dim=0)

                extended_input_ids_list.append(extended_input_ids_b)
                pmask_list.append(pmask_b)

                coupled_input_ids_list.append(coupled_input_ids_b)
                coupled_pmask_list.append(coupled_pmask_b)

            extended_input_ids_list += coupled_input_ids_list
            pmask_list += coupled_pmask_list
            reward_list += coupled_reward_list

            mask_ratio = torch.ones(2 * B, L0 + L1, device=input_ids.device, dtype=torch.float)

        elif config.training.method == "TraceRL":

            for b in range(B):
                step_map_i = step_map_list[b]

                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)

            step_map = torch.as_tensor(step_map_list, dtype=torch.long)

            assert step_map.shape[1] == L1

            extended_input_ids_list, pmask_list, reward_list = [], [], []
            mask_ratio_list = []
            for b in range(B):
                step_b = step_map[b]
                while True:
                    out = one_round_vectorized(
                        input_ids_b=input_ids[b],
                        step_map_b=step_b,
                        L0=L0,
                        L1=L1,
                        block_size=block_size,
                        mask_id=mask_id,
                    )
                    extended_b, pmask_b, step_b, has_any = out
                    if not has_any:
                        break

                    extended_input_ids_list.append(extended_b)
                    pmask_list.append(pmask_b)
                    reward_list.append(reward[b])

                    t_b = torch.zeros(L0 + L1, device=input_ids.device, dtype=torch.float)
                    for j in range(int((L1 - 1) / block_size) + 1):
                        start = j * block_size
                        end = min(L1, (j + 1) * block_size)
                        block_len = end - start
                        n_pred = pmask_b[L0 + start : L0 + end].sum().item()
                        mask_ratio_block = (n_pred / block_len) if block_len > 0 else 0.0
                        t_val = 1.0 - mask_ratio_block
                        t_b[L0 + start : L0 + end] = t_val
                    mask_ratio_list.append(t_b)
            mask_ratio = torch.stack(mask_ratio_list, dim=0)

        elif config.training.method == "TraceESPO":
            for b in range(B):
                step_map_i = step_map_list[b]
                for j in range(int((L1 - 1) / block_size) + 1):
                    start = j * block_size
                    end = min(L1, (j + 1) * block_size)
                    step_map_list[b][start:end] = collapse_k_unique(step_map_i[start:end], config.training.shrink)

            step_map = torch.as_tensor(step_map_list, dtype=torch.long)
            assert step_map.shape[1] == L1

            extended_input_ids_list, pmask_list, reward_list, mask_ratio_list = [], [], [], []

            for b in range(B):
                step_b = step_map[b]
                while True:
                    out = one_round_vectorized(
                        input_ids_b=input_ids[b],
                        step_map_b=step_b,
                        L0=L0,
                        L1=L1,
                        block_size=block_size,
                        mask_id=mask_id,
                    )
                    extended_b, pmask_b, step_b, has_any = out
                    if not has_any:
                        break

                    extended_input_ids_list.append(extended_b)
                    pmask_list.append(pmask_b)
                    reward_list.append(reward[b])
                    mask_ratio_b = torch.ones(L0 + L1, device=input_ids.device, dtype=torch.float)
                    for j in range(int((L1 - 1) / block_size) + 1):
                        start = j * block_size
                        end = min(L1, (j + 1) * block_size)
                        block_len = end - start
                        n_pred = pmask_b[L0 + start : L0 + end].sum().item()
                        ratio = (n_pred / block_len) if block_len > 0 else 1.0
                        if ratio <= 0:
                            ratio = 1.0
                        mask_ratio_b[L0 + start : L0 + end] = ratio
                    mask_ratio_list.append(mask_ratio_b)

            mask_ratio = torch.stack(mask_ratio_list, dim=0)

        elif config.training.method == "ESPO":
            time_epsilon = config.training.get("time_epsilon", 1e-3)
            mask_ratio = torch.zeros(B, L0, device=input_ids.device)
            p_mask = torch.zeros(B, L0, dtype=torch.bool, device=input_ids.device)
            for j in range(int((L1 - 1) / block_size) + 1):
                start = j * block_size
                end = min(L1, (j + 1) * block_size)
                t, mask_ratio_j, p_mask_j = get_mask(time_epsilon, B, end - start, input_ids.device)
                mask_ratio = torch.cat([mask_ratio, mask_ratio_j], dim=1)
                p_mask = torch.cat([p_mask, p_mask_j], dim=1)
            noisy_input_ids = torch.where(p_mask, input_ids[:, :], mask_id)
            extended_input_ids = torch.cat([input_ids, noisy_input_ids[:, start_pos:]], dim=1)
            extended_input_ids_list = torch.unbind(extended_input_ids, dim=0)
            pmask_list = torch.unbind(p_mask, dim=0)
            reward_list = [reward[b] for b in range(B)]

        else:
            raise NotImplementedError(f"Unknown method: {config.training.method}")

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
        idx = keep.nonzero(as_tuple=True)[0]

        extended_input_ids = extended_input_ids[idx]
        p_mask = p_mask[idx]
        tok_idx_ext = tok_idx_ext[idx]
        labels = labels[idx]
        mask_ratio = mask_ratio[idx]

        reward_list = [reward_list[i] for i in idx.tolist()]

        return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list, mask_ratio

    method_handler = get_rl_sdar_method_handler(config.training.method)
    build_ctx = RLBuildContext(
        input_ids_lm=input_ids_lm,
        step_map_list=step_map_list,
        reward_list=reward_list,
        config=config,
        start_pos=start_pos,
        post_num=post_num,
        mask_id=mask_id,
        pad_id=pad_id,
        collect_training_data=collect_training_data,
        collapse_k_unique=collapse_k_unique,
        one_round_vectorized=one_round_vectorized,
        train_dataset_class=train_dataset_class,
        simple_collate_fn=simple_collate,
    )
    dataset_lm, collate_fn_lm = method_handler.build(build_ctx)

    meta = {
        "start_pos": start_pos,
        "L0": L0,
        "L1": L1,
        "drop_num": drop_num,
        "num_lm_prompts": int(input_ids_lm.shape[0]),
        "json_path": json_path,
        "num_json_rows": len(dataset_load),
    }
    return dataset_lm, collate_fn_lm, meta
