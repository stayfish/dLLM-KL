import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch


@dataclass
class PairYTData:
    prompt: str | None
    response: str | None
    sample_idx: int
    response_token_ids: torch.Tensor
    step_map: torch.Tensor
    step_values: torch.Tensor
    y_t: torch.Tensor
    pred_mask_t: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "sample_idx": self.sample_idx,
            "response_token_ids": self.response_token_ids,
            "step_map": self.step_map,
            "step_values": self.step_values,
            "y_t": self.y_t,
            "pred_mask_t": self.pred_mask_t,
        }


def _normalize_step_map(step_map: List[int], target_len: int) -> List[int]:
    if target_len <= 0:
        return []
    if len(step_map) == target_len:
        return step_map
    if len(step_map) > target_len:
        return step_map[:target_len]
    if not step_map:
        return list(range(target_len))
    pad_val = max(step_map) + 1
    return step_map + [pad_val] * (target_len - len(step_map))


def build_yt_from_step_map(
    response_token_ids: torch.Tensor,
    step_map: torch.Tensor,
    mask_token_id: int,
) -> Dict[str, torch.Tensor]:
    # step_values: sorted unique rollout steps for this sample.
    step_values = torch.unique(step_map, sorted=True)
    l = response_token_ids.shape[0]
    t = step_values.shape[0]

    # y_t[t_idx, pos] = token if step_map[pos] < current_step else mask.
    # This matches the noisy state definition used in one_round_vectorized.
    y_t = torch.full((t, l), fill_value=mask_token_id, dtype=response_token_ids.dtype)
    pred_mask_t = torch.zeros((t, l), dtype=torch.bool)
    for i, cur_step in enumerate(step_values):
        visible = step_map < cur_step
        pred_now = step_map == cur_step
        y_t[i, visible] = response_token_ids[visible]
        pred_mask_t[i] = pred_now

    return {"step_values": step_values, "y_t": y_t, "pred_mask_t": pred_mask_t}


def collect_pair_yt_tensors(
    data: Sequence[Dict[str, Any]],
    input_ids: torch.Tensor,
    start_pos: int,
    mask_token_id: int,
) -> List[Dict[str, Any]]:
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be rank-2 tensor, got shape={tuple(input_ids.shape)}")
    bsz, total_len = input_ids.shape
    if not (0 <= start_pos < total_len):
        raise ValueError(f"start_pos must be in [0, {total_len - 1}], got {start_pos}")
    if len(data) != bsz:
        raise ValueError(f"len(data)={len(data)} must equal input_ids.shape[0]={bsz}")

    l1 = total_len - start_pos
    all_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(data):
        prompt = row.get("prompt")
        response = row.get("response")
        step_map_raw = row.get("step_map", [])
        step_map = _normalize_step_map(step_map_raw, l1)

        response_ids_t = input_ids[i, start_pos:].detach().cpu().to(torch.long)
        step_map_t = torch.tensor(step_map, dtype=torch.long)

        yt_pack = build_yt_from_step_map(
            response_token_ids=response_ids_t,
            step_map=step_map_t,
            mask_token_id=mask_token_id,
        )

        row_data = PairYTData(
            prompt=prompt,
            response=response,
            sample_idx=i,
            response_token_ids=response_ids_t,
            step_map=step_map_t,
            step_values=yt_pack["step_values"],
            y_t=yt_pack["y_t"],
            pred_mask_t=yt_pack["pred_mask_t"],
        )
        all_rows.append(row_data.to_dict())
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect y_t tensors aligned to collect_training_data(input_ids, step_map_list, ...)."
    )
    parser.add_argument("--input_json", required=True, help="Path to rl_data.json containing step_map list")
    parser.add_argument("--input_ids_pt", required=True, help="Path to torch tensor .pt, shape [B, L]")
    parser.add_argument("--start_pos", required=True, type=int, help="L0 in collect_training_data")
    parser.add_argument("--output_pt", required=True, help="Output .pt path")
    parser.add_argument(
        "--mask_token_id",
        type=int,
        required=True,
        help="Mask token id used to build y_t",
    )
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    input_ids = torch.load(args.input_ids_pt, map_location="cpu")
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError(f"input_ids file must contain a torch.Tensor, got {type(input_ids)}")

    rows = collect_pair_yt_tensors(
        data=data,
        input_ids=input_ids,
        start_pos=args.start_pos,
        mask_token_id=int(args.mask_token_id),
    )
    torch.save(rows, args.output_pt)
    print(f"Saved {len(rows)} samples to: {args.output_pt}")


if __name__ == "__main__":
    main()
