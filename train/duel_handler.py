import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer


@dataclass
class PairYTData:
    prompt: str
    response: str
    response_token_ids: torch.Tensor
    step_map: torch.Tensor
    step_values: torch.Tensor
    y_t: torch.Tensor
    pred_mask_t: torch.Tensor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
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
    data: List[Dict[str, Any]],
    tokenizer: Any,
    mask_token_id: int,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for row in data:
        prompt = row["prompt"]
        response = row["response"]
        step_map_raw = row.get("step_map", [])

        response_ids = tokenizer.encode(response, add_special_tokens=False)
        step_map = _normalize_step_map(step_map_raw, len(response_ids))

        response_ids_t = torch.tensor(response_ids, dtype=torch.long)
        step_map_t = torch.tensor(step_map, dtype=torch.long)

        yt_pack = build_yt_from_step_map(
            response_token_ids=response_ids_t,
            step_map=step_map_t,
            mask_token_id=mask_token_id,
        )

        row_data = PairYTData(
            prompt=prompt,
            response=response,
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
        description="Collect y_t tensors for each (prompt, response) from rl_data.json."
    )
    parser.add_argument("--input_json", required=True, help="Path to rl_data.json")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name/path")
    parser.add_argument("--output_pt", required=True, help="Output .pt path")
    parser.add_argument(
        "--mask_token_id",
        type=int,
        default=None,
        help="Optional override for mask token id",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    mask_token_id = args.mask_token_id
    if mask_token_id is None:
        if tokenizer.mask_token_id is None:
            raise ValueError(
                "Tokenizer has no mask_token_id. Please pass --mask_token_id explicitly."
            )
        mask_token_id = int(tokenizer.mask_token_id)

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = collect_pair_yt_tensors(data, tokenizer, mask_token_id)
    torch.save(rows, args.output_pt)
    print(f"Saved {len(rows)} samples to: {args.output_pt}")


if __name__ == "__main__":
    main()
