from train.utils import get_config
from train.prompting_utils import UniversalPrompting
from transformers import AutoTokenizer
import json
from train.rl_sdar_method_registry import get_rl_sdar_method_handler
from train.rl_sdar_method_registry import RLBuildContext
# from train.rl_sdar_method_registry import TrainDataset
import torch
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    def __init__(self, extended_input_ids, p_mask, tok_idx_ext, labels, reward, mask_ratio):
        self.extended_input_ids = extended_input_ids
        self.p_mask = p_mask
        self.tok_idx_ext = tok_idx_ext
        self.labels = labels
        self.reward   = reward
        self.mask_ratio = mask_ratio
        self.logp_old_tok = torch.full(
            (len(extended_input_ids), p_mask.shape[1]), 
            float('-inf')
        )
        self.logp_old_seq = torch.full(
            (len(extended_input_ids),), float('-inf')
        )

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
            self.mask_ratio[idx],
        )


def simple_collate(batch):
    idx, extended_input_ids, p_mask, tok_idx_ext, labels, reward, mask_ratio = zip(*batch)
    return {
        "ids":        torch.tensor(idx),
        "extended_input_ids":  torch.stack(extended_input_ids),
        "p_mask":  torch.stack(p_mask),
        "tok_idx_ext":  torch.stack(tok_idx_ext),
        "labels":  torch.stack(labels),
        "reward":     reward,
        "mask_ratio": torch.stack(mask_ratio),
    }

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


def main():

    def collect_training_data(input_ids, step_map_list, reward):

        B, L = input_ids.shape
        L0    = start_pos
        L1    = L - L0
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
                    #pmask_b_j = torch.rand(end - start) <= torch.linspace(lower, upper, steps=end - start)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
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
                    #pmask_b_j = torch.rand(end - start) <= torch.linspace(lower, upper, steps=end - start)
                    pmask_b = torch.cat([pmask_b, pmask_b_j], dim=0)
                    coupled_pmask_b = torch.cat([coupled_pmask_b, ~pmask_b_j], dim=0)

                    noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    noise_b_j = noise_b_j.masked_fill_(pmask_b_j, mask_id)

                    coupled_noise_b_j = input_ids[b, (L0 + start):(L0 + end)].clone()
                    coupled_noise_b_j = coupled_noise_b_j.masked_fill_(~pmask_b_j, mask_id)

                    extended_input_ids_b = torch.cat([extended_input_ids_b, noise_b_j], dim=0)
                    coupled_input_ids_b  = torch.cat([coupled_input_ids_b, coupled_noise_b_j], dim=0)
                
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

        
            # 与 TraceRL 相同的多轮 step_map 生成，同时为每条样本构造 mask_ratio 供 ESPO loss 使用。
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
                    # mask_ratio: 每个分块内「预测位置数/该块长度」，与 ESPO get_elbo 的按块加权一致
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
        else:
            raise NotImplementedError(f"Unknown method: {config.training.method}")

        extended_input_ids = torch.stack(extended_input_ids_list, dim=0)
        p_mask =  torch.stack(pmask_list, dim=0).to(torch.bool)
        
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
        tok_idx_ext  = torch.cat([tok_idx, tok_idx_resp], dim=1)

        keep = p_mask.view(p_mask.size(0), -1).any(dim=1)
        idx  = keep.nonzero(as_tuple=True)[0]          # LongTensor of indices
        # idx: used to delete p_mask (with no mask positions)

        extended_input_ids = extended_input_ids[idx]
        p_mask            = p_mask[idx]
        tok_idx_ext       = tok_idx_ext[idx]
        labels            = labels[idx]
        mask_ratio        = mask_ratio[idx]

        reward_list = [reward_list[i] for i in idx.tolist()]

        return extended_input_ids, p_mask, tok_idx_ext, labels, reward_list, mask_ratio
       

    config = get_config()
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model, trust_remote_code=True)
    uni_prompting = UniversalPrompting(tokenizer, max_prompt_len=config.training.max_prompt_len, max_gen_length=config.training.max_gen_length, ignore_id=-100)

    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id

    
    with open("./k3_tok_in_reward/temp_data/" + config.dataset.optimization_data + ".json", 'r') as f:
        dataset_load = json.load(f)

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
    L0    = start_pos
    L1    = L - L0
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
    method_handler = get_rl_sdar_method_handler("DUEL")
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
        train_dataset_class=TrainDataset,
        simple_collate_fn=simple_collate,
    )
    dataset_lm, collate_fn_lm = method_handler.build(build_ctx)
    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=config.training.batch_size_lm,
        sampler=None,
        collate_fn=collate_fn_lm,
        num_workers=0
    )

    for step, batch in enumerate(train_dataloader_lm, start=1):
        print(batch)
        break
    


if __name__ == "__main__":
    main()