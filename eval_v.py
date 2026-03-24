import os
import sys
import subprocess
import re
from termcolor import cprint

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf, cli_conf.config

if __name__ == "__main__":
    config, config_path = get_config()

    project_name = config.experiment.project
    eval_type = config.dataset.data_type
    wandb_run = None
    try:
        import wandb

        wandb_cfg = OmegaConf.to_container(config, resolve=True)
        wandb_run = wandb.init(
            project=project_name,
            name=project_name,
            config=wandb_cfg,
        )

        if config_path and os.path.exists(config_path):
            cfg_art = wandb.Artifact(name=f"{project_name}-config", type="eval_config")
            cfg_art.add_file(config_path)
            wandb_run.log_artifact(cfg_art)
    except Exception as e:
        cprint(f"[wandb] init failed, continue without wandb. error={e}", color="yellow")

    def begin_with(file_name):
        with open(file_name, "w") as f:
            f.write("")
        
    def sample(model_base, strategy):
        cprint(f"This is sampling.", color = "green")
        if model_base == "dream":
            subprocess.run(
                f'python dream_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "llada" or model_base == "mmada":
            subprocess.run(
                f'python mmada_v_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "sdar":
            subprocess.run(
                f'python sdar_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "trado":
            subprocess.run(
                f'python trado_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
        elif model_base == "lladav":
            subprocess.run(
                f'python lladav_sample.py '
                f'config=../configs/{project_name}.yaml '
                f'rollout.remasking_strategy={strategy}',
                shell=True,
                cwd='sample',
                check=True,
            )
    
    def reward(strategy):
        cprint(f"This is the rewarding.", color = "green")
        subprocess.run(
            f'python reward_v.py '
            f'config=../configs/{project_name}.yaml '
            f'rollout.remasking_strategy={strategy}',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    def execute(strategy):
        cprint(f"This is the execution.", color = "green")
        subprocess.run(
            f'python execute.py '
            f'config=../configs/{project_name}.yaml '
            f'rollout.remasking_strategy={strategy}',
            shell=True,
            cwd='reward',
            check=True,
        )
    
    
    
    os.makedirs(f"{project_name}/results", exist_ok=True)
    
    remasking_strategies = config.rollout.remasking_strategy
    if not isinstance(remasking_strategies, (list, ListConfig)):
        remasking_strategies = [remasking_strategies]
    
    for strategy in remasking_strategies:
        cprint(f"--- Running evaluation for strategy: {strategy} ---", "green")
        sample(config.model_base, strategy)
        if eval_type == "code":
            execute(strategy)
        reward(strategy)

        # ---- Log each strategy's results to wandb ----
        if wandb_run is not None:
            try:
                pretrained_model = config.model
                dataset = config.dataset.eval_dataset
                outputs_name = (
                    "eval-" + pretrained_model.replace("/", ".") + "-" + dataset + "-" + str(strategy)
                )
                results_path = os.path.join(project_name, "results", f"results-{outputs_name}.txt")
                if os.path.exists(results_path):
                    with open(results_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    def _last_float(pattern: str):
                        matches = re.findall(pattern, content)
                        if not matches:
                            return None
                        return float(matches[-1])

                    acc = _last_float(r"acc:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
                    avg_len = _last_float(r"avg length:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

                    log_dict = {}
                    if acc is not None:
                        log_dict[f"eval/acc/{strategy}"] = acc
                    if avg_len is not None:
                        log_dict[f"eval/avg_length/{strategy}"] = avg_len

                    if log_dict:
                        wandb_run.log(log_dict)

                    # Also attach the results text file for traceability.
                    res_art = wandb.Artifact(
                        name=f"{project_name}-results-{strategy}",
                        type="eval_results",
                    )
                    res_art.add_file(results_path)
                    wandb_run.log_artifact(res_art)
                else:
                    cprint(f"[wandb] results file not found: {results_path}", color="yellow")
            except Exception as e:
                cprint(f"[wandb] log strategy failed. error={e}", color="yellow")

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:
            pass


