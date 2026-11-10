from pathlib import Path
from typing import Callable
import yaml
import optuna
from optuna.samplers import TPESampler


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as f:
        return yaml.safe_load(f) or {}


def save_config(config: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def get_or_tune_params(
    section: str,
    config_path: str | Path,
    search_space: Callable[[optuna.Trial], dict],
    objective: Callable[[dict], float],
    n_trials: int = 40,
    direction: str = "minimize",
    force: bool = False,
    seed: int = 42,
) -> dict:
 
    config = load_config(config_path)

    cached = config.get(section, {}).get("best_params")
    if cached and not force:
        return dict(cached)


    def _obj(trial: optuna.Trial) -> float:
        params = search_space(trial)
        return objective(params)

    study = optuna.create_study(direction=direction, sampler=TPESampler(seed=seed))
    study.optimize(_obj, n_trials=n_trials)

    config[section] = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": n_trials,
    }
    save_config(config, config_path)
    print(f"[tuning] '{section}': best={study.best_value:.4f}, сохранено в {config_path}")

    return dict(study.best_params)