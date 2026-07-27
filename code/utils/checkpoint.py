"""Checkpoint compatibility across MAGNET, FluoResFM, and UNiFMIR."""
from pathlib import Path
import warnings
import torch

MODEL_KEYS = ("model_state_dict", "state_dict", "model")
OPTIMIZER_KEYS = ("optimizer_state_dict", "optimizer")
SCHEDULER_KEYS = ("scheduler_state_dict", "lr_scheduler")


def _first_mapping(container, names):
    for name in names:
        value = container.get(name)
        if isinstance(value, dict):
            return value
    return None


def extract_model_state(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(checkpoint).__name__}")
    state = _first_mapping(checkpoint, MODEL_KEYS)
    if state is not None:
        return state
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError(f"No model weights found; expected one of {MODEL_KEYS}")


def _strip_prefix(state, prefix):
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix):]: value for key, value in state.items()}
    return state


def _best_state_for_model(state, model):
    candidates = [state]
    for prefix in ("module.", "model.", "module.model."):
        candidate = _strip_prefix(state, prefix)
        if candidate is not state:
            candidates.append(candidate)
    model_state = model.state_dict()
    return max(candidates, key=lambda candidate: sum(
        key in model_state and model_state[key].shape == value.shape
        for key, value in candidate.items() if torch.is_tensor(value)
    ))


def load_checkpoint(path, trainer, device, optimizer=None, scheduler=None,
                    mode="pretrained"):
    """Load weights and print a concise, shared status report."""
    path = Path(path) if path else None
    if path is None:
        print("[Checkpoint] No checkpoint requested; starting from scratch.")
        return 0, None
    path = path.expanduser().resolve()
    print(f"[Checkpoint] mode={mode} | loading={path}")
    if not path.is_file():
        print(f"[Checkpoint] FAILED | file not found: {path}")
        return 0, None
    if isinstance(device, int):
        device = torch.device("cuda", device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        target = trainer.model
        state = _best_state_for_model(extract_model_state(checkpoint), target)
        target_state = target.state_dict()
        mismatch = [key for key in state if key in target_state and state[key].shape != target_state[key].shape]
        missing = sorted(set(target_state) - set(state))
        unexpected = sorted(set(state) - set(target_state))
        allowed = []
        allowed_mismatch = []
        if trainer.family == "unifmir":
            allowed = [key for key in unexpected if key.startswith("upsamplesr.")]
            allowed_mismatch = [
                key for key in mismatch
                if key == "conv_firstdT.weight"
            ]
            if allowed_mismatch:
                for key in allowed_mismatch:
                    warnings.warn(
                        "Skipping UNiFMIR denoise input-head weight because "
                        f"the configured input channels changed: {key} "
                        f"{tuple(state[key].shape)} -> {tuple(target_state[key].shape)}",
                        stacklevel=2,
                    )
                    state[key] = target_state[key]
        forbidden = sorted(set(unexpected) - set(allowed))
        forbidden_mismatch = sorted(set(mismatch) - set(allowed_mismatch))
        if missing or forbidden_mismatch or forbidden:
            raise RuntimeError(
                f"Incompatible {trainer.family} checkpoint {path}: missing={len(missing)}, "
                f"unexpected={len(forbidden)}, shape_mismatch={len(forbidden_mismatch)}"
            )
        if allowed:
            warnings.warn(
                "Ignoring legacy UNiFMIR x2 SR-head weights because pseudo-SR uses "
                f"scale=1: {allowed}", stacklevel=2)
        target.load_state_dict(
            {key: value for key, value in state.items() if key in target_state},
            strict=True,
        )
    except Exception as error:
        print(f"[Checkpoint] FAILED | {type(error).__name__}: {error}")
        raise

    optimizer_restored = False
    if optimizer is not None:
        state_opt = _first_mapping(checkpoint, OPTIMIZER_KEYS)
        if state_opt:
            try:
                optimizer.load_state_dict(state_opt)
                optimizer_restored = True
            except (ValueError, RuntimeError) as error:
                warnings.warn(f"Optimizer state was not restored: {error}", stacklevel=2)
    scheduler_restored = False
    if scheduler is not None:
        state_sched = _first_mapping(checkpoint, SCHEDULER_KEYS)
        if state_sched:
            try:
                scheduler.load_state_dict(state_sched)
                scheduler_restored = True
            except (ValueError, RuntimeError) as error:
                warnings.warn(f"Scheduler state was not restored: {error}", stacklevel=2)
    epoch = int(checkpoint.get("epoch") or 0)
    print(
        f"[Checkpoint] SUCCESS | family={trainer.family} | epoch={epoch} | "
        f"weights={len(target_state)} | optimizer={optimizer_restored} | "
        f"scheduler={scheduler_restored}"
    )
    return epoch, checkpoint


def make_checkpoint(trainer, optimizer, scheduler, epoch, train_l1, eval_metrics, task):
    return {
        "epoch": int(epoch),
        "model_state_dict": trainer.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_l1": float(train_l1),
        "eval_metrics": eval_metrics,
        "task": task,
    }
