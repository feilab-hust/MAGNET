"""Model-family-independent checkpoint loading and inference helper."""
from pathlib import Path
import torch
import yaml
from models import MultiModel_Trainer
from utils.checkpoint import load_checkpoint


def load_model(config_path, checkpoint_path=None, device="cuda"):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["model"]
    model = MultiModel_Trainer.from_config(cfg)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, device, mode="inference")
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, batch):
    return model.forward_batch(batch)["prediction"]
