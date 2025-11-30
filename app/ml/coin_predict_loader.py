_loaded_model = None

import os

import torch

from app.ml.coin_predict_model import TransformerModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COIN_MODEL_PATH = os.getenv("COIN_MODEL_PATH", "plus_120/코인20/transformer_target5.pth")

_loaded_model = None


def get_model():
    global _loaded_model
    if _loaded_model is None:
        if not os.path.exists(COIN_MODEL_PATH):
            raise FileNotFoundError(f"모델 파일 없음: {COIN_MODEL_PATH}")

        model = TransformerModel()
        checkpoint = torch.load(COIN_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        _loaded_model = model
    return _loaded_model
