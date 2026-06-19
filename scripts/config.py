from dataclasses import dataclass
from pathlib import Path


# Опорные пути считаем относительно корня репозитория.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Все основные параметры обучения держим в одном конфиге.
@dataclass
class TrainConfig:
    DATA_DIR: str = str(PROJECT_DIR / "data")
    SAVE_PATH: str = str(PROJECT_DIR / "artifacts" / "best_model.pt")
    TEXT_MODEL_NAME: str = "distilbert-base-uncased"
    IMAGE_MODEL_NAME: str = "resnet18"
    IMAGE_SIZE: int = 224
    HIDDEN_DIM: int = 256
    DROPOUT: float = 0.15
    BATCH_SIZE: int = 4
    EPOCHS: int = 5
    TEXT_MAX_LENGTH: int = 128
    TEXT_LR: float = 2e-5
    IMAGE_LR: float = 1e-4
    CLASSIFIER_LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    VAL_SIZE: float = 0.15
    NUM_WORKERS: int = 2
    SEED: int = 42
    TEXT_MODEL_UNFREEZE: str = "transformer.layer.5"
    IMAGE_MODEL_UNFREEZE: str = "layer4"
