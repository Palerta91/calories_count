import os
import random
import re
from functools import partial

import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

try:
    from .config import TrainConfig
    from .dataset import MultimodalDataset, collate_fn, get_transforms, prepare_datasets
except ImportError:
    from config import TrainConfig
    from dataset import MultimodalDataset, collate_fn, get_transforms, prepare_datasets


class BaseMultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Отдельно кодируем текст и изображение.
        self.text_model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.image_model = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0,
        )

        self.text_proj = nn.Linear(
            self.text_model.config.hidden_size,
            config.HIDDEN_DIM,
        )
        self.image_proj = nn.Linear(
            self.image_model.num_features,
            config.HIDDEN_DIM,
        )

    def forward(self, input_ids, attention_mask, image):
        text_features = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        image_features = self.image_model(image)

        text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)
        return text_emb, image_emb


class MultimodalRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base_model = BaseMultimodalModel(config)

        # На вход regressor подаем fusion-признак и массу порции.
        self.regressor = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM + 1, config.HIDDEN_DIM // 2),
            nn.LayerNorm(config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, 1),
        )

    def forward(self, input_ids, attention_mask, image, mass):
        text_emb, image_emb = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image=image,
        )

        # Объединяем модальности через поэлементное произведение.
        fused_emb = text_emb * image_emb
        fused_emb = torch.cat([fused_emb, mass], dim=1)
        return self.regressor(fused_emb).squeeze(1)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_requires_grad(module, unfreeze_pattern="", verbose=False):
    if not unfreeze_pattern:
        for _, param in module.named_parameters():
            param.requires_grad = False
        return

    # Размораживаем только слои, которые подходят под шаблон.
    pattern = re.compile(unfreeze_pattern)

    for name, param in module.named_parameters():
        if pattern.search(name):
            param.requires_grad = True
            if verbose:
                print(f"Разморожен слой: {name}")
        else:
            param.requires_grad = False


def validate(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in val_loader:
            # Переносим батч на выбранное устройство.
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }
            targets = batch["target"].to(device)

            predictions = model(**inputs)
            loss = criterion(predictions, targets)

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_items += batch_size

    return total_loss / total_items


def train(config=None):
    if config is None:
        config = TrainConfig()

    # Фиксируем seed для воспроизводимости.
    seed_everything(config.SEED)
    os.makedirs(os.path.dirname(config.SAVE_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MultimodalRegressor(config).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    set_requires_grad(
        model.base_model.text_model,
        config.TEXT_MODEL_UNFREEZE,
    )
    set_requires_grad(
        model.base_model.image_model,
        config.IMAGE_MODEL_UNFREEZE,
    )

    # В оптимизатор включаем все обучаемые части модели.
    optimizer = AdamW(
        [
            {"params": model.base_model.text_model.parameters(), "lr": config.TEXT_LR},
            {"params": model.base_model.image_model.parameters(), "lr": config.IMAGE_LR},
            {"params": model.base_model.text_proj.parameters(), "lr": config.CLASSIFIER_LR},
            {"params": model.base_model.image_proj.parameters(), "lr": config.CLASSIFIER_LR},
            {"params": model.regressor.parameters(), "lr": config.CLASSIFIER_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = nn.L1Loss()

    train_df, val_df, _ = prepare_datasets(config)

    train_loader = DataLoader(
        MultimodalDataset(train_df, get_transforms(config)),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=config.TEXT_MAX_LENGTH,
        ),
    )
    val_loader = DataLoader(
        MultimodalDataset(val_df, get_transforms(config, ds_type="val")),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=config.TEXT_MAX_LENGTH,
        ),
    )

    best_mae = float("inf")
    history = []

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Собираем входы модели из текстовой и визуальной части.
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }
            targets = batch["target"].to(device)

            optimizer.zero_grad()
            predictions = model(**inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_mae = validate(model, val_loader, device, criterion)
        train_mae = total_loss / len(train_loader)

        history.append(
            {
                "epoch": epoch + 1,
                "train_mae": train_mae,
                "val_mae": val_mae,
            }
        )

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"train_mae={train_mae:.4f} | "
            f"val_mae={val_mae:.4f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            # Сохраняем лучший чекпоинт по валидационной метрике.
            torch.save(model.state_dict(), config.SAVE_PATH)

    return pd.DataFrame(history)


def predict_on_test(config=None):
    if config is None:
        config = TrainConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, test_df = prepare_datasets(config)
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)

    test_loader = DataLoader(
        MultimodalDataset(test_df, get_transforms(config, ds_type="test")),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            max_length=config.TEXT_MAX_LENGTH,
        ),
    )

    model = MultimodalRegressor(config).to(device)
    model.load_state_dict(torch.load(config.SAVE_PATH, map_location=device))
    criterion = nn.L1Loss()
    predictions = []

    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in test_loader:
            # На тесте дополнительно сохраняем предсказания для анализа ошибок.
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "image": batch["image"].to(device),
                "mass": batch["mass"].to(device),
            }
            targets = batch["target"].to(device)

            outputs = model(**inputs)
            loss = criterion(outputs, targets)

            batch_size = len(targets)
            total_loss += loss.item() * batch_size
            total_items += batch_size

            errors = torch.abs(outputs - targets)

            for i in range(batch_size):
                predictions.append(
                    {
                        "dish_id": batch["dish_id"][i],
                        "ingredients_text": batch["ingredients_text"][i],
                        "target": targets[i].item(),
                        "prediction": outputs[i].item(),
                        "absolute_error": errors[i].item(),
                        "image_path": batch["image_path"][i],
                    }
                )

    metrics = {"mae": total_loss / total_items}
    predictions = pd.DataFrame(predictions).sort_values(
        "absolute_error",
        ascending=False,
    )

    return metrics, predictions.reset_index(drop=True)


def show_hard_examples(predictions, top_n=5):
    import matplotlib.pyplot as plt

    hard_examples = predictions.head(top_n)
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 4 * top_n))

    if top_n == 1:
        axes = [axes]

    for axis, (_, row) in zip(axes, hard_examples.iterrows()):
        image = plt.imread(row["image_path"])

        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{row['dish_id']} | true={row['target']:.1f} | "
            f"pred={row['prediction']:.1f} | "
            f"abs_error={row['absolute_error']:.1f}",
            fontsize=11,
        )
        axis.text(
            0.0,
            -0.12,
            row["ingredients_text"],
            transform=axis.transAxes,
            fontsize=9,
            wrap=True,
            va="top",
        )

    plt.tight_layout()
    plt.show()
