import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def replace_ingredient_ids(raw_value, ingredients_map):
    ingredients = []

    for item in str(raw_value).split(";"):
        if not item:
            continue

        # Убираем служебный префикс и получаем числовой id ингредиента.
        item_id = item.replace("ingr_", "").lstrip("0")
        item_id = int(item_id) if item_id else 0

        if item_id in ingredients_map:
            ingredients.append(ingredients_map[item_id])

    return ", ".join(ingredients) if ingredients else "unknown"


def load_data(data_dir):
    # Читаем таблицы с блюдами и словарем ингредиентов.
    dish = pd.read_csv(os.path.join(data_dir, "dish.csv"))
    ingredients = pd.read_csv(os.path.join(data_dir, "ingredients.csv"))

    ingredients_map = dict(zip(ingredients["id"].astype(int), ingredients["ingr"]))

    # Переводим технические id ингредиентов в обычный текст.
    dish["ingredients_text"] = dish["ingredients"].apply(
        lambda value: replace_ingredient_ids(value, ingredients_map)
    )
    # Путь до изображения каждого блюда формируем сразу.
    dish["image_path"] = dish["dish_id"].apply(
        lambda dish_id: os.path.join(data_dir, "images", str(dish_id), "rgb.png")
    )
    dish["total_calories"] = dish["total_calories"].astype("float32")
    # Массу уменьшаем по масштабу перед подачей в модель.
    dish["total_mass"] = dish["total_mass"].astype("float32") / 1000

    # Оставляем только строки, для которых есть изображение.
    dish = dish[dish["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return dish


def prepare_datasets(config):
    dataframe = load_data(config.DATA_DIR)

    # Тестовый сплит уже дан в датасете, train делим еще и на val.
    train_df = dataframe[dataframe["split"] == "train"].copy()
    test_df = dataframe[dataframe["split"] == "test"].copy()

    train_df, val_df = train_test_split(
        train_df,
        test_size=config.VAL_SIZE,
        random_state=config.SEED,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_transforms(config, ds_type="train"):
    if ds_type == "train":
        return A.Compose(
            [
                # Для обучения берем мягкие аугментации.
                A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE, p=1.0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )


class MultimodalDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        # Изображения лежат во вложенных папках по dish_id.
        image = Image.open(row["image_path"]).convert("RGB")
        image = np.array(image)
        image = self.transform(image=image)["image"]

        return {
            "dish_id": row["dish_id"],
            "ingredients_text": row["ingredients_text"],
            "image": image,
            "mass": np.float32(row["total_mass"]),
            "target": np.float32(row["total_calories"]),
            "image_path": row["image_path"],
        }


def collate_fn(batch, tokenizer, max_length):
    text_batch = [item["ingredients_text"] for item in batch]

    # Токенизируем текстовые признаки в батче.
    text_input = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    return {
        "dish_id": [item["dish_id"] for item in batch],
        "ingredients_text": text_batch,
        "input_ids": text_input["input_ids"],
        "attention_mask": text_input["attention_mask"],
        "image": torch.stack([item["image"] for item in batch]),
        "mass": torch.tensor(
            [item["mass"] for item in batch],
            dtype=torch.float32,
        ).unsqueeze(1),
        "target": torch.tensor(
            [item["target"] for item in batch],
            dtype=torch.float32,
        ),
        "image_path": [item["image_path"] for item in batch],
    }


load_dataset_frame = load_data
