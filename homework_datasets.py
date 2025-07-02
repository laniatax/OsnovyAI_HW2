import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


class CustomCSVDataset(Dataset):
    def __init__(self, csv_path, target_column):
        data = pd.read_csv(csv_path)
        assert target_column in data.columns, f"Целевая переменная '{target_column}' отсутствует в наборе данных."

        self.y = data[target_column]
        self.X = data.drop(columns=[target_column])

        # Обработка категориальных признаков
        for col in self.X.select_dtypes(include=["object", "category"]).columns:
            self.X[col] = LabelEncoder().fit_transform(self.X[col].astype(str))

        # Кодирование целевой переменной
        if self.y.dtype == "object" or self.y.dtype.name == "category":
            self.y = LabelEncoder().fit_transform(self.y.astype(str))

        # Нормализация числовых признаков
        self.X = StandardScaler().fit_transform(self.X)
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y.values if isinstance(self.y, pd.Series) else self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
