import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import logging
from homework_datasets import CustomCSVDataset
from homework_model_modification import LinearRegressionWithRegularization

logging.basicConfig(level=logging.INFO)

def run_experiment(dataset_path, target_column, learning_rates, batch_sizes, optimizers):
    results = []

    for lr in learning_rates:
        for bs in batch_sizes:
            for opt_name in optimizers:
                # Загрузка датасета с правильной целевой переменной
                dataset = CustomCSVDataset(dataset_path, target_column)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

                model = LinearRegressionWithRegularization(dataset[0][0].shape[0])
                criterion = nn.MSELoss()
                optimizer = {
                    'SGD': optim.SGD(model.parameters(), lr=lr),
                    'Adam': optim.Adam(model.parameters(), lr=lr),
                    'RMSprop': optim.RMSprop(model.parameters(), lr=lr)
                }[opt_name]

                # Обучение
                model.train()
                total_loss = 0
                for epoch in range(20):
                    epoch_loss = 0
                    for x_batch, y_batch in dataloader:
                        optimizer.zero_grad()
                        y_pred = model(x_batch).squeeze()
                        loss = criterion(y_pred, y_batch) + model.regularization_loss()
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                    total_loss = epoch_loss

                results.append((lr, bs, opt_name, total_loss))
                logging.info(f"LR: {lr}, Batch size: {bs}, Optimizer: {opt_name}, Final loss: {total_loss:.4f}")

    # Сохраняем результаты в виде таблицы
    os.makedirs("plots", exist_ok=True)
    with open("plots/hyperparameter_results.txt", "w") as f:
        f.write("LR\tBatchSize\tOptimizer\tFinalLoss\n")
        for r in results:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\n")

if __name__ == "__main__":
    run_experiment(
        dataset_path="data/BostonHousing.csv",
        target_column="medv",
        learning_rates=[0.001, 0.01, 0.1],
        batch_sizes=[8, 16],
        optimizers=["SGD", "Adam", "RMSprop"]
    )
