import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class LinearRegressionWithRegularization(nn.Module):
    def __init__(self, input_dim, l1_lambda=0.01, l2_lambda=0.01):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def regularization_loss(self):
        l1 = self.l1_lambda * torch.norm(self.linear.weight, 1)
        l2 = self.l2_lambda * torch.norm(self.linear.weight, 2)
        return l1 + l2


class LogisticRegressionMulticlass(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)  # логиты

    def predict(self, x):
        logits = self.forward(x)
        return torch.argmax(F.softmax(logits, dim=1), dim=1)


def calculate_metrics(y_true, y_pred, average='macro'):
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    precision = precision_score(y_true_np, y_pred_np, average=average, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, average=average, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, average=average, zero_division=0)

    # ROC-AUC только для бинарной классификации
    if len(np.unique(y_true_np)) == 2:
        try:
            roc = roc_auc_score(y_true_np, y_pred_np)
        except:
            roc = 0.0
    else:
        roc = None

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc
    }


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path="plots/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
