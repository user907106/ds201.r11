import numpy as np
np.set_printoptions(linewidth=np.inf)
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from train import train, evaluate, compute_metrics
from data_utils.mnist import MnistDataset, collate_fn as mnist_collate
from data_utils.vinafood21 import VinaFood21, collate_fn as vina_collate

from models.lenet import LeNet
from models.googlenet import GoogleNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(dataset_name, batch_size=32):
    if dataset_name == "MNIST":
        train_dataset = MnistDataset(
            img_path="data/mnist/train-images.idx3-ubyte",
            label_path="data/mnist/train-labels.idx1-ubyte"
        )
        test_dataset = MnistDataset(
            img_path="data/mnist/t10k-images.idx3-ubyte",
            label_path="data/mnist/t10k-labels.idx1-ubyte"
        )
        num_classes = 10
    else:
        train_dataset = VinaFood21("data/VinaFood21/train", (224, 224))
        test_dataset = VinaFood21("data/VinaFood21/test", (224, 224))
        num_classes = 21

    collate_fn = mnist_collate
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, test_loader, num_classes

def get_model(model_name, num_classes):
    match model_name:
        case "LeNet":
            model = LeNet(num_classes).to(device)
        case "GoogleNet":
            model = GoogleNet(num_classes).to(device)
        case _:
            raise ValueError(f"Unknown model name: {model_name}")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("model_name", type=str)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)

    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr

    train_loader, test_loader, num_classes = get_dataloader(dataset_name, batch_size)
    model = get_model(model_name, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    best_acc = 0.0
    best_model = None
    best_trues, best_preds = None, None

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        loss = train(model, train_loader, criterion, optimizer, device)

        print(f"Train Loss: {loss:.4f} | ", end="")

        trues, preds = evaluate(model, test_loader, device)
        scores = compute_metrics(trues, preds)
        for score_name, score_value in scores.items():
            print(f"{score_name}: {score_value:.4f} | ", end="")
        print()

        current_acc = scores["accuracy"]
        if current_acc > best_acc:
            best_acc = current_acc
            best_model = model
            best_trues, best_preds = trues, preds

    print(f"\nBest Accuracy for Model {model_name}:", best_acc)
    print(confusion_matrix(best_trues, best_preds))
