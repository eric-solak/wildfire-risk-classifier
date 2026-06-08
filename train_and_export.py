"""
Run once from the repo root to train the model and save inference artifacts:
    python train_and_export.py

Outputs:
    artifacts/model.pt    — model weights + input_size
    artifacts/scaler.pkl  — fitted StandardScaler
"""
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import RiskClassifier

os.makedirs("artifacts", exist_ok=True)


def prepare_data(data_src: str = "src/final_dataset.csv"):
    df = pd.read_csv(data_src)
    df.dropna(inplace=True)
    df["fire_weather_index"] = df["fire_weather_index"].clip(lower=0, upper=100)
    df["risk_level"] = np.where(
        df["fire_weather_index"] > 30, 2,
        np.where(df["fire_weather_index"] >= 10, 1, 0),
    )
    df["temp_humidity_interaction"] = df["temp_mean"] * (1 - df["humidity_min"] / 100)
    df["wind_fwi_interaction"] = df["wind_speed_max"] * df["fire_weather_index"]

    X = df.drop(columns=["risk_level", "occured", "frp"])
    y = df["risk_level"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    def to_tensor(arr, dtype):
        return torch.tensor(arr, dtype=dtype)

    return (
        to_tensor(X_train_s, torch.float32),
        to_tensor(y_train.values, torch.long),
        to_tensor(X_val_s, torch.float32),
        to_tensor(y_val.values, torch.long),
        to_tensor(X_test_s, torch.float32),
        to_tensor(y_test.values, torch.long),
        scaler,
        list(X.columns),
    )


def train_model(X_train, y_train, X_val, y_val, input_size, epochs=100, patience=5):
    model = RiskClassifier(input_size)

    y_np = y_train.cpu().numpy()
    classes = np.unique(y_np)
    counts = np.bincount(y_np)
    weights = torch.tensor(len(y_np) / (len(classes) * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | train_loss={loss.item():.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean().item()
    print(f"Test accuracy: {acc:.4f}  ({(preds == y_test).sum().item()}/{len(y_test)})")


if __name__ == "__main__":
    print("Preparing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols = prepare_data()
    input_size = X_train.shape[1]
    print(f"Input size: {input_size}")
    print(f"Features: {feature_cols}")

    print("\nTraining...")
    model = train_model(X_train, y_train, X_val, y_val, input_size)

    print("\nEvaluating...")
    evaluate(model, X_test, y_test)

    print("\nSaving artifacts...")
    torch.save({"state_dict": model.state_dict(), "input_size": input_size}, "artifacts/model.pt")
    with open("artifacts/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Saved: artifacts/model.pt, artifacts/scaler.pkl")
