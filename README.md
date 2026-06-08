# Wildfire Risk Classification

A multilayer perceptron (MLP) model trained on global wildfire and meteorological data to classify wildfire risk into low, medium, and high categories. Using 118k+ samples from NASA FIRMS and Open-Meteo, the model achieves **94% test accuracy** through feature engineering, weighted loss, regularization, and early stopping.

**Live API:** https://wildfire-risk-classifier.onrender.com/docs

A full detailed technical report is available at:  
`docs/Report/main.pdf`

## Overview

- **Model Type:** Feed-forward neural network (MLP)
- **Input:** 17 engineered environmental and meteorological features
- **Output:** 3 wildfire risk classes (low / medium / high)
- **Dataset Size:** 118,858 observations (all numerical, no missing values)
- **Performance:**
  - Test Accuracy: **94.32%**
  - Macro F1-score: **0.939**
- **Key Techniques:**
  - Weighted Cross Entropy (class imbalance)
  - Batch Normalization
  - Dropout regularization
  - Early stopping
  - L2 weight decay
  - Feature engineering (temperature–humidity and wind–FWI interactions)

## Deployment

The model is served as a REST API built with FastAPI and containerized with Docker. It is deployed on Render and publicly accessible.

- `POST /predict` — accepts 15 environmental features, returns risk level + probabilities
- `GET /health` — health check
- Interactive docs: https://wildfire-risk-classifier.onrender.com/docs

## Project Structure

```plaintext
wildfire-risk-classifier/
├── artifacts/
│   ├── model.pt          # trained model weights
│   └── scaler.pkl        # fitted StandardScaler
├── docs/
│   └── Report/main.pdf
├── src/
│   ├── 01_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── model.py              # RiskClassifier architecture
├── main.py               # FastAPI app
├── train_and_export.py   # training script
├── Dockerfile
└── requirements.txt
```
