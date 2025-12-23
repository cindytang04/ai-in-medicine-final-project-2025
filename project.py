# ==============================
# @Authors: Cindy Tang, Brandon Dang
# Student ID: 258355049, 258355055
#
# description:
# This project implements multiple machine learning and deep learning models to 
# predict molecular BBBP using SMILES representations and molecular graphs.
# Models include traditional ML baselines, CNNs on SMILES encodings, and Graph Neural Networks (GCNs) using DeepChem and DGL.
#
# ==============================

# ==============================
# description:
#   Configure environment variables to control backend selection,
#   suppress excessive logging, and disable third-party experiment
#   tracking (e.g., Weights & Biases).
# ==============================
import os 
os.environ["DGLBACKEND"] = "pytorch"
os.environ["DGL_SKIP_GRAPHBOLT"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOGLEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLED"] = "true"

# ==============================
# description:
#   Suppress Python warnings and TensorFlow logging output to
#   improve readability of training logs.
# ==============================
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ==============================
# description:
#   Import core scientific, machine learning, and deep learning
#   libraries required for molecular modeling and evaluation.
# ==============================
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
torch.device("cpu")

# ==============================
# description:
#   Disable RDKit molecule parsing warnings which can arise from
#   invalid or uncommon SMILES strings.
# ==============================
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import deepchem as dc
from deepchem.feat import MolGraphConvFeaturizer, CircularFingerprint

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore


# Reproducibility
# ==============================
# description:
#   Set fixed random seeds for Python, NumPy, and TensorFlow to
#   ensure reproducible experiments.
#
# @param:
#   seed (int):
#       Random seed value (default = 42)
#
# @return:
#   None
# ==============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Metrics + plotting helpers
# ==============================
# description:
#   Compute standard binary classification metrics using predicted
#   probabilities.
#
# @param:
#   y_true (array-like):
#       Ground truth binary labels
#
#   y_prob (array-like):
#       Predicted probabilities for the positive class
#
#   threshold (float):
#       Classification threshold for converting probabilities
#       into binary predictions (default = 0.5)
#
# @return:
#   dict:
#       Dictionary containing ROC-AUC, accuracy, and F1 score
# ==============================
def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_prob = np.asarray(y_prob).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

# ==============================
# description:
#   Plot the Receiver Operating Characteristic (ROC) curve for
#   a binary classifier.
#
# @param:
#   y_true (array-like):
#       Ground truth binary labels
#
#   y_prob (array-like):
#       Predicted probabilities for the positive class
#
#   label (str):
#       Label for the ROC curve legend
#
# @return:
#   None
# ==============================
def plot_roc(y_true, y_prob, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=label)


# 1) Load BBBP (MolNet)
# ==============================
# description:
#   Load the BBBP (Blood–Brain Barrier Penetration) dataset from
#   DeepChem MolNet using a scaffold split and raw SMILES strings.
# ==============================
print("Loading BBBP from DeepChem MolNet...")
tasks, (train_raw, valid_raw, test_raw), transformers = dc.molnet.load_bbbp(
    featurizer="Raw",          # IMPORTANT: American spelling
    split="scaffold"
)

train_smiles = np.array(train_raw.ids)
valid_smiles = np.array(valid_raw.ids)
test_smiles  = np.array(test_raw.ids)

y_train = train_raw.y.reshape(-1).astype(int)
y_valid = valid_raw.y.reshape(-1).astype(int)
y_test  = test_raw.y.reshape(-1).astype(int)

print("BBBP sizes:", len(train_smiles), len(valid_smiles), len(test_smiles))
print("Positive rate (train):", round(float(y_train.mean()), 3))


# 2) ECFP + RandomForest

print("\n=== 2) ECFP + RandomForest baseline ===")
ecfp = CircularFingerprint(size=2048, radius=2)

# ==============================
# description:
#   Convert SMILES strings into fixed-length Extended-Connectivity
#   Fingerprints (ECFP) for use in classical machine learning models.
#
# @param:
#   smiles_list (array-like):
#       List of SMILES strings
#
# @return:
#   np.ndarray:
#       2D array of fingerprint features (n_samples × n_bits)
# ==============================
def featurize_ecfp(smiles_list):
    X = ecfp.featurize(smiles_list)  # (n, 2048)
    return np.asarray(X)

# ==============================
# description:
#   Train a Random Forest classifier using ECFP features and
#   evaluate performance on the test set.
# ==============================
X_train_fp = featurize_ecfp(train_smiles)
X_valid_fp = featurize_ecfp(valid_smiles)
X_test_fp  = featurize_ecfp(test_smiles)

rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
rf.fit(X_train_fp, y_train)

rf_prob = rf.predict_proba(X_test_fp)[:, 1]
rf_metrics = binary_metrics(y_test, rf_prob)
print("RF metrics:", rf_metrics)


# 3) SMILES CNN (TensorFlow)

print("\n=== 3) SMILES CNN (TensorFlow) ===")

# Build char vocab from TRAIN only (safe; avoids leakage)
all_chars = sorted(list(set("".join(train_smiles))))

# Reserve: 0 = PAD, 1 = UNK, 2.. = actual chars
char_to_idx = {c: i + 2 for i, c in enumerate(all_chars)}
PAD_IDX = 0
UNK_IDX = 1
vocab_size = max(char_to_idx.values()) + 1  # inclusive max + 1

# ==============================
# description:
#   Convert SMILES strings into fixed-length integer sequences
#   suitable for neural network input.
#
# @param:
#   smiles_list (array-like):
#       List of SMILES strings
#
#   max_len (int):
#       Maximum sequence length (default = 120)
#
# @return:
#   np.ndarray:
#       Integer-encoded SMILES sequences
# ==============================
def smiles_to_int(smiles_list, max_len=120):
    arr = np.zeros((len(smiles_list), max_len), dtype=np.int32)
    for i, s in enumerate(smiles_list):
        for j, ch in enumerate(s[:max_len]):
            arr[i, j] = char_to_idx.get(ch, UNK_IDX)
    return arr

MAX_LEN = 120
X_train_smi = smiles_to_int(train_smiles, MAX_LEN)
X_valid_smi = smiles_to_int(valid_smiles, MAX_LEN)
X_test_smi  = smiles_to_int(test_smiles,  MAX_LEN)

# ==============================
# description:
#   Build a 1D convolutional neural network for SMILES-based
#   molecular classification.
#
# @param:
#   vocab_size (int):
#       Number of unique characters in the SMILES vocabulary
#
#   max_len (int):
#       Maximum SMILES sequence length
#
# @return:
#   tf.keras.Model:
#       Compiled Keras CNN model
# ==============================
def build_smiles_cnn(vocab_size, max_len):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=64, input_length=max_len, mask_zero=False),
        layers.Conv1D(128, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam", 
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model

# ==============================
# description:
#   Train the SMILES CNN model and record loss and AUC
#   metrics across epochs.
# ==============================
smiles_model = build_smiles_cnn(vocab_size, MAX_LEN)
hist = smiles_model.fit(
    X_train_smi, y_train,
    validation_data=(X_valid_smi, y_valid),
    epochs=30,
    batch_size=32,
    verbose=1
)

smiles_prob = smiles_model.predict(X_test_smi, verbose=0).reshape(-1)
smiles_metrics = binary_metrics(y_test, smiles_prob)
print("SMILES CNN metrics:", smiles_metrics)

#training curves
plt.figure()
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.title("SMILES CNN Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(hist.history["auc"], label="train_auc")
plt.plot(hist.history["val_auc"], label="val_auc")
plt.title("SMILES CNN AUC")
plt.xlabel("epoch")
plt.ylabel("AUC")
plt.legend()
plt.tight_layout()
plt.show()

# 4) GCN (DeepChem + DGL)
print("\n=== 4) GCN GNN (DeepChem + DGL) ===")

gcn_featurizer = MolGraphConvFeaturizer(use_edges=True)

# ==============================
# description:
#   Convert SMILES strings into molecular graph objects compatible
#   with DeepChem graph neural networks.
#
# @param:
#   smiles_list (array-like):
#       List of SMILES strings
#
#   y (array-like):
#       Corresponding binary labels
#
# @return:
#   dc.data.NumpyDataset:
#       Dataset containing molecular graphs and labels
# ==============================
def featurize_graph(smiles_list, y):
    graphs = []
    y_out = []
    ids_out = []
    bad = 0

    feats = gcn_featurizer.featurize(smiles_list)
    for s, g, label in zip(smiles_list, feats, y):
        if g is None:
            bad += 1
            continue
        graphs.append(g)
        y_out.append([int(label)])  # shape (n, n_tasks=1)
        ids_out.append(s)

    if bad:
        print(f"[WARN] Dropped {bad} invalid SMILES.")

    X = np.array(graphs, dtype=object)
    Y = np.array(y_out, dtype=np.int32)
    return dc.data.NumpyDataset(X=X, y=Y, ids=np.array(ids_out))

# ==============================
# description:
#   Train a Graph Convolutional Network (GCN) on molecular graphs
#   and evaluate validation ROC-AUC across epochs.
# ==============================
gcn_train = featurize_graph(train_smiles, y_train)
gcn_valid = featurize_graph(valid_smiles, y_valid)
gcn_test  = featurize_graph(test_smiles,  y_test)

gcn_model = dc.models.GCNModel(
    n_tasks=1,
    mode="classification",
    dropout=0.2,
    batch_size=32,
    learning_rate=1e-3,
    device="cpu"
)


train_losses = []
valid_rocs = []

EPOCHS = 30
for epoch in range(1, EPOCHS + 1):
    loss = gcn_model.fit(gcn_train, nb_epoch=1)
    train_losses.append(float(loss))

    valid_scores = gcn_model.evaluate(
        gcn_valid,
        metrics=[dc.metrics.Metric(dc.metrics.roc_auc_score)],
        transformers=[]
    )
    # DeepChem returns dict keys like "roc_auc_score"
    valid_roc = float(valid_scores.get("roc_auc_score", np.nan))
    valid_rocs.append(valid_roc)

    print(f"Epoch {epoch:02d} | train_loss={loss:.4f} | valid_roc_auc={valid_roc:.4f}")

gcn_pred = gcn_model.predict(gcn_test)

# Extract P(class=1)
if gcn_pred.ndim == 3:
    # (N, n_tasks, n_classes)
    gcn_prob = gcn_pred[:, 0, 1]
elif gcn_pred.ndim == 2:
    # could be (N, 2) or (N, 1)
    gcn_prob = gcn_pred[:, 1] if gcn_pred.shape[1] == 2 else gcn_pred[:, 0]
else:
    gcn_prob = gcn_pred

gcn_prob = np.asarray(gcn_prob).reshape(-1)

gcn_metrics = binary_metrics(gcn_test.y.reshape(-1), gcn_prob)
print("GCN metrics:", gcn_metrics)

# GCN curves
plt.figure()
plt.plot(train_losses)
plt.title("GCN Train Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(valid_rocs)
plt.title("GCN Valid ROC-AUC")
plt.xlabel("epoch")
plt.ylabel("ROC-AUC")
plt.tight_layout()
plt.show()

# Compare
# ==============================
# description:
#   Compare Random Forest, SMILES CNN, and GCN models using
#   test-set ROC-AUC, accuracy, and F1 score.
# ==============================
print("\n=== 5) Comparison ===")
results = pd.DataFrame([
    {"model": "ECFP + RF",   "roc_auc": rf_metrics["roc_auc"],     "accuracy": rf_metrics["accuracy"],     "f1": rf_metrics["f1"]},
    {"model": "SMILES CNN",  "roc_auc": smiles_metrics["roc_auc"], "accuracy": smiles_metrics["accuracy"], "f1": smiles_metrics["f1"]},
    {"model": "GCN (GNN)",   "roc_auc": gcn_metrics["roc_auc"],    "accuracy": gcn_metrics["accuracy"],    "f1": gcn_metrics["f1"]},
]).sort_values("roc_auc", ascending=False)

print(results.to_string(index=False))

plt.figure()
plt.bar(results["model"], results["roc_auc"])
plt.title("Test ROC-AUC Comparison (BBBP)")
plt.ylabel("ROC-AUC")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# ==============================
# description:
#   Visualize performance comparisons using bar plots and
#   ROC curve overlays.
# ==============================
# ROC overlay
plt.figure()
plot_roc(y_test, rf_prob, "ECFP+RF")
plot_roc(y_test, smiles_prob, "SMILES CNN")
plot_roc(gcn_test.y.reshape(-1), gcn_prob, "GCN (GNN)")
plt.plot([0, 1], [0, 1], "--")
plt.title("ROC Curves (Test)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()
