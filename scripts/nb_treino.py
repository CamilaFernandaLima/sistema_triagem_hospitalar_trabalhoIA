"""
Naive Bayes — GaussianNB
=======================================================================
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, roc_auc_score, roc_curve,
    f1_score
)
import os
import warnings
warnings.filterwarnings("ignore")

# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
TRAIN_PATH = "data/data_split/train_normalized.csv"
TEST_PATH = "data/data_split/test_normalized.csv"
TARGET_COLUMN = "KTAS_target_binario"
OUTPUT_DIR = "resultados/nb"
RANDOM_STATE = 42

FEATURES = [
    "Sex", "Age", "Injury", "Mental", "Pain", "NRS_pain",
    "SBP", "DBP", "HR", "RR", "BT", "Saturation"
]


def carregar_dados():
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n>>'{path}' não encontrado.\n"
                "Execute split.py antes deste script."
            )
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    print(f"Treino: {train.shape}   Teste: {test.shape}\n")
    return train, test


def separar_xy(df):
    features = [c for c in FEATURES if c in df.columns]
    return df[features], df[TARGET_COLUMN]


def treinar_e_avaliar(X_train, y_train, X_test, y_test):
    print("=" * 57)
    print("  TREINAMENTO NAIVE BAYES  (GaussianNB)")
    print("=" * 57 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    modelo = GaussianNB()
    modelo.fit(X_train, y_train)

    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.20).astype(int)

    # ── Métricas principais ────────────────────────────────────
    acc_treino = accuracy_score(y_train, modelo.predict(X_train))
    acc_teste = accuracy_score(y_test, y_pred)
    f1_emerg = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)
    gap = acc_treino - acc_teste

    with open(f"{OUTPUT_DIR}/nb_metricas.txt", "w", encoding="utf-8") as f:
        f.write("=== MÉTRICAS — NAIVE BAYES ===\n\n")
        f.write(f"Modelo: GaussianNB\n\n")
        f.write(f"Acurácia no Treino: {acc_treino:.4f}\n")
        f.write(f"Acurácia no Teste: {acc_teste:.4f}\n")
        f.write(f"Gap (treino - teste): {gap:.4f} ")
        f.write("(sem overfitting significativo)\n" if gap < 0.05 else "(overfitting moderado)\n")
        f.write(f"F1 Emergência (teste): {f1_emerg:.4f}\n")
        f.write(f"AUC-ROC: {auc:.4f}\n")

    with open(f"{OUTPUT_DIR}/nb_relatorio_classificacao.txt", "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE CLASSIFICAÇÃO — NAIVE BAYES ===\n\n")
        f.write("Modelo: GaussianNB\n\n")
        f.write(classification_report(
            y_test, y_pred,
            target_names=["Não-Emergência (0)", "Emergência (1)"]
        ))

    print(f"- Métricas salvas: {OUTPUT_DIR}/nb_metricas.txt")
    print(f"- Relatório salvo: {OUTPUT_DIR}/nb_relatorio_classificacao.txt")

    # ── Matriz de Confusão ─────────────────────────────────────
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não-Emergência", "Emergência"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title("Matriz de Confusão — Naive Bayes\nGaussianNB")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nb_matriz_confusao.png", dpi=150)
    plt.close()
    print(f"- Matriz de confusão: {OUTPUT_DIR}/nb_matriz_confusao.png")

    # ── Curva ROC ──────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E53935", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.set_title("Curva ROC — Naive Bayes\nGaussianNB")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nb_curva_roc.png", dpi=150)
    plt.close()
    print(f"- Curva ROC: {OUTPUT_DIR}/nb_curva_roc.png")

    return modelo, acc_treino, acc_teste, f1_emerg, auc


if __name__ == "__main__":
    print("=" * 57)
    print("Naive Bayes (GaussianNB)")
    print("=" * 57 + "\n")

    train, test = carregar_dados()
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)

    treinar_e_avaliar(X_train, y_train, X_test, y_test)

    print(f"\nResultados salvos em '{OUTPUT_DIR}/'")