"""
Busca do Melhor Threshold — Naive Bayes
=======================================================================
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência
Testa thresholds de 0.20 a 0.60 e avalia F1, Recall e Precisão.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import os
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
TRAIN_PATH = "data/data_split/train_normalized.csv"
TEST_PATH = "data/data_split/test_normalized.csv"
TARGET_COLUMN = "KTAS_target_binario"
OUTPUT_DIR = "resultados/nb"

FEATURES = [
    "Sex", "Age", "Injury", "Mental", "Pain", "NRS_pain",
    "SBP", "DBP", "HR", "RR", "BT", "Saturation"
]

THRESHOLDS = np.arange(0.10, 0.61, 0.05)

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


def buscar_melhor_threshold(X_train, y_train, X_test, y_test):
    print("=" * 57)
    print("BUSCA DO MELHOR THRESHOLD — NAIVE BAYES")
    print("=" * 57 + "\n")
    print(f"   {'Threshold':<12} {'F1-Emerg.':<14} {'Recall':<12} {'Precisão':<12} {'Acur. Teste'}")
    print("   " + "-" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    modelo = GaussianNB()
    modelo.fit(X_train, y_train)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    resultados = []

    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)

        f1 = f1_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_test, y_pred)

        resultados.append({
            "Threshold": round(threshold, 2),
            "F1_Emergencia": f1,
            "Recall": recall,
            "Precisao": precision,
            "Acuracia_Teste": acc
        })

        print(f"   {threshold:.2f}         {f1:.4f}         {recall:.4f}       {precision:.4f}       {acc:.4f}")

    df_res = pd.DataFrame(resultados)

    # ── Melhor threshold por F1 ───────────────────────────────
    melhor = df_res.loc[df_res["F1_Emergencia"].idxmax()]
    print(f"\n>> Melhor threshold por F1-Emergência: {melhor['Threshold']:.2f}  "
          f"(F1={melhor['F1_Emergencia']:.4f}  Recall={melhor['Recall']:.4f})")

    # ── Salvar tabela em txt ──────────────────────────────────
    with open(f"{OUTPUT_DIR}/nb_busca_threshold.txt", "w", encoding="utf-8") as f:
        f.write("=== BUSCA DO MELHOR THRESHOLD — NAIVE BAYES ===\n\n")
        f.write(f"{'Threshold':<12} {'F1-Emerg.':<14} {'Recall':<12} {'Precisão':<12} {'Acur. Teste'}\n")
        f.write("-" * 62 + "\n")
        for _, row in df_res.iterrows():
            f.write(f"{row['Threshold']:<12.2f} {row['F1_Emergencia']:<14.4f} "
                    f"{row['Recall']:<12.4f} {row['Precisao']:<12.4f} {row['Acuracia_Teste']:.4f}\n")
        f.write("\n" + "-" * 62 + "\n")
        f.write(f"Melhor Threshold       : {melhor['Threshold']:.2f}\n")
        f.write(f"F1-Emergência          : {melhor['F1_Emergencia']:.4f}\n")
        f.write(f"Recall                 : {melhor['Recall']:.4f}\n")
        f.write(f"Precisão               : {melhor['Precisao']:.4f}\n")
        f.write(f"Acurácia Teste         : {melhor['Acuracia_Teste']:.4f}\n")

    print(f">>Tabela salva            : {OUTPUT_DIR}/nb_busca_threshold.txt")

    # ── Gráfico F1, Recall e Precisão por Threshold ───────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df_res["Threshold"], df_res["F1_Emergencia"], marker="o",
            color="#E53935", lw=2, label="F1-Emergência")
    ax.plot(df_res["Threshold"], df_res["Recall"], marker="s", linestyle="--",
            color="#1976D2", lw=2, label="Recall")
    ax.plot(df_res["Threshold"], df_res["Precisao"], marker="^", linestyle="--",
            color="#43A047", lw=2, label="Precisão")

    ax.axvline(x=melhor["Threshold"], color="black", linestyle="--",
               lw=1.2, label=f"Melhor threshold={melhor['Threshold']:.2f}")
    ax.axvline(x=0.50, color="gray", linestyle=":", lw=1, label="Threshold padrão (0.50)")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("F1, Recall e Precisão por Threshold — Naive Bayes\n(quanto menor o threshold, mais casos classificados como Emergência)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nb_busca_threshold.png", dpi=150)
    plt.close()
    print(f">>Gráfico salvo: {OUTPUT_DIR}/nb_busca_threshold.png")

    return melhor["Threshold"]


if __name__ == "__main__":
    print("=" * 57)
    print("Busca do Melhor Threshold - Naive Bayes")
    print("=" * 57 + "\n")

    train, test = carregar_dados()
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)

    melhor_threshold = buscar_melhor_threshold(X_train, y_train, X_test, y_test)

    print(f"\nUse threshold={melhor_threshold:.2f} no nb_treino.py para o modelo final!")
    print(f"\nResultados salvos em '{OUTPUT_DIR}/'")