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
    f1_score, recall_score, precision_score
)
import os
import warnings
warnings.filterwarnings("ignore")

import json

from util import carregar_dados, separar_xy, FEATURES, TARGET_COLUMN

# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
TRAIN_PATH = "data/data_split/train_normalized.csv"
TEST_PATH = "data/data_split/test_normalized.csv"
OUTPUT_DIR = "resultados/nb"
RANDOM_STATE = 42
THRESHOLDS = np.arange(0.15, 0.61, 0.05)


def treinar_e_avaliar(X_train, y_train, X_test, y_test, melhor_threshold):
    print("=" * 57)
    print("  TREINAMENTO NAIVE BAYES  (GaussianNB)")
    print("=" * 57 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    modelo = GaussianNB()
    modelo.fit(X_train, y_train)

    y_prob = modelo.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= melhor_threshold).astype(int)

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

    json_resumo = {
        "Modelo": "Naive Bayes",
        "Acuracia_Treino": float(acc_treino),
        "Acuracia_Teste":  float(acc_teste),
        "Gap":  float(gap),
        "F1_Emergencia": float(f1_emerg),
        "AUC_ROC": float(auc),
        "Parametros": {
            "Modelo": "GaussianNB",
            "threshold": float(melhor_threshold)
        }
    }
    with open (f"{OUTPUT_DIR}/nb_resumo.json", "w", encoding="utf-8") as f:

        json.dump(json_resumo, f, indent=4)

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
    print("Naive Bayes (GaussianNB)")
    print("=" * 57 + "\n")
    print(">>Busca do Melhor Threshold - Naive Bayes")

    train, test = carregar_dados(TRAIN_PATH, TEST_PATH)
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)

    melhor_threshold = buscar_melhor_threshold(X_train, y_train, X_test, y_test)

    treinar_e_avaliar(X_train, y_train, X_test, y_test, melhor_threshold)

    print(f"\nResultados salvos em '{OUTPUT_DIR}/'")