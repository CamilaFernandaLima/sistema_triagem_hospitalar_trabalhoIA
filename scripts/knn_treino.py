"""
KNN — K-Nearest Neighbors
=======================================================================
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, roc_auc_score, roc_curve,
    f1_score
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
OUTPUT_DIR = "resultados/knn"
RANDOM_STATE = 42

K_RANGE = range(1, 50, 2)   # K ímpares de 1 a 49
METRIC = "euclidean"


def treinar_e_avaliar(X_train, y_train, X_test, y_test,melhor_k):
    print("=" * 57)
    print(f"TREINAMENTO KNN  (K={melhor_k}, métrica={METRIC})")
    print("=" * 57 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    modelo = KNeighborsClassifier(n_neighbors=melhor_k, metric=METRIC, weights='uniform')
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # ── Métricas Principais ─────────────────────────────────────
    acc_treino = accuracy_score(y_train, modelo.predict(X_train))
    acc_teste = accuracy_score(y_test, y_pred)
    f1_emerg = f1_score(y_test, y_pred, pos_label=1)
    auc = roc_auc_score(y_test, y_prob)
    gap = acc_treino - acc_teste

    json_resumo = {
        "Modelo": " K-Nearest Neighbors",
        "Acuracia_Treino": float(acc_treino),
        "Acuracia_Teste":  float(acc_teste),
        "Gap":  float(gap),
        "F1_Emergencia": float(f1_emerg),
        "AUC_ROC": float(auc),
        "Parametros": {
            "K": melhor_k,
            "Metrica": METRIC
        }
    }
    with open(f"{OUTPUT_DIR}/knn_resumo.json", "w", encoding="utf-8") as f:
        json.dump(json_resumo, f, indent=4, ensure_ascii=False)

    with open(f"{OUTPUT_DIR}/knn_metricas.txt", "w", encoding="utf-8") as f:
        f.write("=== MÉTRICAS — KNN ===\n\n")
        f.write(f"K : {melhor_k}\n")
        f.write(f"Métrica de distância : {METRIC}\n\n")
        f.write(f"Acurácia no Treino : {acc_treino:.4f}\n")
        f.write(f"Acurácia no Teste : {acc_teste:.4f}\n")
        f.write(f"Gap (treino - teste) : {gap:.4f} ")
        f.write("(sem overfitting significativo)\n" if gap < 0.05 else "(overfitting moderado)\n")
        f.write(f"F1 Emergência (teste) : {f1_emerg:.4f}\n")
        f.write(f"AUC-ROC : {auc:.4f}\n")

    with open(f"{OUTPUT_DIR}/knn_relatorio_classificacao.txt", "w", encoding="utf-8") as f:
        f.write("=== RELATÓRIO DE CLASSIFICAÇÃO — KNN ===\n\n")
        f.write(f"K={melhor_k}  |  métrica={METRIC}\n\n")
        f.write(classification_report(
            y_test, y_pred,
            target_names=["Não-Emergência (0)", "Emergência (1)"]
        ))

    print(f"- Métricas salvas: {OUTPUT_DIR}/knn_metricas.txt")
    print(f"- Relatório salvo: {OUTPUT_DIR}/knn_relatorio_classificacao.txt")

    # ── Matriz de Confusão ─────────────────────────────────────
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não-Emergência", "Emergência"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusão — KNN\nK={melhor_k}  métrica={METRIC}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/knn_matriz_confusao.png", dpi=150)
    plt.close()
    print(f"- Matriz de confusão: {OUTPUT_DIR}/knn_matriz_confusao.png")

    # ── Curva ROC ──────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1976D2", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.set_title(f"Curva ROC — KNN\nK={melhor_k}  métrica={METRIC}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/knn_curva_roc.png", dpi=150)
    plt.close()
    print(f"- Curva ROC: {OUTPUT_DIR}/knn_curva_roc.png")

    return modelo, acc_treino, acc_teste, f1_emerg, auc

def buscar_melhor_k(X_train, y_train, X_test, y_test):
    print("=" * 57)
    print("BUSCA DO MELHOR K — KNN")
    print("=" * 57 + "\n")
    print(f"   {'K':<6} {'F1-Emerg.':<14} {'Acur. Treino':<16} {'Acur. Teste':<14} {'Gap'}")
    print("   " + "-" * 58)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    resultados = []

    for k in K_RANGE:
        modelo = KNeighborsClassifier(n_neighbors=k, metric=METRIC, weights='uniform')
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)
        acc_treino = accuracy_score(y_train, modelo.predict(X_train))
        acc_teste = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        gap = acc_treino - acc_teste

        resultados.append({
            "K": k,
            "F1_Emergencia": f1,
            "Acuracia_Treino": acc_treino,
            "Acuracia_Teste": acc_teste,
            "Gap": gap
        })

        print(f"   K={k:<4} {f1:.4f}         {acc_treino:.4f}           {acc_teste:.4f}         {gap:.4f}")

    df_res = pd.DataFrame(resultados)

    # ── Melhor K por F1 ───────────────────────────────────────
    melhor = df_res.loc[df_res["F1_Emergencia"].idxmax()]
    print(f"\n>> Melhor K por F1-Emergência: K={int(melhor['K'])}  "
          f"(F1={melhor['F1_Emergencia']:.4f}  Gap={melhor['Gap']:.4f})")

    # ── Salvar tabela em txt ───────────────────────────────────
    with open(f"{OUTPUT_DIR}/knn_busca_k.txt", "w", encoding="utf-8") as f:
        f.write("=== BUSCA DO MELHOR K — KNN ===\n\n")
        f.write(f"{'K':<6} {'F1-Emerg.':<14} {'Acur. Treino':<16} {'Acur. Teste':<14} {'Gap'}\n")
        f.write("-" * 58 + "\n")
        for _, row in df_res.iterrows():
            f.write(f"K={int(row['K']):<4} {row['F1_Emergencia']:.4f}         "
                    f"{row['Acuracia_Treino']:.4f}           "
                    f"{row['Acuracia_Teste']:.4f}         "
                    f"{row['Gap']:.4f}\n")
        f.write("\n" + "-" * 58 + "\n")
        f.write(f"Melhor K por F1-Emergência: K={int(melhor['K'])}\n")
        f.write(f"F1-Emergência: {melhor['F1_Emergencia']:.4f}\n")
        f.write(f"Acurácia Treino: {melhor['Acuracia_Treino']:.4f}\n")
        f.write(f"Acurácia Teste: {melhor['Acuracia_Teste']:.4f}\n")
        f.write(f"Gap: {melhor['Gap']:.4f}\n")

    print(f"- Tabela salva: {OUTPUT_DIR}/knn_busca_k.txt")

    # ── Gráfico F1 e Gap por K ────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(df_res["K"], df_res["F1_Emergencia"], marker="o",
             color="#1976D2", lw=2, label="F1-Emergência")
    ax1.set_xlabel("K (número de vizinhos)")
    ax1.set_ylabel("F1-Emergência", color="#1976D2")
    ax1.tick_params(axis="y", labelcolor="#1976D2")

    ax2 = ax1.twinx()
    ax2.plot(df_res["K"], df_res["Gap"], marker="s", linestyle="--",
             color="#E53935", lw=2, label="Gap (treino - teste)")
    ax2.set_ylabel("Gap (treino - teste)", color="#E53935")
    ax2.tick_params(axis="y", labelcolor="#E53935")
    ax2.axhline(y=0.10, color="#E53935", linestyle=":", lw=1, alpha=0.5)

    ax1.axvline(x=int(melhor["K"]), color="black", linestyle="--",
                lw=1.2, label=f"Melhor K={int(melhor['K'])}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("F1-Emergência e Gap por K — KNN\n(linha pontilhada vermelha = gap 0.10)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/knn_busca_k.png", dpi=150)
    plt.close()
    print(f"- Gráfico salvo: {OUTPUT_DIR}/knn_busca_k.png")

    return int(melhor["K"])


if __name__ == "__main__":
    print("=" * 57)

    print("=" * 57)
    print("K-Nearest Neighbors (KNN)")
    print("=" * 57 + "\n")

    print(">>Busca do Melhor K — KNN")

    train, test = carregar_dados(TRAIN_PATH, TEST_PATH)
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)

    melhor_k = buscar_melhor_k(X_train, y_train, X_test, y_test)

    treinar_e_avaliar(X_train, y_train, X_test, y_test, melhor_k)

    print(f"\nResultados salvos em '{OUTPUT_DIR}/'")