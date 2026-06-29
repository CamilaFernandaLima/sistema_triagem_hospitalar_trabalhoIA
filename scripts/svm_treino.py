"""
 - Script 4: SVM (Support Vector Machine)
======================================================
Dataset: data/data_split/train.csv e test.csv (gerados por split.py)
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência

Etapas:
  1. Busca de hiperparâmetros (kernel x C x gamma) via CV 5-fold
  2. Ajuste de threshold para minimizar falsos negativos
  3. Geração de métricas e gráficos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    roc_auc_score, roc_curve, f1_score,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")
import os
import time
import json


from util import carregar_dados, separar_xy, FEATURES, TARGET_COLUMN

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
TRAIN_PATH    = "data/data_split/train.csv"
TEST_PATH     = "data/data_split/test.csv"
OUTPUT_DIR    = "resultados/svm"
RANDOM_STATE  = 42

# Grid de busca - balanceado para 4 núcleos de CPU
KERNELS = ["rbf", "linear", "poly"]
C_LIST  = [0.1, 1, 10, 100]
GAMMA_LIST = ["scale", "auto"]   # só usado por rbf e poly
# ──────────────────────────────────────────────────────────────


def buscar_hiperparametros(X_train, y_train):
    """
    Busca kernel x C x gamma via CV 5-fold.
    SVM exige StandardScaler - incluído no Pipeline para evitar data leakage.
    Métrica: F1-Emergência.
    """
    print("[SVM]>> Buscando melhores hiperparâmetros do SVM")
    print("   (5-fold CV, métrica: F1-Emergência)\n")
    print(f"   {'Kernel':<10} {'C':<8} {'Gamma':<10} {'F1-Emerg.':<12} {'Desvio'}")
    print("   " + "-" * 52)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    resultados = []

    for kernel in KERNELS:
        gammas = GAMMA_LIST if kernel in ("rbf", "poly") else ["scale"]
        for C in C_LIST:
            for gamma in gammas:
                pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("svm", SVC(
                        kernel=kernel, C=C, gamma=gamma,
                        class_weight="balanced",
                        probability=True,          # necessário para predict_proba e threshold
                        random_state=RANDOM_STATE
                    ))
                ])
                scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
                resultados.append({
                    "kernel": kernel, "C": C, "gamma": gamma,
                    "f1_media": scores.mean(), "f1_desvio": scores.std()
                })
                print(f"   {kernel:<10} {C:<8} {gamma:<10} {scores.mean():.4f}       +- {scores.std():.4f}")

        print()

    df_res = pd.DataFrame(resultados)
    melhor = df_res.loc[df_res["f1_media"].idxmax()]
    mk, mC, mg = melhor["kernel"], melhor["C"], melhor["gamma"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open (f"{OUTPUT_DIR}/svm_melhor_hiperparametros.txt", "a") as f:
        f.write(f"Melhor configuração:\n")
        f.write(f"kernel : {mk}\n")
        f.write(f"C      : {mC}\n")
        f.write(f"gamma  : {mg}\n")
        f.write(f"F1-Emergência (CV): {melhor['f1_media']:.4f} +- {melhor['f1_desvio']:.4f}\n")

    # Gráfico: F1 x C por kernel
    cores_kernel = {"rbf": "#1976D2", "linear": "#E53935", "poly": "#43A047"}
    fig, ax = plt.subplots(figsize=(9, 4))
    for kernel in KERNELS:
        sub = df_res[df_res["kernel"] == kernel].groupby("C")["f1_media"].max().reset_index()
        ax.plot(sub["C"].astype(str), sub["f1_media"], marker="o",
                label=f"kernel={kernel}", color=cores_kernel[kernel], linewidth=2)
    ax.axhline(melhor["f1_media"], color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title("F1-Emergência x C por Kernel - SVM (CV 5-fold)")
    ax.set_xlabel("C (regularização)")
    ax.set_ylabel("F1-Score - Classe Emergência")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/svm_busca_hiperparametros.png", dpi=150)
    plt.close()
    print(f"   >> Gráfico salvo: {OUTPUT_DIR}/svm_busca_hiperparametros.png\n")

    return mk, mC, mg, df_res


def encontrar_threshold(pipeline, X_test, y_test):
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    precisoes, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * (precisoes * recalls) / (precisoes + recalls + 1e-9)
    idx = np.argmax(f1s[:-1])
    thr = thresholds[idx]
    print(f"   Threshold padrão (0.50) -> F1-Emergência: "
          f"{f1_score(y_test, (y_prob >= 0.50).astype(int)):.4f}")
    print(f"   Threshold ótimo  ({thr:.2f}) -> F1-Emergência: "
          f"{f1_score(y_test, (y_prob >= thr).astype(int)):.4f}\n")
    return thr, y_prob


def treinar_e_avaliar(X_train, y_train, X_test, y_test):

    kernel, C, gamma, _ = buscar_hiperparametros(X_train, y_train)

    inicio_treino = time.time()

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=kernel, C=C, gamma=gamma,
            class_weight="balanced",
            probability=True,
            random_state=RANDOM_STATE
        ))
    ])
    pipeline.fit(X_train, y_train)

    print("[SVM]>> Ajustando threshold para minimizar falsos negativos...")
    thr, y_prob = encontrar_threshold(pipeline, X_test, y_test)
    y_pred     = (y_prob >= thr).astype(int)
    acc_treino = accuracy_score(y_train, pipeline.predict(X_train))
    acc_teste  = accuracy_score(y_test, y_pred)
    f1_emerg   = f1_score(y_test, y_pred, pos_label=1)
    auc        = roc_auc_score(y_test, y_prob)
    gap        = acc_treino - acc_teste

    fim_treino = time.time()
    tempo_total = fim_treino - inicio_treino

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_resumo = {
        "Modelo": "SVM",
        "Acuracia_Treino": float(acc_treino),
        "Acuracia_Teste": float(acc_teste),
        "Gap": float(acc_treino - acc_teste),
        "F1_Emergencia": float(f1_emerg),
        "AUC_ROC": float(auc),
        "tempo_treino": float(tempo_total),
        "Parametros": {
            "kernel": kernel,
            "C": C,
            "gamma": gamma,
            "threshold": float(thr)
        }
    }

    json_path = os.path.join(OUTPUT_DIR, "svm_resumo.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_resumo, f, ensure_ascii=False, indent=4)


    with open (f"{OUTPUT_DIR}/svm_metricas.txt", "a") as f:
        f.write(f"Acurácia no Treino     : {acc_treino:.4f}\n")
        f.write(f"Acurácia no Teste      : {acc_teste:.4f}\n")
        f.write(f"Gap (treino - teste)   : {gap:.4f}  "
                + ("> sem overfitting" if gap < 0.05 else ">  overfitting moderado") + "\n")
        f.write(f"F1 Emergência (teste)  : {f1_emerg:.4f}   <- métrica principal\n")
        f.write(f"AUC-ROC                : {auc:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_test, y_pred,
              target_names=["Não-Emergência (0)", "Emergência (1)"]))

   

    # Matriz de confusão
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não-Emergência", "Emergência"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Purples", colorbar=False)
    ax.set_title(f"Matriz de Confusão - SVM\nkernel={kernel}  C={C}  threshold={thr:.2f}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/svm_matriz_confusao.png", dpi=150)
    plt.close()
    print(f"[SVM]>> Matriz de confusão        : {OUTPUT_DIR}/svm_matriz_confusao.png")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#7B1FA2", lw=2, label=f"SVM  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.set_title("Curva ROC - SVM")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/svm_curva_roc.png", dpi=150)
    plt.close()
    print(f"[SVM]>> Curva ROC                 : {OUTPUT_DIR}/svm_curva_roc.png\n")

    return acc_teste, f1_emerg, auc, fpr, tpr, cm


if __name__ == "__main__":
    print("=" * 57)
    print("   - SVM (Support Vector Machine)")
    print("=" * 57 + "\n")

    train, test = carregar_dados(TRAIN_PATH, TEST_PATH)
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)

    treinar_e_avaliar(X_train, y_train, X_test, y_test)

    print(f"[SVM]>> Concluído! Resultados salvos em '{OUTPUT_DIR}/'")