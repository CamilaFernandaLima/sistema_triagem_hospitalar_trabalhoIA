"""
Árvore de Decisão com controle de overfitting
=======================================================================
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência
Tarefa : Testar critérios (gini, entropy, log_loss) x profundidades via
         validação cruzada, escolher a melhor combinação pelo F1 da classe
         Emergência (classe crítica — minimizar falsos negativos).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, roc_auc_score, roc_curve,
    f1_score
)
from sklearn.model_selection import cross_val_score
import os
import warnings
warnings.filterwarnings("ignore")
import json
import time
from util import carregar_dados, separar_xy, TARGET_COLUMN, FEATURES


# CONFIGURAÇÃO

TRAIN_PATH    = "data/data_split/train.csv"
TEST_PATH     = "data/data_split/test.csv"
OUTPUT_DIR    = "resultados/dt"
RESUMO_TXT    = f"{OUTPUT_DIR}/dt_resumo.txt"
RANDOM_STATE  = 42

CRITERIOS    = ["gini", "entropy", "log_loss"]
PROFUNDIDADES = [2, 3, 4, 5, 6, 7, 8, 10, None]



def buscar_melhor_combinacao(X_train, y_train):
    """
    Testa todos os critérios x profundidades via CV 5-fold.
    Métrica de seleção: F1 da classe Emergência (1) — evitar falsos negativos.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[DT]>>[DT]Buscando melhor critério x profundidade (5-fold CV, métrica: F1-Emergência)...\n")
    print("Árvore de Decisão (anti-overfitting)\n")
    print("Critérios: gini, entropy, log_loss\n")
    print("Métrica de seleção: F1 da classe Emergência (1)\n\n")
    print(f"{'Critério':<12} {'Profund.':<12} {'F1-Emerg.':<12} {'Desvio'}\n")
    print("-" * 48 + "\n")

    
    resultados = []

    for criterio in CRITERIOS:
        for prof in PROFUNDIDADES:
            modelo = DecisionTreeClassifier(
                criterion=criterio,
                max_depth=prof,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
            # f1 com pos_label=1 = F1 da classe Emergência
            scores = cross_val_score(
                modelo, X_train, y_train, cv=5,
                scoring="f1"          # f1 binário — pos_label=1 por padrão
            )
            resultados.append({
                "criterio": criterio,
                "max_depth": prof,
                "label": str(prof) if prof else "Sem limite",
                "f1_media": scores.mean(),
                "f1_desvio": scores.std()
            })
            print(f"   {criterio:<12} {str(prof):<12} {scores.mean():.4f}       ± {scores.std():.4f}")

        print()  # separa blocos por critério

    df_res = pd.DataFrame(resultados)
    melhor = df_res.loc[df_res["f1_media"].idxmax()]
    melhor_criterio = melhor["criterio"]
    melhor_prof = melhor["max_depth"]
    if melhor_prof is not None and not isinstance(melhor_prof, int):
        melhor_prof = int(melhor_prof)

    with open (f"{RESUMO_TXT}", "a") as f:
        f.write(f"Melhor configuração:\n")
        f.write(f"Critério    : {melhor_criterio}\n")
        f.write(f"Profundidade: {melhor_prof}\n")
        f.write(f"F1-Emergência (CV): {melhor['f1_media']:.4f} ± {melhor['f1_desvio']:.4f}\n")
        f.write(f"\n")

    # >>>> Gráfico comparativo: F1 x profundidade por critério
    cores = {"gini": "#1976D2", "entropy": "#E53935", "log_loss": "#43A047"}
    fig, ax = plt.subplots(figsize=(12, 5))

    for criterio in CRITERIOS:
        sub = df_res[df_res["criterio"] == criterio]
        ax.plot(sub["label"], sub["f1_media"], marker="o",
                label=criterio, color=cores[criterio], linewidth=2)
        ax.fill_between(
            sub["label"],
            sub["f1_media"] - sub["f1_desvio"],
            sub["f1_media"] + sub["f1_desvio"],
            alpha=0.12, color=cores[criterio]
        )

    ax.axvline(x=str(melhor_prof) if melhor_prof else "Sem limite",
               color="black", linestyle="--", linewidth=1.2, label="Melhor config.")
    ax.set_title("F1-Emergência x Profundidade por Critério (Validação Cruzada 5-fold)")
    ax.set_xlabel("Profundidade máxima (max_depth)")
    ax.set_ylabel("F1-Score — Classe Emergência (1)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dt_busca_criterio_profundidade.png", dpi=150)
    plt.close()
    print(f"[DT]>>Gráfico comparativo salvo: {OUTPUT_DIR}/dt_busca_criterio_profundidade.png\n")

    return melhor_criterio, melhor_prof, df_res


def treinar_e_avaliar(X_train, y_train, X_test, y_test, criterio, prof):
    print("=" * 57)
    print(f"  TREINAMENTO FINAL  (critério={criterio}, max_depth={prof})")
    print("=" * 57 + "\n")

    inicio_treino = time.time()

    modelo = DecisionTreeClassifier(
        criterion=criterio,
        max_depth=prof,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    # >>>> Métricas principais
    acc_treino = accuracy_score(y_train, modelo.predict(X_train))
    acc_teste  = accuracy_score(y_test, y_pred)
    f1_emerg   = f1_score(y_test, y_pred, pos_label=1)
    auc        = roc_auc_score(y_test, y_prob)
    gap        = acc_treino - acc_teste

    fim_treino = time.time()

    tempo_total = fim_treino - inicio_treino

    json_resumo = {
        "Modelo": " Árvore de Decisão",
        "Acuracia_Treino": float(acc_treino),
        "Acuracia_Teste":  float(acc_teste),
        "Gap":  float(gap),
        "F1_Emergencia": float(f1_emerg),
        "AUC_ROC": float(auc),
        "tempo_treino": float(tempo_total),
        "Parametros": {
            "profundidade": prof,
            "criterio": criterio
        }
    }

    json_path = f"{OUTPUT_DIR}/dt_resumo.json"
    with open(json_path, "w") as f:
        json.dump(json_resumo, f, indent=4)
    print(f"[DT]>>Métricas salvas em JSON: {json_path}\n")


    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open (f"{RESUMO_TXT}", "a", encoding="utf-8") as f:
        f.write(f"Critério: {criterio}\n")
        f.write(f"Profundidade: {prof}\n")
        f.write(f"Acurácia no Treino     : {acc_treino:.4f}\n")
        f.write(f"Acurácia no Teste      : {acc_teste:.4f}\n")
        f.write(f"Gap (treino - teste)   : {gap:.4f} " + ("sem overfitting significativo" if gap < 0.05 else "overfitting moderado"))
        f.write(f"F1 Emergência (teste)  : {f1_emerg:.4f}\n")
        f.write(f"AUC-ROC                : {auc:.4f}\n")
        f.write(f"Critério: {criterio}\n")
        f.write(f"Profundidade: {prof}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_test, y_pred,
              target_names=["Não-Emergência (0)", "Emergência (1)"]))
        

    
    feature_names = X_train.columns.tolist()

    # >>>> Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não-Emergência", "Emergência"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusão\ncritério={criterio}  max_depth={prof}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dt_matriz_confusao.png", dpi=150)
    plt.close()
    print(f"[DT]>>Matriz de confusão        : {OUTPUT_DIR}/dt_matriz_confusao.png")

    # >>>> Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E53935", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.set_title(f"Curva ROC — Árvore de Decisão\ncritério={criterio}  max_depth={prof}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dt_curva_roc.png", dpi=150)
    plt.close()
    print(f"[DT]>>Curva ROC                 : {OUTPUT_DIR}/dt_curva_roc.png")

    # >>>> Importância das variáveis
    importancias = pd.Series(modelo.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(importancias.index, importancias.values, color="#43A047")
    for bar, val in zip(bars, importancias.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Importância (critério: " + criterio + ")")
    ax.set_title(f"Importância das Variáveis — Árvore de Decisão\ncritério={criterio}  max_depth={prof}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/dt_importancia_variaveis.png", dpi=150)
    plt.close()
    print(f"[DT]>>Importância das variáveis : {OUTPUT_DIR}/dt_importancia_variaveis.png")

    # >>>> Visualização da árvore e regras (só se profundidade ≤ 5)
    prof_num = prof if prof is not None else 999
    if prof_num <= 5:
        fig, ax = plt.subplots(figsize=(22, 9))
        plot_tree(modelo, ax=ax, feature_names=feature_names,
                  class_names=["Não-Emergência", "Emergência"],
                  filled=True, rounded=True, fontsize=9)
        ax.set_title(f"Árvore de Decisão — critério={criterio}  max_depth={prof}", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/dt_arvore.png", dpi=110)
        plt.close()
        print(f"[DT]>>Visualização da árvore    : {OUTPUT_DIR}/dt_arvore.png")

        regras = export_text(modelo, feature_names=feature_names)
        with open(f"{OUTPUT_DIR}/dt_regras.txt", "a", encoding="utf-8") as f:
            f.write(f"Critério: {criterio}  |  max_depth: {prof}\n\n")
            f.write(regras)

        print(f"[DT]>>Regras em texto           : {OUTPUT_DIR}/dt_regras.txt")
    else:
        print("   (Árvore muito profunda para visualizar — omitido)")

    return modelo, acc_treino, acc_teste, f1_emerg, auc


if __name__ == "__main__":
    print("=" * 57)
    print("Árvore de Decisão (anti-overfitting)")
    print("  Seleção: F1-Emergência  |  Critérios: gini, entropy, log_loss")
    print("=" * 57 + "\n")

    train, test = carregar_dados(TRAIN_PATH,TEST_PATH)
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)


    melhor_criterio, melhor_prof, df_busca = buscar_melhor_combinacao(X_train, y_train)

    modelo, acc_treino, acc_teste, f1_emerg, auc = treinar_e_avaliar(
        X_train, y_train, X_test, y_test, melhor_criterio, melhor_prof
    )

    with open(f"{RESUMO_TXT}", "a", encoding="utf-8") as f:
        f.write(f"Critério escolhido    : {melhor_criterio}\n")
        f.write(f"Profundidade escolhida: {melhor_prof}\n")
        f.write(f"F1-Emergência (teste) : {f1_emerg:.4f}\n")
        f.write(f"AUC-ROC               : {auc:.4f}\n")

    print(f"\n>>Concluído! Resultados em '{OUTPUT_DIR}/'")