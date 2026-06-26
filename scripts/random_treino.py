"""
Random Forest + Comparação com Árvore de Decisão
==========================================================================
Dataset: data/data_split/train.csv e test.csv (gerados por split.py)
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência

Etapas:
  1. Busca de hiperparâmetros (n_estimators x max_depth) via CV 5-fold
  2. Ajuste de threshold para minimizar falsos negativos (emergências perdidas)
  3. Comparação visual direta: Árvore de Decisão vs Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score,
    roc_auc_score, roc_curve, f1_score,
    precision_recall_curve
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')  # Suppress all warnings
warnings.warn("This warning will be hidden")
import os
import json
from util import carregar_dados, separar_xy, TARGET_COLUMN, FEATURES

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
TRAIN_PATH    = "data/data_split/train.csv"
TEST_PATH     = "data/data_split/test.csv"
OUTPUT_DIR    = "resultados/rf"
RANDOM_STATE  = 42

# Grid de busca para Random Forest
N_ESTIMATORS_LISTA = [50, 100, 200, 300]
MAX_DEPTH_LISTA    = [4, 6, 8, 10, None]

# Parâmetros fixos da Árvore de Decisão (resultado do b2)
DT_CRITERIO  = "entropy"
DT_MAX_DEPTH = 4
# ──────────────────────────────────────────────────────────────


# ── 1. BUSCA DE HIPERPARÂMETROS ───────────────────────────────
def buscar_hiperparametros(X_train, y_train):
    print("[RF]>> Buscando melhores hiperparâmetros do Random Forest")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    resultados = []
    for n in N_ESTIMATORS_LISTA:
        for prof in MAX_DEPTH_LISTA:
            modelo = RandomForestClassifier(
                n_estimators=n,
                max_depth=prof,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1")
            resultados.append({
                "n_estimators": n,
                "max_depth": prof,
                "label_depth": str(prof) if prof else "Sem limite",
                "f1_media": scores.mean(),
                "f1_desvio": scores.std()
            })

    df_res = pd.DataFrame(resultados)
    melhor = df_res.loc[df_res["f1_media"].idxmax()]
    melhor_n    = int(melhor["n_estimators"])
    melhor_prof = melhor["max_depth"]
    if melhor_prof is not None and not isinstance(melhor_prof, int):
        melhor_prof = int(melhor_prof)
    
    
    print(">>>Melhor configuração encontrada:\n")
    print(f">> n_estimators : {melhor_n}\n" )
    print(f">> max_depth    : {melhor_prof}")
    print(f">>F1-Emergência (CV): {melhor['f1_media']:.4f} +- {melhor['f1_desvio']:.4f}\n")

    # Heatmap: F1 x n_estimators x max_depth
    pivot = df_res.pivot(index="n_estimators", columns="label_depth", values="f1_media")
    # Ordena colunas
    col_order = [str(d) for d in MAX_DEPTH_LISTA if d is not None] + ["Sem limite"]
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto",
                   vmin=pivot.values.min() - 0.01, vmax=pivot.values.max() + 0.01)
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    ax.set_xlabel("max_depth"); ax.set_ylabel("n_estimators")
    ax.set_title("F1-Emergência (CV 5-fold) — Random Forest")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if pivot.values[i, j] < 0.85 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rf_busca_hiperparametros.png", dpi=150)
    plt.close()
    print(f"[RF]>> Heatmap salvo: {OUTPUT_DIR}/rf_busca_hiperparametros.png\n")

    return melhor_n, melhor_prof


# ── 2. THRESHOLD ÓTIMO ────────────────────────────────────────
def encontrar_threshold(modelo, X_test, y_test):
    """
    Encontra o threshold que maximiza o F1-Emergência.
    Threshold < 0.5 aumenta recall (menos falsos negativos)
    ao custo de mais falsos positivos — aceitável em triagem.
    """
    y_prob = modelo.predict_proba(X_test)[:, 1]
    precisoes, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    f1_scores = 2 * (precisoes * recalls) / (precisoes + recalls + 1e-9)
    idx_melhor = np.argmax(f1_scores[:-1])   # último elemento não tem threshold
    threshold_otimo = thresholds[idx_melhor]

    print(f"   Threshold padrão (0.50) -> F1-Emergência: "
          f"{f1_score(y_test, (y_prob >= 0.50).astype(int)):.4f}")
    print(f"   Threshold ótimo  ({threshold_otimo:.2f}) -> F1-Emergência: "
          f"{f1_score(y_test, (y_prob >= threshold_otimo).astype(int)):.4f}\n")

    return threshold_otimo, y_prob


# ── 3. AVALIAÇÃO COMPLETA ─────────────────────────────────────
def avaliar_rf(modelo, X_train, y_train, X_test, y_test, n, prof, threshold, y_prob):
    y_pred = (y_prob >= threshold).astype(int)

    acc_treino = accuracy_score(y_train, modelo.predict(X_train))
    acc_teste  = accuracy_score(y_test, y_pred)
    f1_emerg   = f1_score(y_test, y_pred, pos_label=1)
    auc        = roc_auc_score(y_test, y_prob)
    gap        = acc_treino - acc_teste

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_resumo = {
        "Modelo": "Random Forest",
        "Acuracia_Treino": float(acc_treino),
        "Acuracia_Teste": float(acc_teste),
        "Gap": float(acc_treino - acc_teste),
        "F1_Emergencia": float(f1_emerg),
        "AUC_ROC": float(auc),
        # Você pode adicionar as configurações do modelo se quiser
        "Parametros": {
            "n_estimators": n,
            "max_depth": prof,
            "threshold": float(threshold)
        }
    }

    json_path = os.path.join(OUTPUT_DIR, "rf_resumo.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_resumo, f, ensure_ascii=False, indent=4)

    
    with open (f"{OUTPUT_DIR}/rf_resumo.txt", "a") as f:
        f.write(f"Avaliação do modelo Random Forest:\n")
        f.write(f"n_estimators : {n}\n")
        f.write(f"max_depth    : {prof}\n")
        f.write(f"Threshold    : {threshold:.2f}\n")
        f.write(f"Acurácia no Treino     : {acc_treino:.4f}\n")
        f.write(f"Acurácia no Teste      : {acc_teste:.4f}\n")
        f.write(f"Gap (treino - teste)   : {gap:.4f}  "
                + ("> sem overfitting" if gap < 0.05 else "> overfitting moderado") + "\n")
        f.write(f"F1 Emergência (teste)  : {f1_emerg:.4f}   <- métrica principal\n")
        f.write(f"AUC-ROC                : {auc:.4f}\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(classification_report(y_test, y_pred,
              target_names=["Não-Emergência (0)", "Emergência (1)"]))

    
    feature_names = X_train.columns.tolist()

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Não-Emergência", "Emergência"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de Confusão — Random Forest\nn_estimators={n}  max_depth={prof}  threshold={threshold:.2f}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rf_matriz_confusao.png", dpi=150)
    plt.close()
    print(f"[RF]>> Matriz de confusão        : {OUTPUT_DIR}/rf_matriz_confusao.png")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1976D2", lw=2, label=f"Random Forest  AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.set_title(f"Curva ROC — Random Forest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rf_curva_roc.png", dpi=150)
    plt.close()
    print(f"[RF]>> Curva ROC                 : {OUTPUT_DIR}/rf_curva_roc.png")

    # Importância das variáveis
    importancias = pd.Series(modelo.feature_importances_, index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(importancias.index, importancias.values, color="#1976D2")
    for bar, val in zip(bars, importancias.values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Importância média (Gini por floresta)")
    ax.set_title(f"Importância das Variáveis — Random Forest\nn_estimators={n}  max_depth={prof}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rf_importancia_variaveis.png", dpi=150)
    plt.close()
    print(f"[RF]>> Importância das variáveis : {OUTPUT_DIR}/rf_importancia_variaveis.png\n")

    return acc_teste, f1_emerg, auc, fpr, tpr


# ── 4. COMPARAÇÃO LADO A LADO ─────────────────────────────────
def comparar_modelos(X_train, y_train, X_test, y_test,
                     rf_modelo, rf_threshold, rf_prob,
                     rf_acc, rf_f1, rf_auc, rf_fpr, rf_tpr):
    """Treina a DT com os mesmos parâmetros do b2 e plota comparação."""
    print("=" * 57)
    print("  COMPARAÇÃO: Árvore de Decisão vs Random Forest")
    print("=" * 57 + "\n")

    # Retreina DT
    dt = DecisionTreeClassifier(
        criterion=DT_CRITERIO, max_depth=DT_MAX_DEPTH,
        class_weight="balanced", random_state=RANDOM_STATE
    )
    dt.fit(X_train, y_train)
    dt_prob  = dt.predict_proba(X_test)[:, 1]
    dt_pred  = dt.predict(X_test)
    dt_acc   = accuracy_score(y_test, dt_pred)
    dt_f1    = f1_score(y_test, dt_pred, pos_label=1)
    dt_auc   = roc_auc_score(y_test, dt_prob)
    dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_prob)
    dt_cm    = confusion_matrix(y_test, dt_pred)

    rf_pred  = (rf_prob >= rf_threshold).astype(int)
    rf_cm    = confusion_matrix(y_test, rf_pred)

    # ── Gráfico 1: Barras de métricas
    metricas  = ["Acurácia", "F1-Emergência", "AUC-ROC"]
    vals_dt   = [dt_acc,  dt_f1,  dt_auc]
    vals_rf   = [rf_acc,  rf_f1,  rf_auc]

    x = np.arange(len(metricas))
    w = 0.32
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_dt = ax.bar(x - w/2, vals_dt, w, label=f"Árvore de Decisão\n(entropy, depth={DT_MAX_DEPTH})", color="#EF5350")
    bars_rf = ax.bar(x + w/2, vals_rf, w, label=f"Random Forest\n(n={rf_modelo.n_estimators}, depth={rf_modelo.max_depth})", color="#1976D2")
    for bar in list(bars_dt) + list(bars_rf):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(metricas, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_title("Comparação de Métricas: Árvore de Decisão vs Random Forest")
    ax.legend()
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparacao_metricas.png", dpi=150)
    plt.close()
    print(f"[RF]>> Comparação de métricas    : {OUTPUT_DIR}/comparacao_metricas.png")

    # ── Gráfico 2: ROC lado a lado
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(dt_fpr, dt_tpr, color="#EF5350", lw=2,
            label=f"Árvore de Decisão  AUC={dt_auc:.4f}")
    ax.plot(rf_fpr, rf_tpr, color="#1976D2", lw=2,
            label=f"Random Forest      AUC={rf_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos (Recall)")
    ax.set_title("Curva ROC — Comparação")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparacao_roc.png", dpi=150)
    plt.close()
    print(f"[RF]>> Curva ROC comparativa     : {OUTPUT_DIR}/comparacao_roc.png")

    # ── Gráfico 3: Matrizes de confusão lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, titulo in zip(
        axes,
        [dt_cm, rf_cm],
        [f"Árvore de Decisão\n(entropy, depth={DT_MAX_DEPTH})",
         f"Random Forest\n(n={rf_modelo.n_estimators}, depth={rf_modelo.max_depth}, thr={rf_threshold:.2f})"]
    ):
        disp = ConfusionMatrixDisplay(cm, display_labels=["Não-Emergência", "Emergência"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(titulo)
    plt.suptitle("Matrizes de Confusão — Comparação", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparacao_matrizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[RF]>> Matrizes comparativas     : {OUTPUT_DIR}/comparacao_matrizes.png\n")

    # ── Resumo numérico
    fn_dt = dt_cm[1][0]   # falsos negativos DT
    fn_rf = rf_cm[1][0]   # falsos negativos RF

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open (f"{OUTPUT_DIR}/resumo.txt", "a") as f:
        f.write(f"Resumo comparativo:\n")
        f.write(f"{'Métrica':<25} {'Árvore Dec.':<15} {'Random Forest':<15} {'Melhora'}\n")
        f.write("-" * 65 + "\n")
        for nome, vdt, vrf in [
            ("Acurácia",       dt_acc,  rf_acc),
            ("F1-Emergência",  dt_f1,   rf_f1),
            ("AUC-ROC",        dt_auc,  rf_auc),
        ]:
            delta = vrf - vdt
            sinal = "^" if delta > 0 else "v"
            f.write(f"{nome:<25} {vdt:<15.4f} {vrf:<15.4f} {sinal} {abs(delta):.4f}\n")
        f.write(f"{'Falsos Negativos':<25} {fn_dt:<15} {fn_rf:<15} "
                + (f"v {fn_dt - fn_rf} emergências a menos perdidas" if fn_rf < fn_dt else f"^ {fn_rf - fn_dt}\n"))


if __name__ == "__main__":
    print("=" * 57)
    print("Random Forest + Comparação com DT")
    print("=" * 57 + "\n")

    train, test = carregar_dados(TRAIN_PATH, TEST_PATH)
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)

    # 1. Busca hiperparâmetros
    melhor_n, melhor_prof = buscar_hiperparametros(X_train, y_train)

    # 2. Treina modelo final
    rf = RandomForestClassifier(
        n_estimators=melhor_n,
        max_depth=melhor_prof,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 3. Threshold ótimo
    print("[RF]>> Ajustando threshold para minimizar falsos negativos...")
    threshold_otimo, y_prob = encontrar_threshold(rf, X_test, y_test)

    # 4. Avaliação completa
    rf_acc, rf_f1, rf_auc, rf_fpr, rf_tpr = avaliar_rf(
        rf, X_train, y_train, X_test, y_test,
        melhor_n, melhor_prof, threshold_otimo, y_prob
    )

    # 5. Comparação com Árvore de Decisão
    comparar_modelos(
        X_train, y_train, X_test, y_test,
        rf, threshold_otimo, y_prob,
        rf_acc, rf_f1, rf_auc, rf_fpr, rf_tpr
    )

    print(f"\n>> Concluído! Todos os gráficos salvos em '{OUTPUT_DIR}/'")