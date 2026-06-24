"""
Busca do Melhor K — KNN
=======================================================================
Testa K de 1 a 49 com weights='uniform' e métrica euclidiana.
Avalia F1-Emergência e Gap (treino - teste) para cada K.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import os
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
TRAIN_PATH = "data/data_split/train_normalized.csv"
TEST_PATH = "data/data_split/test_normalized.csv"
TARGET_COLUMN = "KTAS_target_binario"
OUTPUT_DIR = "resultados/knn"

FEATURES = [
    "Sex", "Age", "Injury", "Mental", "Pain", "NRS_pain",
    "SBP", "DBP", "HR", "RR", "BT", "Saturation"
]

K_RANGE = range(1, 50, 2)   # K ímpares de 1 a 49
METRIC = "euclidean"
# ──────────────────────────────────────────────────────────────

def carregar_dados():
    for path in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n>>'{path}' não encontrado.\n"
                "Execute split.py antes deste script."
            )
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test


def separar_xy(df):
    features = [c for c in FEATURES if c in df.columns]
    return df[features], df[TARGET_COLUMN]


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
    print("Busca do Melhor K — KNN")
    print("=" * 57 + "\n")

    train, test = carregar_dados()
    X_train, y_train = separar_xy(train)
    X_test,  y_test  = separar_xy(test)

    melhor_k = buscar_melhor_k(X_train, y_train, X_test, y_test)

    print(f"\n- Use K={melhor_k} no knn_treino.py para o modelo final!")
    print(f"\n- Resultados em '{OUTPUT_DIR}/'")