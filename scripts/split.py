"""
Divisão Treino/Teste com Estratificação e Normalização
================================================================
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência
Divisão: 80% treino / 20% teste com estratificação e Z-score
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # <-- Importação adicionada
import os

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
DATASET_PATH  = "data\dataset_triagem_limpo.csv"   # arquivo gerado por limpeza_dados.py
TARGET_COLUMN = "KTAS_target_binario"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
OUTPUT_DIR    = "data\data_split" 

# Colunas clínicas usadas como features (descarta texto e leakage)
FEATURES = [
    "Sex", "Age", "Injury", "Mental", "Pain", "NRS_pain",
    "SBP", "DBP", "HR", "RR", "BT", "Saturation"
]

# Variáveis contínuas que necessitam de Z-score para o KNN
VAR_CONTINUAS = ["Age", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation"]
# ──────────────────────────────────────────────────────────────


def carregar_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\n>>Arquivo '{DATASET_PATH}' não encontrado.\n"
            "Certifique-se de que executou 'limpeza_dados.py'\n"
            "e que o arquivo gerado está na mesma pasta deste script."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f">>Dataset carregado: {df.shape[0]} pacientes x {df.shape[1]} colunas")
    return df


def preparar_xy(df):
    features_existentes = [c for c in FEATURES if c in df.columns]
    ausentes = [c for c in FEATURES if c not in df.columns]
    if ausentes:
        print(f">> Colunas ausentes (serão ignoradas): {ausentes}")

    X = df[features_existentes]
    y = df[TARGET_COLUMN]

    print(f"\n>>Features usadas ({len(features_existentes)}): {features_existentes}")
    print(f">>Target: '{TARGET_COLUMN}'")
    print(f">>>>Distribuição: {dict(y.value_counts().rename({1: 'Emergência (1)', 0: 'Não-Emergência (0)'}))}\n")
    return X, y


def dividir_e_salvar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y          #  mantém proporção de classes em treino e teste
    )

    print(f">>Divisão realizada:")
    print(f">>>>Treino : {len(X_train)} amostras (80%)")
    print(f">>>>Teste  : {len(X_test)}  amostras (20%)\n")

    # Verificação de estratificação
    print(">>Proporção das classes por conjunto:")
    for nome, y_part in [("Treino", y_train), ("Teste", y_test)]:
        prop = y_part.value_counts(normalize=True).sort_index() * 100
        print(f"   {nome}: Não-Emergência={prop.get(0, 0):.1f}%  Emergência={prop.get(1, 0):.1f}%")
    print()

    # Cria o diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # VERSÃO ORIGINAL 
    # ----------------------------------------------------------
    train_df = X_train.copy(); train_df[TARGET_COLUMN] = y_train
    test_df  = X_test.copy();  test_df[TARGET_COLUMN]  = y_test

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path,   index=False, encoding="utf-8")

    # ----------------------------------------------------------
    # VERSÃO NORMALIZADA COM Z-SCORE 
    # ----------------------------------------------------------
    scaler = StandardScaler()
    
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()

    # O scaler ajusta os parâmetros no treino e apenas transforma o teste (evita Data Leakage)
    X_train_norm[VAR_CONTINUAS] = scaler.fit_transform(X_train[VAR_CONTINUAS])
    X_test_norm[VAR_CONTINUAS] = scaler.transform(X_test[VAR_CONTINUAS])

    train_norm_df = X_train_norm.copy(); train_norm_df[TARGET_COLUMN] = y_train
    test_norm_df  = X_test_norm.copy();  test_norm_df[TARGET_COLUMN]  = y_test

    train_norm_path = os.path.join(OUTPUT_DIR, "train_normalized.csv")
    test_norm_path  = os.path.join(OUTPUT_DIR, "test_normalized.csv")

    train_norm_df.to_csv(train_norm_path, index=False, encoding="utf-8")
    test_norm_df.to_csv(test_norm_path,   index=False, encoding="utf-8")

    print(f">>Arquivos salvos em '{OUTPUT_DIR}/':")
    print(f">>>>[Original] {train_path} e {test_path}")
    print(f">>>>[Normalizado] {train_norm_path} e {test_norm_path}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("=" * 57)
    print("Divisão Treino/Teste com Estratificação e Escalonamento")
    print("=" * 57 + "\n")

    df = carregar_dataset()
    X, y = preparar_xy(df)
    dividir_e_salvar(X, y)

    print("\n>> Fluxo split.py concluído com sucesso.")