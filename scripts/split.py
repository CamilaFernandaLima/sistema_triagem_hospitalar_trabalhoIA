"""
Divisão Treino/Teste com Estratificação, One-Hot Encoding e Robust Scaling
================================================================
Target : KTAS_target_binario  ->  1 = Emergência | 0 = Não-Emergência
Divisão: 80% treino / 20% teste com estratificação, dummies e escala robusta
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DE CAMINHOS E ATRIBUTOS
# ──────────────────────────────────────────────────────────────
DATASET_PATH  = os.path.join("data", "dataset_triagem_limpo.csv")
TARGET_COLUMN = "KTAS_target_binario"
RANDOM_STATE  = 42
TEST_SIZE     = 0.20
OUTPUT_DIR    = os.path.join("data", "data_split")

# Colunas categóricas que passarão por One-Hot Encoding
# Descartamos o KTAS_expert (pois gerou o target) e mantemos as nominais
CATEGORICAL_COLUMNS = ["Sex", "Injury", "Mental", "Pain"]

# Variáveis contínuas que necessitam do Robust Scaler customizado
VAR_CONTINUAS = ["Age", "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation"]
# ──────────────────────────────────────────────────────────────


def carregar_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"\n>> Arquivo '{DATASET_PATH}' não encontrado.\n"
            "Certifique-se de que a estrutura de pastas 'data/' existe."
        )
    df = pd.read_csv(DATASET_PATH)
    print(f">> Dataset original carregado: {df.shape[0]} pacientes x {df.shape[1]} colunas")
    return df


def aplicar_one_hot_encoder(dataframe, categorical_columns, nan_as_category=False):
    """Transforma variáveis categóricas em binárias (0 ou 1) removendo multicolinearidade."""
    original_columns = list(dataframe.columns)
    
    # drop_first=True evita redundância estatística (ex: Sex_1 vira a referência e Sex_2 vira a dummy)
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns,
                               dummy_na=nan_as_category, drop_first=True)
    
    new_columns = [col for col in dataframe.columns if col not in original_columns]
    return dataframe, new_columns


def aplicar_robust_scaler_treino_teste(X_train, X_test, colunas_continas):
    """
    Aplica o Robust Scaler baseado em percentis (5% e 95%).
    Calcula os parâmetros no treino e aplica no teste para evitar Data Leakage.
    """
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    for col in colunas_continas:
        if col in X_train.columns:
            # Coleta as estatísticas estritamente do conjunto de treino
            var_median = X_train[col].median()
            quartile1 = X_train[col].quantile(0.05)
            quartile3 = X_train[col].quantile(0.95)
            interquantile_range = quartile3 - quartile1
            
            # Proteção contra divisão por zero se a base for muito homogênea
            if int(interquantile_range) == 0:
                quartile1 = X_train[col].quantile(0.05)
                quartile3 = X_train[col].quantile(0.95)
                interquantile_range = quartile3 - quartile1
            
            # Aplica no Treino
            z_train = (X_train[col] - var_median) / interquantile_range
            X_train_scaled[col] = round(z_train, 3)
            
            # Aplica no Teste usando a MEDIANA e o IQR do TREINO (Rigor Científico!)
            z_test = (X_test[col] - var_median) / interquantile_range
            X_test_scaled[col] = round(z_test, 3)
            
    return X_train_scaled, X_test_scaled


def dividir_e_salvar(df):
    # 1. Separar as colunas úteis (reunindo contínuas e categóricas)
    todas_features = CATEGORICAL_COLUMNS + VAR_CONTINUAS
    X_bruto = df[todas_features]
    y = df[TARGET_COLUMN]

    # 2. Divisão Estratificada Holdout (80% Treino / 20% Teste) antes das transformações complexas
    X_train_bruto, X_test_bruto, y_train, y_test = train_test_split(
        X_bruto, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f">> Divisão de linhas realizada:")
    print(f">>>> Treino : {len(X_train_bruto)} amostras (80%)")
    print(f">>>> Teste  : {len(X_test_bruto)}  amostras (20%)\n")

    # 3. Geração da VERSÃO ORIGINAL (Para a Árvore de Decisão)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    train_df = X_train_bruto.copy(); train_df[TARGET_COLUMN] = y_train
    test_df  = X_test_bruto.copy();  test_df[TARGET_COLUMN]  = y_test
    
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test.csv")
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path,   index=False, encoding="utf-8")

    # 4. Geração da VERSÃO PROCESSADA (One-Hot Encoding + Robust Scaler para KNN/Naive Bayes)
    # 4a. Aplicar One-Hot Encoder no dataset completo de features de forma consistente
    X_encoded, novas_colunas = aplicar_one_hot_encoder(X_bruto, CATEGORICAL_COLUMNS)
    
    # Re-separar as linhas do encoded usando os mesmos índices originais do split
    X_train_encoded = X_encoded.loc[X_train_bruto.index]
    X_test_encoded = X_encoded.loc[X_test_bruto.index]

    # 4b. Aplicar Robust Scaler nas variáveis contínuas (protegido contra vazamento)
    X_train_norm, X_test_norm = aplicar_robust_scaler_treino_teste(
        X_train_encoded, X_test_encoded, VAR_CONTINUAS
    )

    # Recompor DataFrames com os Targets
    train_norm_df = X_train_norm.copy(); train_norm_df[TARGET_COLUMN] = y_train
    test_norm_df  = X_test_norm.copy();  test_norm_df[TARGET_COLUMN]  = y_test

    train_norm_path = os.path.join(OUTPUT_DIR, "train_normalized.csv")
    test_norm_path  = os.path.join(OUTPUT_DIR, "test_normalized.csv")
    train_norm_df.to_csv(train_norm_path, index=False, encoding="utf-8")
    test_norm_df.to_csv(test_norm_path,   index=False, encoding="utf-8")

    print(f">> Engenharia de Recursos concluída:")
    print(f">>>> Novas colunas dummy criadas: {novas_colunas}")
    print(f">> Arquivos salvos com sucesso em '{OUTPUT_DIR}':")
    print(f">>>> [Árvore]  {train_path} e {test_path}")
    print(f">>>> [Robust]  {train_norm_path} e {test_norm_path}")


if __name__ == "__main__":
    dataset = carregar_dataset()
    dividir_e_salvar(dataset)

    print("\n>> Fluxo split.py concluído com sucesso e pronto para os modelos!")  