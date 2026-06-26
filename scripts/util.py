import pandas as pd
import os

# Constantes globais
FEATURES = [
    "Sex", "Age", "Injury", "Mental", "Pain", "NRS_pain",
    "SBP", "DBP", "HR", "RR", "BT", "Saturation"
]
TARGET_COLUMN = "KTAS_target_binario"

def carregar_dados(train_path, test_path):
    """Verifica se os arquivos existem e os carrega."""
    for path in [train_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\n>> '{path}' não encontrado.\n"
                "Execute o script de separação (split.py) antes deste script."
            )
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f">> Treino: {train.shape}   Teste: {test.shape}")
    return train, test

def separar_xy(df):
    """Separa as features (X) da variável alvo (y)."""
    features_presentes = [c for c in FEATURES if c in df.columns]
    return df[features_presentes], df[TARGET_COLUMN]