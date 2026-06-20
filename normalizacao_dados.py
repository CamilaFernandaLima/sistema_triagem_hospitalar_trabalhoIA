import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('dataset_triagem_limpo.csv')

# 1. Separa as colunas por tipo operacional
# Categóricas e Target ficam de fora da normalização matemática
var_categoricas_e_target = ['Sex', 'Injury', 'Mental', 'Pain', 'KTAS_target_binario']
# Variáveis contínuas que precisam entrar na mesma escala geométrica
var_continuas = ['Age', 'NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']

# 2. Cria uma cópia do dataset para aplicar a normalização
df_normalizado = df.copy()

# 3. Inicializa e aplica o StandardScaler apenas nas colunas contínuas
scaler = StandardScaler()
df_normalizado[var_continuas] = scaler.fit_transform(df[var_continuas])

# 4. Salva o novo arquivo
df_normalizado.to_csv('dataset_triagem_normalizado.csv', index=False, encoding='utf-8')
