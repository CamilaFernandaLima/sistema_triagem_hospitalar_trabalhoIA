import pandas as pd
import numpy as np

# Carregar o arquivo tratando os separadores e codificação
df = pd.read_csv('data.csv', na_values=['??', '?', 'null', ' '], encoding='latin1', sep=None, engine='python')

print("ANÁLISE INICIAL DOS DADOS BRUTOS:")
print(f"Total de registros mapeados: {df.shape[0]} pacientes.\n")

colunas_clinicas = [
    'Sex', 'Age', 'Injury', 'Mental', 'Pain', 'NRS_pain', 
    'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation', 'KTAS_expert'
]

# Forçar todas as colunas de interesse a serem numéricas
# O parâmetro errors='coerce' transforma sujeiras como '#BOÞ!' em vazios (NaN) 
for col in colunas_clinicas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Quantidade de valores vazios antes da limpeza técnica:")
print(df[colunas_clinicas].isna().sum())
print("-" * 50)


# TRATAMENTO DE LIMPEZA E TIPAGEM (IMPUTAÇÃO)

# 1. Tratamento para Variáveis Categóricas/Codificadas (Substituição pela Moda)
var_categoricas = ['Sex', 'Injury', 'Mental', 'Pain']
for col in var_categoricas:
    moda_da_coluna = df[col].mode()[0]
    df[col] = df[col].fillna(moda_da_coluna)

# 2. Tratamento para Variáveis Numéricas/Sinais Vitais (Substituição pela Mediana)
var_numericas = ['Age', 'NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
for col in var_numericas:
    mediana_da_coluna = df[col].median()
    df[col] = df[col].fillna(mediana_da_coluna)

# 3. Tratamento e Binarização do Target (KTAS_expert)
# Removemos pacientes que não possuem a classificação do especialista (target vazio)
df = df.dropna(subset=['KTAS_expert'])

# Mapeamento binário conforme o documento técnico:
# níveis 1, 2 e 3 = 1 (Emergência)
# níveis 4 e 5 = 0 (Não-Emergência)
df['KTAS_target_binario'] = df['KTAS_expert'].apply(lambda x: 1 if x in [1, 2, 3] else 0)


# VALIDAÇÃO DA LIMPEZA
print("\nANÁLISE APÓS O PRÉ-PROCESSAMENTO:")
print(f"Total de registros limpos e válidos: {df.shape[0]} pacientes.")
print(f"Distribuição das classes do Target Binário:")
print(df['KTAS_target_binario'].value_counts().rename({1: '1 (Emergência)', 0: '0 (Não-Emergência)'}))
print("Quantidade de valores vazios por coluna após a limpeza:")
print(df[colunas_clinicas].isna().sum())

# salvar uma cópia limpa
df.to_csv('dataset_triagem_limpo.csv', index=False, encoding='utf-8')
print("\nArquivo 'dataset_triagem_limpo.csv' gerado com sucesso!")

