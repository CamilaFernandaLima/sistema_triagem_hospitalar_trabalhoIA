import pandas as pd
import numpy as np

# 1. Carregar o arquivo tratando os separadores e codificação
df = pd.read_csv('data.csv', na_values=['??', '?', 'null', ' '], encoding='latin1', sep=None, engine='python')

print("ANÁLISE INICIAL DOS DADOS BRUTOS:")
print(f"Total de registros mapeados: {df.shape[0]} pacientes.\n")

colunas_clinicas = [
    'Sex', 'Age', 'Injury', 'Mental', 'Pain', 'NRS_pain', 
    'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation', 'KTAS_expert'
]

print("Quantidade de valores vazios antes da limpeza técnica:")
print(df[colunas_clinicas].isna().sum())
print("-" * 50)


# TRATAMENTO DE LIMPEZA E TIPAGEM (IMPUTAÇÃO)

# Correção: Forçar todas as colunas clínicas a serem numéricas.
# O parâmetro errors='coerce' transforma sujeiras como '#BOÞ!' em vazios (NaN) 
for col in colunas_clinicas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Tratamento para Variáveis Categóricas/Codificadas (Substituição pela Moda)
var_categoricas = ['Sex', 'Injury', 'Mental', 'Pain']
for col in var_categoricas:
    moda_da_coluna = df[col].mode()[0]
    df[col] = df[col].fillna(moda_da_coluna)

# 3. Tratamento para Variáveis Numéricas/Sinais Vitais (Substituição pela Mediana)
var_numericas = ['Age', 'NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
for col in var_numericas:
    mediana_da_coluna = df[col].median()
    df[col] = df[col].fillna(mediana_da_coluna)

# 4. Tratamento do Target (KTAS_expert)
df = df.dropna(subset=['KTAS_expert'])


# VALIDAÇÃO DA LIMPEZA
print("\nANÁLISE APÓS O PRÉ-PROCESSAMENTO:")
print(f"Total de registros limpos e válidos: {df.shape[0]} pacientes.")
print("Quantidade de valores vazios por coluna após a limpeza:")
print(df[colunas_clinicas].isna().sum())

# Salvar uma cópia limpa e impecável
df.to_csv('dataset_triagem_limpo.csv', index=False, encoding='utf-8')
print("\nArquivo 'dataset_triagem_limpo.csv' gerado com sucesso!")

