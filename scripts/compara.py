"""
Script de Comparação de Modelos
=======================================================================
Lê os arquivos metricas_padronizadas.json de cada modelo, consolida
os resultados e gera gráficos comparativos.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Usa o modo Agg para não dar erro em ambientes sem interface gráfica
matplotlib.use("Agg")

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────
PASTAS_MODELOS = ["nb", "knn", "dt", "rf", "svm"]
BASE_DIR = "resultados"
OUTPUT_DIR = "resultados/comparacao"

def coletar_dados():
    """Lê todos os JSONs padronizados e retorna um DataFrame."""
    dados = []
    
    for pasta in PASTAS_MODELOS:
        caminho_json = os.path.join(BASE_DIR, pasta, f"{pasta}_resumo.json")
        
        if os.path.exists(caminho_json):
            with open(caminho_json, "r", encoding="utf-8") as f:
                resumo = json.load(f)
                
                # Achata os parâmetros para ficarem na mesma tabela (opcional)
                params = resumo.pop("Parametros", {})
                resumo["Config_Principal"] = str(params)
                
                dados.append(resumo)
        else:
            print(f"[Aviso] Arquivo não encontrado: {caminho_json}")
            
    return pd.DataFrame(dados)

def plotar_comparacao_barras(df, metrica, titulo, arquivo_saida, cor):
    """Função auxiliar para plotar gráficos de barras (suporta dados negativos)."""
    df_ordenado = df.sort_values(by=metrica, ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df_ordenado["Modelo"], df_ordenado[metrica], color=cor, edgecolor="black")
    
    # 1. Adiciona uma linha de base clara no zero
    ax.axvline(0, color='black', linewidth=1)
    
    # 2. Adiciona os valores na ponta de cada barra, respeitando o sinal
    max_val = max(df[metrica])
    min_val = min(df[metrica])
    
    # Define uma margem de segurança para o texto não colar na barra
    margem = (max_val - min_val) * 0.02 if max_val != min_val else 0.01

    for bar in bars:
        valor = bar.get_width()
        
        # Ajusta posição e alinhamento baseado se o valor é positivo ou negativo
        if valor >= 0:
            pos_x = valor + margem
            alinhamento = 'left'
        else:
            pos_x = valor - margem
            alinhamento = 'right'
            
        ax.text(
            pos_x, 
            bar.get_y() + bar.get_height() / 2, 
            f'{valor:.4f}', 
            va='center', ha=alinhamento, fontsize=10
        )

    ax.set_title(titulo, fontsize=14, pad=15)
    ax.set_xlabel(metrica, fontsize=12)
    
    # 3. Ajusta o limite do eixo X dinamicamente para comportar positivos e negativos
    limite_inferior = min_val * 1.2 if min_val < 0 else 0
    limite_superior = max_val * 1.2 if max_val > 0 else 0
    
    # Caso especial onde todos os valores sejam zero (evita erro no gráfico)
    if limite_inferior == 0 and limite_superior == 0:
        limite_superior = 1
        
    ax.set_xlim(limite_inferior, limite_superior)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(arquivo_saida, dpi=150)
    plt.close()
    print(f">> Gráfico salvo: {arquivo_saida}")

if __name__ == "__main__":
    print("=" * 57)
    print("CONSOLIDAÇÃO E COMPARAÇÃO DOS MODELOS")
    print("=" * 57 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_resultados = coletar_dados()
    
    if df_resultados.empty:
        print("Nenhum dado encontrado. Execute os scripts de treino primeiro.")
        exit()

    # 1. Salvar tabela consolidada
    caminho_csv = os.path.join(OUTPUT_DIR, "tabela_comparativa.csv")
    df_resultados.to_csv(caminho_csv, index=False)
    print(f">> Tabela consolidada salva em: {caminho_csv}\n")
    
    # Mostrar no terminal para conferência rápida
    colunas_exibicao = ["Modelo", "F1_Emergencia", "AUC_ROC", "Acuracia_Teste", "Gap"]
    print(df_resultados[colunas_exibicao].to_string(index=False))
    print("\n" + "=" * 57 + "\n")

    # 2. Gerar Gráficos Comparativos
    # Gráfico 1: F1-Score (Métrica mais importante para o seu problema)
    plotar_comparacao_barras(
        df_resultados, 
        metrica="F1_Emergencia", 
        titulo="Comparação: F1-Score (Classe Emergência)", 
        arquivo_saida=os.path.join(OUTPUT_DIR, "comp_f1_score.png"),
        cor="#1976D2"
    )

    # Gráfico 2: AUC-ROC
    plotar_comparacao_barras(
        df_resultados, 
        metrica="AUC_ROC", 
        titulo="Comparação: Área Sob a Curva ROC (AUC)", 
        arquivo_saida=os.path.join(OUTPUT_DIR, "comp_auc_roc.png"),
        cor="#7B1FA2"
    )

    # Gráfico 3: Gap (Overfitting) - *Aqui, valores MENORES são melhores*
    plotar_comparacao_barras(
        df_resultados, 
        metrica="Gap", 
        titulo="Comparação: Gap (Overfitting: Acc Treino - Acc Teste)", 
        arquivo_saida=os.path.join(OUTPUT_DIR, "comp_gap.png"),
        cor="#E53935"
    )

    print("\nComparação concluída com sucesso!")