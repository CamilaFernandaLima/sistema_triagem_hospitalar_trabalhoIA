# ─────────────────────────────────────────────────────────────
# Sistema de Triagem Hospitalar — KTAS (ODS 3)
# Makefile para Windows
# ─────────────────────────────────────────────────────────────

VENV       := venv
PYTHON     := $(VENV)/Scripts/python.exe
STAMP      := $(VENV)/.deps_installed

DATA_RAW   := data.csv
DATASET    := dataset_triagem_limpo.csv

.PHONY: all help install limpar_dados split treinar resultados clean

# ── Pipeline completo ────────────────────────────────────────
all: install limpar_dados split treinar
	@echo.
	@echo ==========================================
	@echo Pipeline concluido com sucesso!
	@echo Dados: data_split\
	@echo Resultados: resultados\
	@echo ==========================================

# ── Ajuda ────────────────────────────────────────────────────
help:
	@echo.
	@echo Alvos disponiveis:
	@echo   make              - Pipeline completo
	@echo   make install      - Cria venv e instala dependencias
	@echo   make limpar_dados - Limpeza e pre-processamento
	@echo   make split        - Divide treino/teste
	@echo   make treinar      - Treina arvore de decisao
	@echo   make resultados   - Abre pasta de resultados
	@echo   make clean        - Remove arquivos gerados
	@echo.

# ── Ambiente virtual e dependências ─────────────────────────
install: $(STAMP)

$(PYTHON):
	@echo Criando ambiente virtual...
	py -3.11 -m venv $(VENV)

$(STAMP): requirements.txt | $(PYTHON)
	@echo Instalando dependencias...
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@type nul > $(STAMP)

# ── Limpeza dos dados ────────────────────────────────────────
limpar_dados: $(DATASET)

$(DATASET): $(DATA_RAW) limpeza_dados.py $(STAMP)
	@echo Executando limpeza...
	$(PYTHON) limpeza_dados.py

# ── Split treino/teste ───────────────────────────────────────
split: data_split/train.csv

data_split/train.csv: $(DATASET) split.py $(STAMP)
	@echo Dividindo treino/teste...
	$(PYTHON) split.py

# ── Treinamento ──────────────────────────────────────────────
treinar: resultados/dt_matriz_confusao.png

resultados/dt_matriz_confusao.png: data_split/train.csv arv_treino.py $(STAMP)
	@echo Treinando modelo...
	$(PYTHON) arv_treino.py

# ── Abrir resultados ─────────────────────────────────────────
resultados:
	@if exist resultados (
		start "" resultados
	) else (
		echo Pasta resultados ainda nao existe.
	)

# ── Limpeza ──────────────────────────────────────────────────
clean:
	@if exist "$(DATASET)" del /Q "$(DATASET)"
	@if exist data_split rmdir /S /Q data_split
	@if exist resultados rmdir /S /Q resultados
	@if exist $(VENV) rmdir /S /Q $(VENV)
	@echo Limpeza concluida.