# ─────────────────────────────────────────────────────────────
# Sistema de Triagem Hospitalar — KTAS (ODS 3)
# Makefile para Windows 
# ─────────────────────────────────────────────────────────────

VENV       := venv
PYTHON     := $(VENV)/Scripts/python.exe
PIP        := $(PYTHON) -m pip
STAMP      := $(VENV)/.deps_installed

DATA_RAW   := data/data.csv
DATASET    := data/dataset_triagem_limpo.csv

TRAIN_FILE := data/data_split/train.csv
TEST_FILE  := data/data_split/test.csv

DT_RESULT  := resultados/dt
RF_RESULT  := resultados/rf
SVM_RESULT := resultados/svm

.PHONY: all help install limpar_dados split decision_tree random_forest svm clean force

# ── Pipeline completo ────────────────────────────────────────
all: install limpar_dados split modelos
	@echo.
	@echo ==============================================
	@echo Pipeline concluido com sucesso!
	@echo ==============================================

# ── Treinamento de modelos ───────────────────────────────────
modelos: decision_tree random_forest svm
	@echo.
	@echo ==============================================
	@echo Modelos treinados com sucesso!
	@echo ==============================================

# ── Ajuda ────────────────────────────────────────────────────
help:
	@echo.
	@echo Alvos disponiveis:
	@echo   make
	@echo   make install
	@echo   make limpar_dados
	@echo   make split
	@echo   make decision_tree
	@echo   make random_forest
	@echo   make resultados
	@echo   make clean
	@echo.

# ── Ambiente virtual ─────────────────────────────────────────
install: $(STAMP)

$(STAMP): scripts/requirements.txt
	@echo [INFO] Criando ambiente virtual...
	@if not exist "$(VENV)" python -m venv $(VENV)

	@echo [INFO] Atualizando pip...
	@$(PIP) install --upgrade pip

	@echo [INFO] Instalando dependencias...
	@$(PIP) install -r scripts/requirements.txt

	@echo ok > $(STAMP)
	@echo [OK] Ambiente pronto.

# ── Limpeza de dados ─────────────────────────────────────────
limpar_dados: $(DATASET)

$(DATASET): $(DATA_RAW) scripts/limpeza_dados.py $(STAMP)
	@echo [INFO] Executando limpeza dos dados...
	@$(PYTHON) scripts/limpeza_dados.py
	@echo [OK] Dataset gerado.

# ── Train/Test Split ─────────────────────────────────────────
split: $(TRAIN_FILE)

$(TRAIN_FILE) $(TEST_FILE): $(DATASET) scripts/split.py $(STAMP)
	@echo [INFO] Dividindo treino/teste...
	@$(PYTHON) scripts/split.py
	@echo [OK] Arquivos de split gerados.

# ── Decision Tree ────────────────────────────────────────────
decision_tree: $(DT_RESULT)

$(DT_RESULT): $(TRAIN_FILE) $(TEST_FILE) scripts/arv_treino.py $(STAMP)
	@echo [INFO] Treinando Decision Tree...
	@$(PYTHON) scripts/arv_treino.py
	@echo [OK] Resultados salvos.

# ── Random Forest ────────────────────────────────────────────
random_forest: $(RF_RESULT)

$(RF_RESULT): $(TRAIN_FILE) $(TEST_FILE) scripts/random_treino.py $(STAMP)
	@echo [INFO] Treinando Random Forest...
	@$(PYTHON) scripts/random_treino.py
	@echo [OK] Resultados salvos.

# -- SVM ─────────────────────────────────────────────────────────
svm: $(SVM_RESULT)
$(SVM_RESULT): $(TRAIN_FILE) $(TEST_FILE) scripts/svm_treino.py $(STAMP)
	@echo [INFO] Treinando SVM...
	@$(PYTHON) scripts/svm_treino.py
	@echo [OK] Resultados salvos.

# ── Forçar rebuild ───────────────────────────────────────────
force:
	@echo Forcando reexecucao...
	@del /Q $(DATASET) 2>nul
	@del /Q $(TRAIN_FILE) 2>nul
	@del /Q $(TEST_FILE) 2>nul
	@del /Q $(DT_RESULT) 2>nul
	@del /Q $(RF_RESULT) 2>nul

# ── Limpeza ──────────────────────────────────────────────────
clean:
	@echo Removendo arquivos gerados...

	@if exist $(DATASET) del /Q $(DATASET)
	@if exist $(TRAIN_FILE) del /Q $(TRAIN_FILE)
	@if exist $(TEST_FILE) del /Q $(TEST_FILE)
	@if exist $(DT_RESULT) rmdir /S /Q $(DT_RESULT)
	@if exist $(RF_RESULT) rmdir /S /Q $(RF_RESULT)
	@if exist $(SVM_RESULT) rmdir /S /Q $(SVM_RESULT)

	@echo [OK] Limpeza concluida.