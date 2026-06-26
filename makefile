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
TRAIN_NORM := data/data_split/train_normalized.csv
TEST_NORM  := data/data_split/test_normalized.csv

DT_RESULT  := resultados/dt
RF_RESULT  := resultados/rf
SVM_RESULT := resultados/svm
NB_RESULT  := resultados/nb
KNN_RESULT := resultados/knn
COMP_RESULT:= resultados/comparacao

.PHONY: all help install limpar_dados split modelos decision_tree random_forest svm naive_bayes knn comparar clean force

# ── Pipeline Completo ────────────────────────────────────────
all: install limpar_dados split modelos comparar
	@echo.
	@echo ==============================================
	@echo Pipeline completo concluido com sucesso!
	@echo Todos os modelos foram treinados e comparados.
	@echo ==============================================

# ── Treinamento de todos os modelos ──────────────────────────
modelos: decision_tree random_forest svm naive_bayes knn
	@echo.
	@echo ==============================================
	@echo Todos os 5 modelos foram treinados com sucesso!
	@echo ==============================================

# ── Ajuda / Comandos disponíveis ─────────────────────────────
help:
	@echo.
	@echo Alvos disponiveis:
	@echo   make install       - Instala dependencias no venv
	@echo   make limpar_dados  - Executa o script de limpeza de dados
	@echo   make split         - Divide os dados em treino e teste
	@echo   make decision_tree - Treina o modelo de Arvore de Decisao
	@echo   make random_forest - Treina o modelo de Random Forest
	@echo   make svm           - Treina o modelo SVM
	@echo   make naive_bayes   - Treina o modelo Naive Bayes
	@echo   make knn           - Treina o modelo KNN
	@echo   make modelos       - Treina todos os 5 modelos sequencialmente
	@echo   make comparar      - Gera tabelas e graficos comparativos
	@echo   make all           - Executa o pipeline completo (tudo acima)
	@echo   make clean         - Remove os resultados e dados gerados
	@echo   make force         - Forca o rebuild completo do projeto

# ── Preparação do Ambiente e Dados ───────────────────────────
install: $(STAMP)

$(STAMP): requirements.txt
	@echo [INFO] Criando ambiente virtual e instalando dependencias...
	@if not exist $(VENV) python -m venv $(VENV)
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo venv_ready > $(STAMP)
	@echo [OK] Ambiente preparado.

limpar_dados: $(DATASET)

$(DATASET): $(DATA_RAW) scripts/limpeza_dados.py $(STAMP)
	@echo [INFO] Executando limpeza e pre-processamento dos dados...
	@$(PYTHON) scripts/limpeza_dados.py
	@echo [OK] Dados limpos prontos.

split: $(TRAIN_FILE) $(TEST_FILE)

$(TRAIN_FILE) $(TEST_FILE): $(DATASET) scripts/split.py $(STAMP)
	@echo [INFO] Dividindo treino/teste e gerando normalizacoes...
	@$(PYTHON) scripts/split.py
	@echo [OK] Arquivos de split gerados (originais e normalizados).

# ── Modelos Individuais ──────────────────────────────────────
decision_tree: $(DT_RESULT)

$(DT_RESULT): $(TRAIN_FILE) $(TEST_FILE) scripts/arv_treino.py $(STAMP)
	@echo [INFO] Treinando Decision Tree...
	@$(PYTHON) scripts/arv_treino.py
	@echo [OK] Resultados de Decision Tree salvos.

random_forest: $(RF_RESULT)

$(RF_RESULT): $(TRAIN_FILE) $(TEST_FILE) scripts/random_treino.py $(STAMP)
	@echo [INFO] Treinando Random Forest...
	@$(PYTHON) scripts/random_treino.py
	@echo [OK] Resultados de Random Forest salvos.

svm: $(SVM_RESULT)

$(SVM_RESULT): $(TRAIN_FILE) $(TEST_FILE) scripts/svm_treino.py $(STAMP)
	@echo [INFO] Treinando SVM...
	@$(PYTHON) scripts/svm_treino.py
	@echo [OK] Resultados de SVM salvos.

naive_bayes: $(NB_RESULT)

$(NB_RESULT): $(TRAIN_NORM) $(TEST_NORM) scripts/nb_treino.py $(STAMP)
	@echo [INFO] Treinando Naive Bayes...
	@$(PYTHON) scripts/nb_treino.py
	@echo [OK] Resultados de Naive Bayes salvos.

knn: $(KNN_RESULT)

$(KNN_RESULT): $(TRAIN_NORM) $(TEST_NORM) scripts/knn_treino.py $(STAMP)
	@echo [INFO] Treinando KNN...
	@$(PYTHON) scripts/knn_treino.pys
	@echo [OK] Resultados de KNN salvos.

# ── Consolidação e Comparação ────────────────────────────────
comparar: $(COMP_RESULT)

$(COMP_RESULT): $(DT_RESULT) $(RF_RESULT) $(SVM_RESULT) $(NB_RESULT) $(KNN_RESULT) scripts/compara.py $(STAMP)
	@echo [INFO] Executando script de comparacao de modelos...
	@$(PYTHON) scripts/compara.py
	@echo [OK] Graficos e tabelas comparativas gerados em '$(COMP_RESULT)'.

# ── Limpeza ──────────────────────────────────────────────────
clean:
	@echo [INFO] Limpando arquivos gerados...
	@if exist resultados rmdir /S /Q resultados
	@if exist data\\data_split rmdir /S /Q data\\data_split
	@if exist $(DATASET) del /Q $(DATASET)
	@echo [OK] Limpeza concluida.

force: clean
	@echo [INFO] Forcando reexecucao total...
	@if exist $(STAMP) del /Q $(STAMP)
	@$(MAKE) all