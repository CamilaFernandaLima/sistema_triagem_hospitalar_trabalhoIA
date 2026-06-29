# Otimização de Triagem Hospitalar Baseada no Protocolo KTAS (utilizando Machine Learning)
O projeto possui caráter de extensão universitária e está alinhado ao Objetivo de Desenvolvimento Sustentável 3 (ODS 3 - Saúde e Bem-Estar) da Organização das Nações Unidas (ONU), em especial com as metas 3.4 (redução da mortalidade prematura) e 3.8 (cobertura universal de saúde e acesso a serviços essenciais). Assim, visa aplicar conceitos de Machine Learning Supervisionado para classificar o nível de urgência de pacientes em Prontos-Socorros, mitigando o risco de erros humanos e otimizando o tempo de resposta em ambientes hospitalares sobrecarregados -- promovendo eficiência e equidade no acesso aos serviços de saúde pública.

A partir de um dataset clínico real contendo 1.268 registros de triagem baseados na escala KTAS (Korean Triage and Acuity Scale) e de três modelos de classificação, o sistema realiza a análise preditiva da gravidade do paciente utilizando variáveis como sinais vitais (pressão arterial, frequência cardíaca, saturação de O2, temperatura) e dados demográficos.

## Estrutura do Projeto

O repositório está organizado para facilitar a reprodutibilidade e a segregação entre dados, código-fonte e resultados gerados:

```text
├── data/
│   ├── data.csv                      # Dataset bruto original
│   ├── dataset_triagem_limpo.csv     # Dataset após pré-processamento
│   └── data_split/                   # Partições de treino, teste e normalização
├── resultados/
│   ├── dt/                           # Saídas da Árvore de Decisão
│   ├── rf/                           # Saídas do Random Forest
│   ├── svm/                          # Saídas do Support Vector Machine
│   ├── nb/                           # Saídas do Naive Bayes
│   ├── knn/                          # Saídas do K-Nearest Neighbors
│   └── comparacao/                   # Gráficos e comparativos
├── scripts/
│   ├── limpeza_dados.py              # Tratamento de nulos e binarização do target
│   ├── split.py                      # Divisão estratificada (80/20) e Robust Scaling
│   ├── arv_treino.py                 # Treinamento da Árvore de Decisão
│   ├── random_treino.py              # Treinamento do Random Forest
│   ├── svm_treino.py                 # Treinamento do SVM
│   ├── nb_treino.py                  # Treinamento do Naive Bayes
│   ├── knn_treino.py                 # Treinamento do KNN
│   ├── compara.py                    # Consolidação de métricas e geração de gráficos
│   └── util.py                       # Funções auxiliares (carregamento de arquivos)
├── makefile                          # Automação do pipeline (Windows)
└── README.md                         # Documentação do projeto

```

## Modelos Avaliados

O sistema treina e compara cinco arquiteturas distintas de classificação, explorando diferentes abordagens matemáticas:

1. **Árvore de Decisão (DT):** Foco em alta interpretabilidade clínica e extração de regras claras.
2. **Random Forest (RF):** Modelo *Ensemble* robusto para lidar com variância.
3. **Support Vector Machine (SVM):** Utilização de *kernel RBF* para mapeamento não-linear de sinais vitais complexos.
4. **K-Nearest Neighbors (KNN):** Classificação baseada em similaridade métrica entre pacientes.
5. **Naive Bayes (NB):** Abordagem puramente probabilística baseada no Teorema de Bayes.

## Critérios de Avaliação e Metodologia

Devido ao forte desbalanceamento natural entre casos leves e graves em hospitais, a Acurácia global não foi utilizada como métrica definidora. O *pipeline* otimiza e seleciona os modelos com base em:

* **F1-Score da Classe Minoritária (Emergência):** Busca o ponto de equilíbrio ideal entre rastrear todos os casos graves (*Recall*) sem sobrecarregar o hospital com falsos alarmes (*Precision*).
* **Calibração Dinâmica de Limiar (*Threshold*):** Os pontos de corte padrão (0.50) foram abandonados e ajustados via varredura na curva *Precision-Recall*.
* **Controle de *Gap* (Treino vs Teste):** Monitoramento rigoroso da diferença de desempenho para evitar *overfitting* (limite de tolerância em torno de 10%).

## Como Executar -> Automação com Makefile

O projeto conta com um `makefile` configurado para ambientes Windows, permitindo a execução de ponta a ponta com comandos simples.

### Comandos Principais

Abra o terminal na raiz do projeto e utilize os seguintes comandos:

**1. Instalar dependências e criar o ambiente virtual (VENV):**

```bash
make install

```

**2. Rodar o pipeline completo:**
Executa a limpeza, divisão de dados, treinamento de todos os 5 modelos e gera o relatório comparativo final.

```bash
make all

```

**3. Executar etapas individuais (Opcional):**
Caso queira rodar os processos separadamente:

* `make limpeza_dados`: Gera o dataset tratado.
* `make split`: Realiza o encoding, particionamento e escalonamento (*Robust Scaler*).
* `make modelos`: Treina todos os algoritmos.
* `make rf` (ou `dt`, `svm`, `nb`, `knn`): Treina um modelo específico.
* `make comparar`: Gera os gráficos de consolidação na pasta `resultados/comparacao`.

**4. Limpar o ambiente:**
Apaga os arquivos gerados (dados processados, *splits* e resultados) para forçar uma nova execução do zero.

```bash
make clean

```

### Integrantes
* Camila Fernanda e Silva Lima
* Maria Fernanda Siqueira de Moraes 

