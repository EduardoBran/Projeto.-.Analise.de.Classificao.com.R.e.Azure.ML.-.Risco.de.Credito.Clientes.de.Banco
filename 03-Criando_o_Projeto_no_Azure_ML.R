####  Análise de Classificação com Linguagem R e Azure ML  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco")
getwd()


####  Criando o Experimento no Azure ML  ####


## Criar Experimento

# - Criar um novo experimento chamado "Classificação - Risco de Crédito"


## Carregando dataset

# - Procurar e arrastar o módulo "German Credit Card UCI dataset"
# - Procurar e arrastar o módulo "ClassTools.zip"


## Convertendo as variáveis do tipo string para factor

# - Procurar e arrastar o módulo "Edit Metadata"
# - Conectar módulo "German Credit Card UCI dataset" em "Edit Metadata"

# - Configurar o módulo "Edit Metadata" selecionando as colunas e na opção Categorical mudar para "Make Categorical"


## Alterando os nomes das variáveis

# - Procurar e arrastar outro módulo "Edit Metadata"
# - Conectar primeiro módulo do "Edit Metadata" neste

# - Configurar o segundo módulo "Edit Metadata" selecionando todas as colunas (iremos mudar o nome de todas)
# - Configurar o atributo "New column names" com os seguintes nomes:
#   CheckingAcctStat, Duration, CreditHistory, Purpose, CreditAmount, SavingsBonds, Employment, InstallmentRatePecnt, SexAndStatus, OtherDetorsGuarantors, PresentResidenceTime, Property, Age, OtherInstallments, Housing, ExistingCreditsAtBank, Job, NumberDependents, Telephone, ForeignWorker, CreditStatus


## Análise Exploratória (Engenharia de Atributos em Variáveis Numéricas)

# - Ao analisar o dataset após as transformações acima, constatar que as variáveis "Duration", "CreditAmount" e "Age"
#   possuem muitos valores únicos.

# - E variáveis com uma grande quantidade de valores únicos podem apresentar problemas durante o trienamento de um Modelo de Classificação.
# - O que fazer neste caso?
#   Iremos converter essas três variáveis de Numéricas para Qualitativas (categóricas/factor)

# - Iremos então criar e colocar os dados destas variáveis em 4 ou 5 categorias diferentes
#   Exemplo: pegaremos a coluna "CreditAmount" que possui 921 valores únicos e colocaremos estes valores em categorias

# - Isto não é obrigatório. O fato de aplicar este tipo de engenharia de atributos neste dataset não significa que devemos aplicar me outro


## Aplicando Engenharia de Atributos em Variáveis Numéricas

# - Uma forma de fazer isso seria procurar e arrastar o módulo "Group Data into Bins"

# - Mas iremos utilizar o módulo "Execute R Script". Procurar e arrastar os módulos "Execute R Script" e "ClassTools.zip"
# - Conectar o último módulo "Edit Metadata" e o módulo "ClassTools.zip" no módulo "Execute R Scrpt"

# - Colar no módulo "Execute R Script":

## Carregando Pacotes
library(dplyr)
# Variável que controla a execução do script
Azure <- TRUE

if(Azure){
  source("src/ClassTools.R")
  Credit <- maml.mapInputPort(1)
}else{
  # R
}

# Forma 1 (Intervalar)
# Criando 3 novas variáveis que foram transformadas de variáveis numéricas para variáveis categóricas com as funções de ClassTools.R 
toFactors <- c("Duration", "CreditAmount", "Age")
maxVals <- c(100, 1000000, 100)
facNames <- unlist(lapply(toFactors, function(x) paste(x, "_f", sep = "")))
Credit[, facNames] <- Map(function(x, y) quantize.num(Credit[, x], maxval = y), toFactors, maxVals)

# Forma 2 (Discreta)
# Criando 3 novas variáveis que foram transformadas de variáveis numéricas para variáveis categóricas aqui
criar_categorias <- function(variavel, num_categorias) {
  # Criar breakpoints para dividir a variável em categorias
  breakpoints <- quantile(variavel, probs = seq(0, 1, length.out = num_categorias + 1))
  
  # Criar categorias
  categorias <- cut(variavel, breaks = breakpoints, labels = seq(1, num_categorias), include.lowest = TRUE)
  
  return(categorias)
}
num_categorias <- 5
Credit$Duration_Categoria <- criar_categorias(Credit$Duration, num_categorias)
Credit$CreditAmount_Categoria <- criar_categorias(Credit$CreditAmount, num_categorias)
Credit$Age_Categoria <- criar_categorias(Credit$Age, num_categorias)

# Output 
if(Azure) maml.mapOutputPort('Credit')


## Dividir Dados em Treino e Teste

# - Procurar e arrastar o módulo "Split Data"
# - Conectar o último módulo "Execute R Script" em "Split Data"
# - Configurar o módulo "Split Data" com Fraction de 0.7 e Random seed 7849


## Aplica Técnica de Balanceamento dos Dados

# - Quando olhamos para a nossa variável alvo CreditStatus, podemos constatar que existem muito mais valores 1 (yes) do que 2 (no),
#   ou seja, nosso dataset possui muito mais BONS pagadores do que MAUS pagadores

# - E este desbalanceamento da variável alvo é um problema, pois se apresentarmos os dados desta forma ao Modelo Preditivo, ele irá
#   "aprender" muito mais sobre o que é um bom pagador o que o mau pagador, tornando assim nosso modelo tendencioso.

# - E agora como resolver este problema? Iremos aplicar a técnica de balanceamento de dados chamada SMOTE

# - Procurar e arrastar o módulo "Select Column in Dataset"
# - Conectar a primeira porta de saída do módulo "Split Data" em "Select Column in Dataset"
# - Configurar o módulo "Select Column in Dataset" para remover as colunas Duration, CreditAmount e Age pois as mesmas foram criadas  
#   de forma categóricas. Não podemos deixas as colunas com as "mesmas" informações pois deixaria o modelo tendencioso

# - Procurar e arrastar o módulo "Edit Metadata"
# - Conectar o módulo "Select Column in Dataset" em "Edit Metadata"
# - Configurar o módulo "Edit Metadata" selecionando nosso variável alvo CreditStatus

# - Procurar e arrastar o módulo "SMOTE"
# - Conectar o módulo "Edit Metadata" em "SMOTE"
# - Configurar o módulo "SMOTE" selecionando nosso variável alvo CreditStatus
#   Configurar SMOTE percentage em 100, adicionar o valor 2 em Numbers e 123456 em Random seed



## Seleção de Variáveis (Feature Selection)

# Método 1

# - Procurar e arrastar o módulo "Filter Based Feature Selection"
# - Conectar o módulo "SMOTE" em "Filter Based Feature Selection"

# - Configurar "Filter Based Feature Selection":
#   Escolher o método Mutual (pois é um problema de classificação)
#   Escolher a variável alvo (target)
#   Escolher o número de variáveis mais relevantes a serem destacadas (escolher 8 e aparecerão as oito mais relevantes)

# Método 2

#  -> Para utilizar este método precisaremos treinar um modelo de classificação antes

# - Procurar e arrastar o módulo "Two-Class Decision Forest" (este é o módulo com o algoritmo de classificação escolhido)
# - Procurar e arrastar o módulo "Train Model"
# - Conectar o módulo "Two-Class Decision Forest" na primeira porta de entrada do módulo "Train Model"
# - Conectar o módulo "SMOTE" na segunda porta de entrada do módulo "Train Model"

# - No módulo "Train Model" escolher a variável alvo/target/preditora (CreditStatus)

#  -> Até aqui criamos um Modelo para fazer a melhor escolha das variáveis que irão ser usadas na criação da próxima versão do Modelo

# - Procurar e arrastar o módulo "Permutation Feature Import" (este módulo também pode usar em Regressão)

# - Conectar o módulo "Train Model" na primeira porta de entrada do módulo "Permutation Feature Import"
# - Conectar a segunda saída do módulo Split Data na segunda porta de entrada do módulo "Permutation Feature Import"

# - No módulo "Permutation Feature Import" configurar a métrica a ser usada para Classification - Accuracy

# - Visualizar o resultado no módulo "Permutation Feature Import"


#  -> Para o nosso experimento foi escolhido o método utilizando lingugem R para seleção das variáveis do modelo



## Criação dos Modelos

#  -> Iremos criar 4 modelos diferentes e compara-los
#  -> Serão 3 modelos criados no Ambiente Azure ML e 1 modelo criado aqui

# - Copiar os módulos de "Select Column in Dataset", "Edit Metadata" e "SMOTE" já existentes e colar para serem usados na criação
#   dos modelos

# - Configurar o módulo "Select Column in Dataset" para deixar as variáveis escolhidas
# - Configurar o módulo "Edit Metadata" selecionando a variável alvo CreditStatus e em Fields selecionar Label
# - Configurar o módulo "SMOTE" com a mesma configuração do anterior

# - Conectar a primeira porta de saída do Módulo "Split Data" no novo módulo "Select Column in Dataset"
# - Conectar o novo módulo "Select Column in Dataset" no novo módulo "Edit Metadata"
# - Conectar o novo módulo "Edit Metadata" no novo módulo "SMOTE" 

# Modelo 1

# - Procurar e arrastar o módulo "Two-Class Bayes Point Machine"
# - Procurar e arrastar o módulo "Train Model"
# - Procurar e arrastar o módulo "Score Model"

# - Configurar o módulo "Two-Class Bayes Point Machine" colocando 100 em Number of training iterations

# - Conectar o módulo "Two-Class Bayes Point Machine" na primeira porta do módulo "Train Model"
# - Conectar o módulo "SMOTE" na segunda porta do módulo "Train Model"
# - Conectar o módulo "Train Model" na primeira porta de "Score Model"
# - Conectar a segunda porta de "Split Data" na segunda porta de "Score Model"

# Modelo 2

# - Procurar e arrastar o módulo "Two-Class Neural Network"
# - Procurar e arrastar o módulo "Train Model"
# - Procurar e arrastar o módulo "Score Model"

# - Conectar o módulo "Two-Class Neural Network" na primeira porta do módulo "Train Model"
# - Conectar o módulo "SMOTE" na segunda porta do módulo "Train Model"
# - Conectar o módulo "Train Model" na primeira porta de "Score Model"
# - Conectar a segunda porta de "Split Data" na segunda porta de "Score Model"

# Modelo 3

# - Procurar e arrastar o módulo "Two-Class Support Vector"
# - Procurar e arrastar o módulo "Train Model"
# - Procurar e arrastar o módulo "Score Model"

# - Conectar o módulo "Two-Class Support Vector" na primeira porta do módulo "Train Model"
# - Conectar o módulo "SMOTE" na segunda porta do módulo "Train Model"
# - Conectar o módulo "Train Model" na primeira porta de "Score Model"
# - Conectar a segunda porta de "Split Data" na segunda porta de "Score Model"


# Comparando os Modelos

# - Procurar e arrastar dois módulos "Evaluete Model"

# - Conectar o módulo "Score Model" do Modelo 1 na primera porta do primeiro "Evaluete Model"
# - Conectar o módulo "Score Model" do Modelo 2 na segunda porta do primeiro "Evaluete Model"

# - Conectar o módulo "Score Model" do Modelo 2 na primera porta do segundo "Evaluete Model"
# - Conectar o módulo "Score Model" do Modelo 3 na segunda porta do segundo "Evaluete Model"

# - Ao clicar em Visualize no módulo "Evaluete Model" ficar atento ao gráfico:
#   A linha azul é do primeiro modelo da primeira porta e a vermelha do segundo modelo na segunda porta
#   A linha que estiver mais próximo da esquerda é o melhor modelo
#   Ao clicar na legenda Azul ou Vermelha vemos as métricas como Accurary e Precision abaixo 

