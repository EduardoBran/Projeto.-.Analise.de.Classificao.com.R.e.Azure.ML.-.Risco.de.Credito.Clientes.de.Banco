####  Análise de Classificação com Linguagem R e Azure ML  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco")
getwd()

## Carregando Pacotes
library(dplyr)
library(ggplot2)
library(ROSE)
library(randomForest)
library(ROCR)  # Gerando uma curva ROC em R
library(caret) # Cria confusion matrix
library(C50)   # Otimização do Modelo (este pacote permite que você dê peso aos erros, construindo assim um resultado ponderado)


#### Carregando e Convertendo os Dados

## Carrega o dataset antes da transformacao (baixado do Azure ML)
df <- read.csv("German_Credit_Card_UCI_-_dataset.csv")
head(df)
dim(df)


## Analisando os tipos das variáveis
str(df)
summary(df)

# - Como constatamos na fonte dos dados, as variáveis do tipo character são do tipo factor, precisa então ser convertidas.


## Convertendo as variáveis

# Selecione apenas as variáveis do tipo caractere para conversão
colunas_chr <- sapply(df, is.character)

# Converta as variáveis do tipo caractere e a última variável para fatores 
df <- mutate_if(df, colunas_chr, as.factor)
df$X1.1 <- as.factor(df$X1.1)
str(df)


## Alterando o nome das variáveis (pesquisado na fonte)
names(df) <- c("CheckingAcctStat", "Duration", "CreditHistory", "Purpose", "CreditAmount", "SavingsBonds", "Employment",
               "InstallmentRatePecnt", "SexAndStatus", "OtherDetorsGuarantors", "PresentResidenceTime", "Property", "Age",
               "OtherInstallments", "Housing", "ExistingCreditsAtBank", "Job", "NumberDependents", "Telephone", "ForeignWorker", "CreditStatus")
head(df)



#### Análise Exploratória para Engenharia de Atributos em Variáveis Numéricas

## Calcular o número de valores únicos para cada variável
unique_values <- lapply(df, function(x) length(unique(x)))

## Exibir o número de valores únicos para cada variável
for (i in seq_along(unique_values)) {
  cat("Variável:", names(unique_values)[i], "\n")
  cat("Unique Values:", unique_values[[i]], "\n\n")
}

# - Através do código acima podemos constatar que as variáveis "Duration", "CreditAmount" e "Age" possuem muitos valores únicos.

# - E variáveis com uma grande quantidade de valores únicos podem apresentar problemas durante o trienamento de um Modelo de Classificação.
# - O que fazer neste caso?
#   Iremos converter essas três variáveis de Numéricas para Qualitativas (categóricas/factor)

# - Iremos então criar e colocar os dados destas variáveis em 4 ou 5 categorias diferentes
#   Exemplo: pegaremos a coluna "CreditAmount" que possui 921 valores únicos e colocaremos estes valores em categorias

# - Isto não é obrigatório. O fato de aplicar este tipo de engenharia de atributos neste dataset não significa que devemos aplicar me outro



#### Aplicando Engenharia de Atributos

# Carregando funções
source("src/ClassTools.R")

## Criando 3 novas variáveis que serão transformadas de variáveis numéricas para variáveis categórias utilizando duas formas

# Forma 1 (Intervalar - Usando Função do script ClassTools )
toFactors <- c("Duration", "CreditAmount", "Age")
maxVals <- c(100, 1000000, 100)
facNames <- unlist(lapply(toFactors, function(x) paste(x, "_f", sep = "")))
df[, facNames] <- Map(function(x, y) quantize.num(df[, x], maxval = y), toFactors, maxVals)

head(df)
str(df)
summary(df)

# Forma 2 (Discreta - Usando Função criada neste Script)
criar_categorias <- function(variavel, num_categorias) {
  # Criar breakpoints para dividir a variável em categorias
  breakpoints <- quantile(variavel, probs = seq(0, 1, length.out = num_categorias + 1))
  
  # Criar categorias
  categorias <- cut(variavel, breaks = breakpoints, labels = seq(1, num_categorias), include.lowest = TRUE)
  
  return(categorias)
}
num_categorias <- 5
df$Duration_Categoria <- criar_categorias(df$Duration, num_categorias)
df$CreditAmount_Categoria <- criar_categorias(df$CreditAmount, num_categorias)
df$Age_Categoria <- criar_categorias(df$Age, num_categorias)

head(df)
str(df)
summary(df)


# A escolha entre representar uma variável numérica como categorias intervalares (forma 1) ou discretas (forma 2) depende
# do contexto do problema e da relação entre os valores dessa variável e a variável de saída (neste caso, CreditStatus).

# Intervalar: Prós   - Mantém a informação sobre a ordem dos valores.
#             Contra - Introduz mais complexidade no modelo, especialmente se o número de categorias for grande.
# Discreta  : Prós   - Reduz a dimensionalidade do conjunto de dados e facilita a interpretação,
#             Contra - Perde a informação sobre a ordem exata dos valores.

# - Vamos manter ambas as variáveis da forma 1 e forma 2 no nosso dataset


## Balanceamento no Dataset

# - Quando olhamos para a nossa variável alvo CreditStatus, podemos constatar que existem muito mais valores 1 (yes) do que 2 (no),
#   ou seja, nosso dataset possui muito mais BONS pagadores do que MAUS pagadores

# - E este desbalanceamento da variável alvo é um problema, pois se apresentarmos os dados desta forma ao Modelo Preditivo, ele irá
#   "aprender" muito mais sobre o que é um bom pagador o que o mau pagador, tornando assim nosso modelo tendencioso.

# - E agora como resolver este problema? Iremos aplicar a técnica de balanceamento de dados chamada SMOTE


##  Aplicando Balanceamento

# Forma 1 (Utilizando função de ClassTools - duplicando linhas)
df_balanceado1 <- df
df_balanceado1 <- equ.Frame(df_balanceado1, 2)

# Forma 2 (Utilizando pacote ROSE - removendo linhas)
df_balanceado2 <- df
df_balanceado2 <- df_balanceado2[, !(names(df_balanceado2) %in% c("Duration", "CreditAmount", "Age", 
                                                                  "Duration_Categoria", "CreditAmount_Categoria", "Age_Categoria"))]
df_balanceado2 <- ROSE(CreditStatus ~ .,  data = df_balanceado2, seed = 123, N = 2 * table(df_balanceado2$CreditStatus)[[2]])
df_balanceado2 <- df_balanceado2$data

df <- df_balanceado1
df2 <- df_balanceado2

summary(df)
summary(df2)

# - Na Forma 1, você duplicou linhas do grupo minoritário para equilibrar as classes, mantendo todas as observações originais do grupo
#   majoritário. Isso pode ser útil quando você deseja simplesmente aumentar o número de observações da classe minoritária.
# - Na Forma 2, você utilizou a técnica SMOTE do pacote ROSE para gerar novas observações sintéticas da classe minoritária, e em seguida,
#   você removeu as observações da classe majoritária que não foram utilizadas no processo de geração sintética. 
#   Isso resulta em um conjunto de dados menor, mas balanceado.



#### Análise Exploratória dos Dados com Gráficos

## Plots (ggplot2)

# Exibindo diversos comparativos de bom pagador ou não dependendo da variável
plots <- lapply(colNames2, function(x) {
  if (x %in% names(df)) {
    if (is.factor(df[[x]])) {
      ggplot(df, aes(x = .data[[x]], fill = CreditStatus)) +
        geom_bar() + 
        facet_grid(. ~ CreditStatus) + 
        ggtitle(paste("Total de Crédito Bom/Ruim por", x))
    } else {
      ggplot() +
        ggtitle(paste("Coluna", x, "não é fator nem numérica. Não é possível criar gráfico."))
    }
  } else {
    ggplot() +
      ggtitle(paste("Coluna", x, "não existe no dataframe."))
  }
})
print(plots)

# Plots CreditStatus vs CheckingAcctStat
plots_CreditStatus <- lapply(colNames2, function(x) {
  if (is.factor(df[[x]]) && x != "CheckingAcctStat") {
    ggplot(df, aes_string("CheckingAcctStat", fill = x)) +
      geom_bar() + 
      facet_grid(paste(x, " ~ CreditStatus")) + 
      ggtitle(paste("Total de Crédito Bom/Ruim CheckingAcctStat e", x))
  } else {
    NULL
  }
})
print(plots_CreditStatus)



#### Seleção de Variáveis (Feature Selection)

#  -> Vamos criar um Modelo de Regressão utilizando o algoritmo RandomForest para nos auxiliar na Seleção de Variáveis

## Criando Modelo (Utilizando RandomForest - pode ser aplicada tanto em Regressão como Classificação)
#  Para este tipo de problema (técnica de feature selecion o atributo "importante = TRUE" precsa estar)

modelo <- randomForest(CreditStatus ~ .
                       - Duration
                       - Age
                       - CreditAmount, 
                       data = df, 
                       ntree = 100, nodesize = 10, importance = T)

modelo <- randomForest(CreditStatus ~
                         CheckingAcctStat + Duration_f + CreditHistory + SavingsBonds + 
                         CreditAmount_f + Property + Employment + Purpose,
                       data = df, 
                       ntree = 100, nodesize = 10, importance = T)

# Visualizando por números
print(modelo$importance)

# Visualizando Modelo Por Gráficos

# forma 1 (quanto mais a direita melhor)
varImpPlot(modelo)

# forma 2 (quando tem poucas variáveis)
barplot(modelo$importance[, 1], main = "Importância das Variáveis", col = "skyblue")      

# forma 3 (usando ggplot, método mais profissional)
importancia_ordenada <- modelo$importance[order(-modelo$importance[, 1]), , drop = FALSE] 

df_importancia <- data.frame(
  Variavel = rownames(importancia_ordenada),
  Importancia = importancia_ordenada[, 1]
)
ggplot(df_importancia, aes(x = reorder(Variavel, -Importancia), y = Importancia)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(title = "Importância das Variáveis", x = "Variável", y = "Importância") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 10))


#  -> Até aqui criamos um Modelo para fazer a melhor escolha das variáveis que irão ser usadas na criação da próxima versão do Modelo



#### Criação do Modelo

# - Iremos criar 4 modelos diferentes e compara-los
# - Serão 3 modelos criados no Ambiente Azure ML e 1 modelo criado aqui utilizando a Linguagem R

# Cross Tabulation     (vendo a quantidade de valores "1" e valores "2")
table(df$CreditStatus)


## Criando dados de treino e teste

# Funcao para gerar dados de treino e dados de teste (50% para treino e 50% para teste por conta de ser modelo de classificação)
splitData <- function(dataframe, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)/2))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset = trainset, testset = testset)
}

# Gerando dados de treino e de teste
splits <- splitData(df, seed = 808)

# Separando os dados
dados_treino <- splits$trainset
dados_teste <- splits$testset

# Verificando o numero de linhas
nrow(dados_treino)
nrow(dados_teste)


## Criando um modelo de classificação baseado em randomForest

# Construindo o modelo (desta vez sem o parâmetro importance e com as variáveis mais significantes indicadas no modelo anterior)
modelo <- randomForest( CreditStatus ~ CheckingAcctStat
                        + Duration_f
                        + Purpose
                        + CreditHistory
                        + SavingsBonds
                        + Employment
                        + CreditAmount_f, 
                        data = dados_treino, 
                        ntree = 100, 
                        nodesize = 10)

# Imprimondo o resultado
print(modelo)



#### Avaliando a perfomance do modelo

## utilizando score (performance)

# Gerando previsões nos dados de teste
previsoes <- data.frame(observado = dados_teste$CreditStatus,
                        previsto = predict(modelo, newdata = dados_teste))

# Visualizando o resultado
View(previsoes)


## Confusion Matrix (criando manualmente)

# Formulas
Accuracy <- function(x){
  (x[1,1] + x[2,2]) / (x[1,1] + x[1,2] + x[2,1] + x[2,2])
}
Recall <- function(x){  
  x[1,1] / (x[1,1] + x[1,2])
}
Precision <- function(x){
  x[1,1] / (x[1,1] + x[2,1])
}
W_Accuracy  <- function(x){
  (x[1,1] + x[2,2]) / (x[1,1] + 5 * x[1,2] + x[2,1] + x[2,2])
}
F1 <- function(x){
  2 * x[1,1] / (2 * x[1,1] + x[1,2] + x[2,1])
}

# Criando a confusion matrix.
confMat <- matrix(unlist(Map(function(x, y){sum(ifelse(previsoes[, 1] == x & previsoes[, 2] == y, 1, 0) )},
                             c(2, 1, 2, 1), c(2, 2, 1, 1))), nrow = 2)

# Criando um dataframe com as estatisticas dos testes
df_mat <- data.frame(Category = c("Credito Ruim", "Credito Bom"),
                     Classificado_como_ruim = c(confMat[1,1], confMat[2,1]),
                     Classificado_como_bom = c(confMat[1,2], confMat[2,2]),
                     Accuracy_Recall = c(Accuracy(confMat), Recall(confMat)),
                     Precision_WAcc = c(Precision(confMat), W_Accuracy(confMat)))
print(df_mat)


## Confusion Matrix (utilizando pacote Caret)
confusionMatrix(previsoes$observado, previsoes$previsto)  # Accuracy : 0.7662 


## Gerando Curva ROC

# Gerando as classes de dados
class1 <- predict(modelo, newdata = dados_teste, type = 'prob')
class2 <- dados_teste$CreditStatus

# Criando curva
pred <- prediction(class1[,2], class2)
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = rainbow(10))



#### Otimização do Modelo

# - Iremas usar uma técnica de otimização usando um conceito de Modelo randomForest ponderado
# - Atraveś do pacote c50 ao recriar o modelo iremos penalizar os erros do modelo.


## Criando uma Cost Function (matriz de peso)
Cost_func <- matrix(c(0, 1.5, 1, 0), nrow = 2, dimnames = list(c("1", "2"), c("1", "2")))


## Recriando o Modelo
modelo_v2  <- C5.0(CreditStatus ~ CheckingAcctStat
                   + Purpose
                   + CreditHistory
                   + SavingsBonds
                   + Employment,
                   data = dados_treino,  
                   trials = 100,
                   cost = Cost_func)

print(modelo_v2)



#### Avaliando a perfomance do modelo_v2

## utilizando score (performance)

# Gerando previsões nos dados de teste
previsoes_v2 <- data.frame(observado = dados_teste$CreditStatus,
                           previsto = predict(modelo_v2, newdata = dados_teste))

# Visualizando o resultado
View(previsoes_v2)


## Criando a confusion matrix (manualmente, utilizando as formulas acima)
confMat_v2 <- matrix(unlist(Map(function(x, y){sum(ifelse(previsoes_v2[, 1] == x & previsoes_v2[, 2] == y, 1, 0) )},
                                c(2, 1, 2, 1), c(2, 2, 1, 1))), nrow = 2)

# Criando um dataframe com as estatisticas dos testes
df_mat_v2 <- data.frame(Category = c("Credito Ruim", "Credito Bom"),
                        Classificado_como_ruim = c(confMat_v2[1,1], confMat_v2[2,1]),
                        Classificado_como_bom = c(confMat_v2[1,2], confMat_v2[2,2]),
                        Accuracy_Recall = c(Accuracy(confMat_v2), Recall(confMat_v2)),
                        Precision_WAcc = c(Precision(confMat_v2), W_Accuracy(confMat_v2)))
print(df_mat_v2)


## Confusion Matrix (utilizando pacote Caret)
confusionMatrix(previsoes_v2$observado, previsoes_v2$previsto)  #  Accuracy : 0.6938  



