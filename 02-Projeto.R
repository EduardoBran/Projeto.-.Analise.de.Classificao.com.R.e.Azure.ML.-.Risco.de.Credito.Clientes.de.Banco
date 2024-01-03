####  Análise de Classificação com Linguagem R e Azure ML  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco")
getwd()

## Carregando Pacotes
library(dplyr)



## Carregando os dados

# Carrega o dataset antes da transformacao (baixado do Azure ML)
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
summary(df)


