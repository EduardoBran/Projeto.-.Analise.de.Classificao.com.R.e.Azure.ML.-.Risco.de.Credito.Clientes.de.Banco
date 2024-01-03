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


## Alterando o nome das variáveis (pesquisado na fonte)

names(df) <- c("CheckingAcctStat", "Duration", "CreditHistory", "Purpose", "CreditAmount", "SavingsBonds", "Employment",
               "InstallmentRatePecnt", "SexAndStatus", "OtherDetorsGuarantors", "PresentResidenceTime", "Property", "Age",
               "OtherInstallments", "Housing", "ExistingCreditsAtBank", "Job", "NumberDependents", "Telephone", "ForeignWorker", "CreditStatus")
head(df)



## Análise Exploratória (Engenharia de Atributos em Variáveis Numéricas)

# Calcular o número de valores únicos para cada variável
unique_values <- lapply(df, function(x) length(unique(x)))

# Exibir o número de valores únicos para cada variável
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



