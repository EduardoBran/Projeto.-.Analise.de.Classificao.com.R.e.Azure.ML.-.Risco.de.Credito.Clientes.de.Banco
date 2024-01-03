####  Análise de Classificação com Linguagem R e Azure ML  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco")
getwd()

## Carregando Pacotes




## Carregando os dados

# Carrega o dataset antes da transformacao (baixado do Azure ML)

df <- read.csv("German_Credit_Card_UCI_-_dataset.csv")
View(df)

## Alterando o nome das variáveis (pesquisado na fonte)

# CheckingAcctStat, Duration, CreditHistory, Purpose, CreditAmount, SavingsBonds, Employment, InstallmentRatePecnt, SexAndStatus, 
# OtherDetorsGuarantors, PresentResidenceTime, Property, Age, OtherInstallments, Housing, ExistingCreditsAtBank, Job, NumberDependents, 
# Telephone, ForeignWorker, CreditStatus

# 