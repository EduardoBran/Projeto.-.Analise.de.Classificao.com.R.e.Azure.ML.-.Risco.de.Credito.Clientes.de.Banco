####  Análise de Classificação com Linguagem R e Azure ML  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco")
getwd()


####  Criando o Experimento no Azure ML  ####


## Criar Experimento

# - Criar um novo experimento chamado "Classificação - Risco de Crédito"


## Carregando dataset

# - Procurar e arrastar o módulo "German Credit Card UCI dataset"


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

