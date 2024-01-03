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


