####  Classificação  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco")
getwd()


## O que é Classificação ?

# - Antes de tudo, podemos dizer que realizamos Classificação todos os dias em várias atividades como quando você vai ao trabalho ou
#   ao Shopping e escolhemos a melhor rota. Ao definir a melhor rota para nosso destino estamos realizando Classificação.

# - E como fazer isso com Machine Learning ? Neste caso teríamos que coletar dados de entrada.
#   (E estes dados seriam as ruas, dias da semana, tipo de veículo, condições climáticas e etc...)

# - Podemos ter modelo binário com duas categorias ou multiclasses com mais de duas categorias.

# - Outro exemplo: pedir empréstimo a um banco. Neste caso teremos apenas duas saídas (ou empreste ou não) e esta decisão é feita
#   através de um modelo preditivo de classificação.

# - Outro exemplo: prever se vai chover (sim/não) é uma classificação binária, enquanto prever a categoria de um animal (gato/cachorro/pássaro)
#   é uma classificação multiclasses.


## Principais da Classificação ?

# - A Classificação (assim como a Regressão) também é método de Aprendizagem Supervisionada, ou seja nós treinamos o modelo com dados
#   de entrada e dados de saída. Nosso modelo então busca a melhor função matemática capaz de prever o resultado a partir de novos
#   dados de entrada.

# - Modelo Classificação é usado quando precisamos é trabalhar e prever variáveis categóricas, ou seja classes/categorias.
#   (Equanto no Modelo de Regressão o objetivo é prever as variáveis quantitativas, fazer previsão de valores numéricos).

# - O objetivo da Classificação é entregar uma classe/categoria.

# - Se entregar duas classes é um problema binário ou Two-class. 
#   Se entregar mais de duas classes é um problema Multi-class.

# - Os erros são medidos pelas taxas de classificações incorretas.
#   (Enquanto em Regressão os erros são medidos pelos Resíduos para avalizar a performance)

# - Alguns erros podem ser mais críticos que outros e trade-offs (escolha) terão que ser feitos.


## Principais Medidas de Performance de Modelos de Classificação

# Confusion Matrix

# - É uma tabela onde temos os valores atuais (históricos) e os valores previstos pelo Modelo.


# E graças a tabela Confusion Matrix podemos extrair as Medidas de Performance tais como:


#  -> Accuracy  : Total de resultados corretos / Total de casos analisados

#  -> Recall    : Total de resultados positivos / Total de resultados corretos

#  -> Precision : Proporção de "true" / Total de resultados corretos

#  -> F-Score   : Balanceamento entre Precision e Recall

#  -> AUC       : Area Under the Curve. Onde temos um gráfico de TP no eixo y e FP no eixo x


# - A maioria dos pacotes com algoritmos de Classificação nos informam os valores das medidas acima.

