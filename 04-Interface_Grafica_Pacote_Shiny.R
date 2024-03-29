####  Testando Modelo via Pacote Shiny  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco")
getwd()


## Publicação do Modelo como um serviço WEB utilizando a plataforma Azure Machine Learning

# - Antes iremos remover os módulos que fazem algumas análises exploratórias e deixar somente os módulos necessários.


# - Feito as escolhas do módulo, clicar no menu inferior em "Set Up Web Service", escolher "Predictive Web Service"
#   (a opção "Retraining Web Service" é para um modelo que já esteja em produção)

# - Após clicar em "Predictive Web Service" o Azure ML automaticamente remove e insere módulos necessários
#   E ao observar na parte superior teremos duas abas, com o modelo original de treino e o modelo a ser colocado em produção.

# - Executar o serviço clicando em "Run"
# - Clicar em "Deploy Web Service"

# - Podemos agora testar diretamente no Azure ML ou utilizar a chamada para API disponibilizada


## Carregando Pacotes
library(shiny)
library(randomForest)

# Carregando o modelo
modelo <- readRDS("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco/modelo.rds")
print(modelo)

# Carregando o dataset antes da transformação (baixado do Azure ML)
df <- read.csv("~/Desktop/DataScience/CienciaDeDados/1.Big-Data-Analytics-com-R-e-Microsoft-Azure-Machine-Learning/15.Projeto-de-Classificação-com-R-e-Azure-ML_-_Risco_de_Credito_Clientes_Banco/dados.csv", stringsAsFactors = TRUE)
head(df)

# Ajustar níveis das variáveis categóricas
df$Duration_f <- as.factor(df$Duration_f)
df$CreditAmount_f <- as.factor(df$CreditAmount_f)



# Interface do Usuário Shiny
ui <- fluidPage(
  titlePanel("Previsão de Risco de Crédito"),
  
  sidebarLayout(
    sidebarPanel(
      # Adicione aqui os controles para entrada de dados, por exemplo:
      selectInput("checking_acct_stat", "Status da Conta Corrente", 
                  choices = c("Menor que 0" = "A11",
                              "Entre 0 e 200" = "A12",
                              "Maior que 200" = "A13",
                              "Não possui" = "A14")),
      sliderInput("duration", "Duração do Crédito", min = 0, max = 100, value = 50),
      selectInput("purpose", "Finalidade do Crédito", 
                  choices = c("Carro (novo)" = "A40",
                              "Carro (usado)" = "A41",
                              "Móveis/Equipamentos" = "A42",
                              "Rádio/Televisão" = "A43",
                              "Eletrodomésticos" = "A44",
                              "Reparos" = "A45",
                              "Educação" = "A46",
                              "Férias" = "A47",
                              "Reciclagem/Retreinamento" = "A48",
                              "Negócios" = "A49",
                              "Outros" = "A410")),
      selectInput("credit_history", "Histórico de Crédito", 
                  choices = c("Sem créditos tomados / todos os créditos pagos pontualmente." = "A30",
                              "Todos os créditos neste banco foram pagos pontualmente." = "A31",
                              "Créditos existentes pagos pontualmente até agora." = "A32",
                              "Atraso no pagamento no passado." = "A33",
                              "Conta crítica / outros créditos existentes (não neste banco)." = "A34")),
      selectInput("savings_bonds", "Títulos de Poupança", 
                  choices = c("Menos de 100 DM (Deutsche Marks) na conta poupança/títulos." = "A61",
                              "Entre 100 DM e menos de 500 DM na conta poupança/títulos." = "A62",
                              "Entre 500 DM e menos de 1000 DM na conta poupança/títulos." = "A63",
                              "1000 DM ou mais na conta poupança/títulos." = "A64",
                              "Desconhecido/nenhuma conta poupança." = "A65")),
      selectInput("employment", "Emprego Atual", 
                  choices = c("Desempregado." = "A71",
                              "Menos de 1 ano no emprego atual." = "A72",
                              "Entre 1 e menos de 4 anos no emprego atual." = "A73",
                              "Entre 4 e menos de 7 anos no emprego atual." = "A74",
                              "7 anos ou mais no emprego atual." = "A75")),
      sliderInput("credit_amount", "Valor do Crédito", min = 0, max = 1000000, value = 500000),
      # Adicione mais controles conforme necessário
      actionButton("predict_button", "Realizar Previsão")
    ),
    
    mainPanel(
      # Adicione aqui os resultados da previsão, por exemplo:
      verbatimTextOutput("prediction_output")
    )
  )
)




# Servidor Shiny
server <- function(input, output) {
  # Reaja ao botão de previsão
  observeEvent(input$predict_button, {
    # Crie um novo conjunto de dados com base nas entradas do usuário
    new_data <- data.frame(
      CheckingAcctStat = factor(input$checking_acct_stat, levels = levels(df$CheckingAcctStat)),
      Duration_f = as.factor(ifelse(as.integer(input$duration) <= 17.6, "(0,17.6]",
                                    ifelse(as.integer(input$duration) <= 31.2, "(17.6,31.2]",
                                           ifelse(as.integer(input$duration) <= 46.8, "(31.2,46.8]",
                                                  ifelse(as.integer(input$duration) <= 62.4, "(46.8,62.4]", "(62.4,100]"))))),
      Purpose = factor(input$purpose, levels = levels(df$Purpose)),
      CreditHistory = factor(input$credit_history, levels = levels(df$CreditHistory)),
      SavingsBonds = factor(input$savings_bonds, levels = levels(df$SavingsBonds)),
      Employment = factor(input$employment, levels = levels(df$Employment)),
      CreditAmount_f = as.factor(ifelse(as.integer(input$credit_amount) <= 3880, "(0,3.88e+03]",
                                        ifelse(as.integer(input$credit_amount) <= 6510, "(3.88e+03,6.51e+03]",
                                               ifelse(as.integer(input$credit_amount) <= 9140, "(6.51e+03,9.14e+03]",
                                                      ifelse(as.integer(input$credit_amount) <= 11770, "(9.14e+03,1.18e+04]", "(1.18e+04,1e+06]"))))))
    
    # Ajustar níveis novamente para garantir correspondência
    for (col in names(new_data)) {
      if (is.factor(new_data[[col]])) {
        levels(new_data[[col]]) <- levels(df[[col]])
      }
    }
    
    # Realize a previsão usando o modelo
    prediction <- tryCatch(
      predict(modelo, new_data),
      error = function(e) {
        return(paste("Erro na previsão:", e))
      }
    )
    
    # Traduza a previsão para mensagens mais compreensíveis
    prediction_message <- switch(as.character(prediction),
                                 "1" = "Crédito Aprovado!",
                                 "2" = "Crédito Reprovado!",
                                 "Erro na previsão: New factor levels not present in the training data" = "Erro na previsão: Novos níveis de fatores não presentes nos dados de treinamento",
                                 "Erro na previsão: Type of predictors in new data do not match that of the training data" = "Erro na previsão: O tipo de preditores nos novos dados não corresponde ao dos dados de treinamento",
                                 "Erro na previsão:" = "Erro na previsão: Ocorreu um erro durante a previsão")
    
    # Mostre a previsão na saída
    output$prediction_output <- renderText({
      paste("Resultado da Previsão: ", prediction_message)
    })
  })
}



# Execute o aplicativo Shiny
shinyApp(ui, server)















