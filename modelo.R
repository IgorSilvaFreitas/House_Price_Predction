##################################################################
######### Housing Prices Competition for Kaggle Learn Users ######
##################################################################





#Pacotes
library(readr)
library(naniar)
library(ggplot2)
library(corrplot)
library(caret)
library(mlr)
library(dplyr)
library(RANN)
library(xgboost)
library(h2o)

#Lendo bases de dados
train <- read_csv("D:/Users/Igor/Documents/Machine Learning/Kaggle competition/home-data-for-ml-course/train.txt")
test <- read_csv("D:/Users/Igor/Documents/Machine Learning/Kaggle competition/home-data-for-ml-course/test.txt")


#Verificando dados faltantes
gg_miss_var(train)

#4 variáveis serão excluidas por possuirem alta quantidade de dados faltantes, são elas:
#-PoolQC
#-MiscFeature
#-Alley
#-Fence
#FireplaceQu
#LotFrontage

train <- train |> 
  select(-c("PoolQC","MiscFeature","Alley","Fence","FireplaceQu","LotFrontage"))
test <- test |> 
  select(-c("PoolQC","MiscFeature","Alley","Fence","FireplaceQu","LotFrontage"))


#Verificando classes das colunas
str(train)


aux <- train |> distinct(MSSubClass)
#Variável aparenta ser categorica, poor isso será transformada
train$MSSubClass <- as.factor(train$MSSubClass)
test$MSSubClass <- as.factor(test$MSSubClass)

#Os anos também serão transformada para categoricas
train$YearBuilt <- as.factor(train$YearBuilt)
train$YearRemodAdd <- as.factor(train$YearRemodAdd)
train$YrSold <- as.factor(train$YrSold)

test$YearBuilt <- as.factor(test$YearBuilt)
test$YearRemodAdd <- as.factor(test$YearRemodAdd)
test$YrSold <- as.factor(test$YrSold)


#Transformando character em factor
train <- train |> mutate_if(is.character, as.factor)
test <- test |> mutate_if(is.character, as.factor)

#Há poucos dados faltantes, essas linhas serão excluidas
sum(is.na(train))
train <- na.omit(train)

#transformando base em data.frame
train <- as.data.frame(train)
test <- as.data.frame(test)

#Analise de correlação das variáveis
corrplot(cor(train), method ='number')


#Separando em amostras treino e teste para verificação de métricas
data <- createDataPartition(train$SalePrice, p=0.75, list = F)

treino <- train[data,]
teste <- train[-data,]

#Realizando alguns pré-processamentos

preproc <- preProcess(train[,2:74], method = c("corr","nzv","center", "scale","YeoJohnson"))
train <- predict(preproc,train)
teste <- predict(preproc,teste)

#Método xgboost
set.seed(100)
controle <- trainControl(method = "cv", number = 5)

modelo_xgb <- caret::train(SalePrice~ ., data=treino, method="xgbLinear",trControl=controle)
modelo_xgb


#Aplicando o modelo na amostra Teste
preditor <- predict(modelo_xgb, teste)

#Estimando o erro fora da amostra
caret::postResample(preditor,teste$SalePrice)


#Aplicando na amostra test disponibilizada na competição
test <- predict(preproc, test)

test <- na.omit(test)

pred <- predict(modelo_xgb, test)

test$SalePrice <- pred

submit <- test[,c(1,68)]


