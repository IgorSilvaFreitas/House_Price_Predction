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

#6 variáveis serão excluidas por possuirem alta quantidade de dados faltantes, são elas:
#-PoolQC
#-MiscFeature
#-Alley
#-Fence
#FireplaceQu
#LotFrontage

train <- train |> 
  select(-c("Id","PoolQC","MiscFeature","Alley","Fence","FireplaceQu","LotFrontage"))
ID <- test |> select("Id")
test <- test |> 
  select(-c("Id","PoolQC","MiscFeature","Alley","Fence","FireplaceQu","LotFrontage"))


#Verificando classes das colunas
str(train)



#Transformando character em factor
train <- train |> mutate_if(is.character, as.factor)
test <- test |> mutate_if(is.character, as.factor)

#Há poucos dados faltantes, essas linhas serão excluidas
sum(is.na(train))
train <- na.omit(train)


#Analise de correlação das variáveis
corrplot(cor(train), method ='number')


#Separando em amostras treino e teste para verificação de métricas
data <- createDataPartition(train$SalePrice, p=0.75, list = F)

treino <- train[data,]
teste <- train[-data,]

## Realizando alguns pré-processamentos



# Verificando variáveis correlacionadas
preproc1 <- preProcess(treino[,1:73], method = c("corr"))
preproc
# Não há variáveis correlacionadas

# Verificando variabilidade das variáveis
preproc2 <- nearZeroVar(treino[,1:73], names=T)
preproc2
# Há 21 variáveis com baixa variabilidade, sendo assim, pouco impactarão no modelo.
# então serão excluídas
# "Street"        "LandContour"   "Utilities"     "LandSlope"     "Condition1"   
#  "Condition2"    "RoofMatl"      "BsmtCond"      "BsmtFinType2"  "Heating"      
#  "LowQualFinSF"  "KitchenAbvGr"  "Functional"    "GarageQual"    "GarageCond"   
#  "PavedDrive"    "EnclosedPorch" "3SsnPorch"     "ScreenPorch"   "PoolArea"     
# "MiscVal"
treino <- treino |> select(-all_of(preproc2))
teste <- teste |> select(-all_of(preproc2))

# normalizando as variáveis
preproc3 <- preProcess(treino[,1:51], method = c("center", "scale"))
treino <- predict(preproc3,treino)
teste <- predict(preproc3,teste)



# tratando possíveis NA's
treino[1,1:25] <- NA
treino[2,26:51] <- NA

# variáveis númericas
treino <- as.data.frame(treino)
preproc5 <- preProcess(treino[,1:51], method = c("knnImpute"))
treino <- predict(preproc5, treino)

#variáveis categóricas
str(treino)

colnames(treino) <- make.names(colnames(treino),unique = T)
colnames(teste) <- make.names(colnames(teste),unique = T)
#------------------------------------------------------------------------------------------
# observações imputadas forçadamente a fim de conseguir rodar o algoritimo de imputação
# geral, como foram apenas 4 não devem influenciar no mdelo final
treino$Exterior1st[3] <- 'AsphShn'
treino$ExterCond[4] <- 'Po'
treino$Foundation[5] <- 'Slab'
treino$SaleCondition[6] <- 'AdjLand'
#------------------------------------------------------------------------------------------
preproc6 = mlr::impute(treino, target = "SalePrice",
                          cols = list(MSZoning = mlr::imputeLearner("classif.rpart"),
                                      LotShape=mlr::imputeLearner("classif.rpart"),
                                      LotConfig=mlr::imputeLearner("classif.rpart"),
                                      Neighborhood=mlr::imputeLearner("classif.rpart"),
                                      BldgType=mlr::imputeLearner("classif.rpart"),
                                      HouseStyle=mlr::imputeLearner("classif.rpart"),
                                      RoofStyle=mlr::imputeLearner("classif.rpart"),
                                      Exterior1st=mlr::imputeLearner("classif.rpart"),
                                      Exterior2nd=mlr::imputeLearner("classif.rpart"),
                                      MasVnrType=mlr::imputeLearner("classif.rpart"),
                                      ExterQual=mlr::imputeLearner("classif.rpart"),
                                      ExterCond=mlr::imputeLearner("classif.rpart"),
                                      Foundation=mlr::imputeLearner("classif.rpart"),
                                      BsmtQual=mlr::imputeLearner("classif.rpart"),
                                      BsmtExposure=mlr::imputeLearner("classif.rpart"),
                                      BsmtFinType1=mlr::imputeLearner("classif.rpart"),
                                      HeatingQC=mlr::imputeLearner("classif.rpart"),
                                      Electrical=mlr::imputeLearner("classif.rpart"),
                                      KitchenQual=mlr::imputeLearner("classif.rpart"),
                                      GarageType=mlr::imputeLearner("classif.rpart"),
                                      GarageFinish=mlr::imputeLearner("classif.rpart"),
                                      SaleType=mlr::imputeLearner("classif.rpart"),
                                      SaleCondition=mlr::imputeLearner("classif.rpart")))
#teste <- mlr::reimpute(teste, preproc6$desc)
treino <- preproc6$data

#Método xgboost
set.seed(100)
controle <- trainControl(method = "cv", number = 5)

modelo_xgb <- caret::train(SalePrice~ ., data=treino, method="xgbLinear",trControl=controle)

#Aplicando o modelo na amostra Teste
preditor <- predict(modelo_xgb, teste)

#Estimando o erro fora da amostra
caret::postResample(preditor,teste$SalePrice)


## Aplicando na amostra test disponibilizada na competição

# gerando o target

test$SalePrice <- NA


# realizando pré processamentos
test <- test |> select(-all_of(preproc2))

test <- predict(preproc3,test)

test <- predict(preproc5, test)

colnames(test) <- make.names(colnames(test),unique = T)

test <- mlr::reimpute(test, preproc6$desc)



pred <- predict(modelo_xgb, test)

test$SalePrice <- pred

submit <- data.frame(Id=ID,SalePrice=pred)


