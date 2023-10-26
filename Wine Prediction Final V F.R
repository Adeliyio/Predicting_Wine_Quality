# Set working directory
setwd("C:/Users/User/Desktop/Personal Projects/Wine Quality Prediction")

# Load libraries
library(readr)
library(tidyverse)
library(corrplot)
library(ggplot2)
library(gbm)
library(GGally)
library(plotly)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(e1071)
library(h2o)
library(viridis)
library(knitr)

# Read CSV file
wine <- read_csv("winequality-red.csv")

# Preprocess - Correct column names and turn quality variable into factor
colnames(wine) <- str_replace_all(colnames(wine), " ", "_")
wine$quality <- as.factor(wine$quality)

# Data Exploration and Visualization
head(wine,10)
summary(wine)
str(wine)

## visualisation 

ggpairs(wine %>% select(-c(residual_sugar, free_sulfur_dioxide, total_sulfur_dioxide, chlorides)),
        mapping = aes(color = quality, alpha = 0.4),
        lower = list(continuous = "points"),
        upper = list(continuous = "blank"),
        axisLabels = "none", switch = "both")

# Interactive Graph 1
plot_ly(wine, x = ~alcohol, y = ~volatile_acidity, size = ~sulphates,
        type = "scatter", mode = "markers", color = ~quality, colors = viridisLite::viridis(6), text = ~quality) %>%
  layout(title = "Wine Quality",
         xaxis = list(title = 'Alcohol'),
         yaxis = list(title = 'Volatile Acidity'),
         coloraxis = list(colorscale = "Viridis", colorbar = list(title = "Quality")))

#  Interactive Graph 2
plot_ly(wine, x = ~alcohol, y = ~volatile_acidity, z = ~sulphates, color = ~quality, 
        hoverinfo = 'text', colors = viridisLite::viridis(6), 
        text = ~paste('Quality:', quality, '<br>Alcohol:', alcohol, '<br>Volatile Acidity:', volatile_acidity, '<br>Sulphates:', sulphates)) %>% 
  add_markers(opacity = 0.8) %>% 
  layout(title = "3D Wine Quality",
         scene = list(xaxis = list(title = 'Alcohol'),
                      yaxis = list(title = 'Volatile Acidity'),
                      zaxis = list(title = 'Sulphates')),
         margin = list(l = 0, r = 0, b = 0, t = 40), # Adjust margin for better title display
         annotations = list(yref = 'paper', xref = "paper", y = 1.05, x = 1.1, text = "Quality", showarrow = FALSE),
         coloraxis = list(colorscale = "Viridis", colorbar = list(title = "Quality")))


# Cross Validation Setup
set.seed(1)
inTrain <- createDataPartition(wine$quality, p = 0.9, list = FALSE)
train <- wine[inTrain,]
valid <- wine[-inTrain,]

# Decision Tree via rpart
rpart_model <- rpart(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, train)
rpart.plot(rpart_model)
rpart_result <- predict(rpart_model, newdata = valid[, !colnames(valid) %in% "quality"], type = 'class')
confusionMatrix(rpart_result, valid$quality)
varImp(rpart_model) %>% as.data.frame() %>% kable()

# Extract accuracy from the confusion matrix
accuracy_rpart <- confusionMatrix(rpart_result, valid$quality)$overall["Accuracy"]
accuracy_rpart

# Random Forest
rf_model <- randomForest(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, train)
rf_result <- predict(rf_model, newdata = valid[, !colnames(valid) %in% "quality"])
confusionMatrix(rf_result, valid$quality)
varImpPlot(rf_model)

# Extract accuracy from the confusion matrix
accuracy_rf <- confusionMatrix(rf_result, valid$quality)$overall["Accuracy"]
accuracy_rf

# xgboost with histogram
data.train <- xgb.DMatrix(data = data.matrix(train[, !colnames(train) %in% "quality"]), label = train$quality)
data.valid <- xgb.DMatrix(data = data.matrix(valid[, !colnames(valid) %in% "quality"]))
parameters <- list(
  booster = "gbtree",
  silent = 0,
  eta = 0.08,
  gamma = 0.7,
  max_depth = 8,
  min_child_weight = 2,
  subsample = 0.9,
  colsample_bytree = 0.5,
  colsample_bylevel = 1,
  lambda = 1,
  alpha = 0,
  objective = "multi:softmax",
  eval_metric = "merror",
  num_class = 7,
  seed = 1,
  tree_method = "hist",
  grow_policy = "lossguide"
)
xgb_model <- xgb.train(parameters, data.train, nrounds = 100)
xgb_pred <- predict(xgb_model, data.valid)
confusionMatrix(as.factor(xgb_pred + 2), valid$quality)


# Extract accuracy from the confusion matrix
accuracy_xgboost <- confusionMatrix(as.factor(xgb_pred + 2), valid$quality)$overall["Accuracy"]
accuracy_xgboost

# Naive Bayes
nb_model <- naiveBayes(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, data = train)
nb_result <- predict(nb_model, newdata = valid[, !colnames(valid) %in% "quality"])
confusionMatrix(nb_result, valid$quality)

# Extract accuracy from the confusion matrix
accuracy_nb <- confusionMatrix(nb_result, valid$quality)$overall["Accuracy"]
accuracy_nb

# Gradient Boosting Regression
gbm_model <- gbm(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, data = train, distribution = "gaussian", n.trees = 100)
gbm_result <- predict.gbm(gbm_model, newdata = valid[, !colnames(valid) %in% "quality"])
gbm_result <- round(gbm_result)


# Convert variables to factors with the same levels
gbm_result <- as.factor(gbm_result)
valid$quality <- as.factor(valid$quality)

# Create confusion matrix
confusionMatrix(gbm_result, valid$quality)

# Extract accuracy from the confusion matrix
accuracy_gbm <- confusionMatrix(gbm_result, valid$quality)$overall["Accuracy"]
accuracy_gbm
# SVM
svm_model <- svm(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, train)
svm_result <- predict(svm_model, newdata = valid[, !colnames(valid) %in% "quality"])
confusionMatrix(svm_result, valid$quality)

# Extract accuracy from the confusion matrix
accuracy_svm <- confusionMatrix(svm_result, valid$quality)$overall["Accuracy"]

#Accuracies
accuracy_rpart
accuracy_rf
accuracy_xgboost
accuracy_nb
accuracy_gbm
accuracy_svm

# Compare models
model_names <- c("Decision Tree (rpart)", "Random Forest", "XGBoost", "Naive Bayes", "Gradient Boosting Regression", "SVM")
accuracies <- c(0.5541, 0.6624, 0.6561, 0.6051, 0.0127, 0.5796)
results <- data.frame(Model = model_names, Accuracy = accuracies)
results <- results[order(results$Accuracy, decreasing = TRUE), ]
results
