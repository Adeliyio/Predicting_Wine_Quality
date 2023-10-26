#https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

setwd("C:/Users/User/Desktop/Personal Projects/Wine Quality Prediction")
library(readr)
library(tidyverse)
library(corrplot)
library(ggplot2)
#install.packages("gbm")
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

# Set working directory
setwd("C:/Users/User/Desktop/Personal Projects/Wine Quality Prediction")

# Read CSV file
wine <- read_csv("winequality-red.csv")

# Preprocess - Correct column names and turn quality variable into factor
colnames(wine) <- str_replace_all(colnames(wine), " ", "_")
wine$quality <- as.factor(wine$quality)

# GGally - ggpairs
ggpairs(wine %>% select(-c(residual_sugar, free_sulfur_dioxide, total_sulfur_dioxide, chlorides)),
        mapping = aes(color = quality, alpha = 0.4),
        lower = list(continuous = "points"),
        upper = list(continuous = "blank"),
        axisLabels = "none", switch = "both")

# Plotly 2D Interactive Graph
plot_ly(wine, x = ~alcohol, y = ~volatile_acidity, size = ~sulphates,
        type = "scatter", mode = "markers", color = ~quality, text = ~quality)

# Plotly 3D Interactive Graph
plot_ly(wine, x = ~alcohol, y = ~volatile_acidity, z = ~sulphates, color = ~quality, 
        hoverinfo = 'text', colors = viridis(3), 
        text = ~paste('Quality:', quality, '<br>Alcohol:', alcohol, '<br>Volatile Acidity:', volatile_acidity, '<br>Sulphates:', sulphates)) %>% 
  add_markers(opacity = 0.8) %>% 
  layout(title = "3D Wine Quality",
         annotations = list(yref = 'paper', xref = "paper", y = 1.05, x = 1.1, text = "quality", showarrow = FALSE),
         scene = list(xaxis = list(title = 'Alcohol'),
                      yaxis = list(title = 'Volatile Acidity'),
                      zaxis = list(title = 'Sulphates')))

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

# Random Forest
rf_model <- randomForest(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, train)
rf_result <- predict(rf_model, newdata = valid[, !colnames(valid) %in% "quality"])
confusionMatrix(rf_result, valid$quality)
varImpPlot(rf_model)

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

# Naive Bayes
nb_model <- naiveBayes(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, data = train)
nb_result <- predict(nb_model, newdata = valid[, !colnames(valid) %in% "quality"])
confusionMatrix(nb_result, valid$quality)

length(valid$quality)

# Gradient Boosting Regression
gbm_model <- gbm(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, data = train, distribution = "gaussian", n.trees = 100)
gbm_result <- predict.gbm(gbm_model, newdata = valid[, !colnames(valid) %in% "quality"])
gbm_result <- round(gbm_result)

# Convert variables to factors with the same levels
gbm_result <- as.factor(gbm_result)
valid$quality <- as.factor(valid$quality)

# Create confusion matrix
confusionMatrix(gbm_result, valid$quality)


# SVM
svm_model <- svm(quality ~ alcohol + volatile_acidity + citric_acid + density + pH + sulphates, train)
svm_result <- predict(svm_model, newdata = valid[, !colnames(valid) %in% "quality"])
confusionMatrix(svm_result, valid$quality)

# Compare models
model_names <- c("Decision Tree (rpart)", "Random Forest", "XGBoost", "Naive Bayes", "Gradient Boosting Regression", "SVM")
accuracies <- c(0.5541, 0.6624, 0.6561, 0.7522, 0.6164, 0.5796)
results <- data.frame(Model = model_names, Accuracy = accuracies)
results <- results[order(results$Accuracy, decreasing = TRUE), ]
results



# Check class imbalance
class_counts <- table(wine$quality)
class_proportions <- prop.table(class_counts)

# Display class counts and proportions
class_summary <- data.frame(Class = names(class_counts), Count = as.vector(class_counts), Proportion = as.vector(class_proportions))
print(class_summary)
