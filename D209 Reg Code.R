library(randomForest)
library(ranger)
library(writexl)
library(OneR)
library(ROCR)
library(WVPlots)
library(caret)

df <- churn_clean

df1 <- df[, c("Bandwidth_GB_Year", "MonthlyCharge", "Contract", "Churn")]
df1$Bandwidth_GB_Year <- bin(df1$Bandwidth_GB_Year, nbins = 4, labels = 
                               c("Lowest 25%", "1QR 25%", "3QR 25%", "Highest 25%"), 
                             method = "content")
df1$MonthlyCharge <- bin(df1$MonthlyCharge, nbins = 4, labels = 
                           c("Lowest 25%", "1QR 25%", "3QR 25%", "Highest 25%"), 
                         method = "content") 

set.seed(2)
row <- nrow(df1)
n <- runif(row)
train <- df1[n < 0.75, ] 
test <- df1[n >= 0.75, ]
train1 <- train
test1 <- test
train1$Churn <- ifelse(train1$Churn == "Yes", 1, 0)
test1$Churn <- ifelse(test1$Churn == "Yes", 1, 0)
train$Churn <- as.factor(train$Churn)
train$Contract <- as.factor(train$Contract)
test$Churn <- as.factor(test$Churn)
test$Contract <- as.factor(test$Contract)

summary(train1)
summary(train)

#rf models
(rf <- randomForest(Churn ~ ., data = train, ntree = 1000, seed = set.seed(2)))
(rf1 <- randomForest(Churn ~., data = train1, ntree = 1000, seed = set.seed(2)))
test$pred.rf <- predict(rf, test)
test1$pred.rf <- predict(rf1, test1)

GainCurvePlot(test, "pred.rf", "Churn", "Pred Vs Churn")

table.rf <- table(test$Churn, test$pred.rf)
(cm.rf <- confusionMatrix(table.rf))

head(rf1)
(mse.rf <- mean(rf1$mse))
(rmse.rf <- sqrt(mse.rf))

#ranger package 
(rf.ran <- ranger(Churn ~., data = train, num.trees = 1000, seed = set.seed(2)))
(rf.ran1 <- ranger(Churn ~., data = train1, num.trees = 1000, seed = set.seed(2)))
test$pred.ran <- predict(rf.ran, test)$predictions
test1$pred.ran <- predict(rf.ran1, test1)$predictions

GainCurvePlot(test, "pred.ran", "Churn", "Pred.Ran Vs Churn")

table.rf.ran <- table(test$Churn, test$pred.ran)
(cm.rf.ran <- confusionMatrix(table.rf.ran))

head(rf.ran1)
mse.ran <- rf.ran1$prediction.error
(rmse.ran <- sqrt(mse.ran))

#cross validation
(rf.cross <- train(Churn ~., data = train, method = "ranger",
                  trControl = trainControl(method = "cv",
                  number = 5, verboseIter = TRUE, seeds = set.seed(2))))
(rf.cross1 <- train(Churn ~., data = train1, method = "ranger",
                   trControl = trainControl(method = "cv",
                    number = 5, verboseIter = TRUE, seeds = set.seed(2))))
test$pred.cross <- predict(rf.cross, test)
test1$pred.cross <- predict(rf.cross1, test1)

GainCurvePlot(test, "pred.cross", "Churn", "Pred.Cross Vs Churn")

table.rf.cross <- table(test$Churn, test$pred.cross)
(cm.rf.cross <- confusionMatrix(table.rf.cross))

head(rf.cross1)
(rmse.rf.cross <- min(rf.cross1$results[["RMSE"]]))

#tuned cross validation
grid <- data.frame(.mtry = c(1, 2, 3, 4, 5), .splitrule = "gini", 
                   .min.node.size = c(1, 2, 3, 4, 5))
grid1 <- data.frame(.mtry = c(1, 2, 3, 4, 5), .splitrule = "variance", 
                    .min.node.size = c(1, 2, 3, 4, 5))

(rf.cr.tune <- train(Churn ~., data = train, method = "ranger",
                    tuneGrid = grid, trControl = trainControl(method = "cv",
                      number = 5, verboseIter = TRUE, seeds = set.seed(2))))
(rf.cr.tune1 <- train(Churn ~., data = train1, method = "ranger",
                     tuneGrid = grid1, trControl = trainControl(method = "cv",
                      number = 5, verboseIter = TRUE, seeds = set.seed(2))))
test$pred.tune <- predict(rf.cr.tune, test)
test1$pred.tune <- predict(rf.cr.tune1, test1)

GainCurvePlot(test, "pred.tune", "Churn", "Pred Vs Churn")

table.rf.tune <- table(test$Churn, test$pred.tune)
(cm.rf.tune <- confusionMatrix(table.rf.tune))

head(rf.cr.tune1)
(rmse.rf.cr.tune <- min(rf.cr.tune1$results[["RMSE"]]))

#comparing models
head(test1)
pred_list <- list(test1$pred.rf, test1$pred.ran, test1$pred.cross, test1$pred.tune)
m <- length(pred_list)
actual_list <- rep(list(test1$Churn), m)
pred <- prediction(pred_list, actual_list)
rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m),
     main = "Test Set ROC Comparison",
     lwd = 2)
legend(x = "bottomright", legend = c("RF", "Ranger", "Cross", "Tuned Cross"),
       fill = 1:m)

write_xlsx(df1, "DF1_Cleaned_rf.xlsx")
write_xlsx(train1, "Train1_Cleaned_rf.xlsx")
write_xlsx(test1, "Test1_Cleaned_rf.xlsx")
write_xlsx(train, "Train_Cleaned_rf.xlsx")
write_xlsx(test, "Test_Cleaned_rf.xlsx")