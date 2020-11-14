set.seed(678)
path <- '/Users/aj/Downloads/2dfd2de0d4f8727f873422c5d959fff5-fa71405126017e6a37bea592440b4bee94bf7b9e/titanic.csv'
titanic <-read.csv(path)

#Data information
head(titanic)
dim(titanic)
str(titanic)
sapply(titanic, function(x) sum(is.na(x)))

#Data preprocessing
library(dplyr)   
# Drop variables
clean_titanic <- titanic
clean_titanic <- select(clean_titanic, -c(PassengerId, Cabin, Name, Ticket)) 
clean_titanic <- mutate(clean_titanic, Pclass = factor(Pclass, levels = c(1, 2, 3), 
                        labels = c('Upper', 'Middle', 'Lower')), 
         Survived = factor(Survived, levels = c(0, 1), labels = c('Died', 'Survived')))
clean_titanic <- na.omit(clean_titanic) 
glimpse(clean_titanic)
tail(clean_titanic)

#Density plot of Age attribute
hist(clean_titanic$Age, las=1)
par(new=TRUE)   
plot(density(clean_titanic$Age), yaxt="n", xaxt="n",
             bty='n', xlab="", ylab="", main='')
axis(4)

#train test split Function
create_train_test <- function(data, size = 0.8, train = TRUE) {
  n_row = nrow(data)
  total_row = size * n_row
  train_sample <- 1: total_row
  if (train == TRUE) {
    return (data[train_sample, ])
  } else {
    return (data[-train_sample, ])
    }
}  

ind <- c()
acc <- c()

# Checking accuracy for assigning 70% to 96% of original data to training data.
for (i in 70:96){
  data_train <- create_train_test(clean_titanic, (i/100), train = TRUE)
  data_test <- create_train_test(clean_titanic, (i/100), train = FALSE)
  dim(data_train)
  
  #install.packages("rpart.plot")
  
  library(rpart)
  library(rpart.plot)
  fit <- rpart(Survived~., data = data_train, method = 'class')
  #rpart.plot(fit, extra = 106)
  
  predict_unseen <-predict(fit, data_test, type = 'class')
  table_mat <- table(data_test$Survived, predict_unseen)
  print(table_mat)
  
  accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
  print(paste(i,'Accuracy of the model is', round(accuracy_Test*100, digits = 2), '% on test data.'))
  ind <- append(ind,i)
  acc <- append(acc,accuracy_Test)
}

plot(ind,acc,xlab = "Percentage of data used to train the model", ylab = "Accuracy" ,type = "o")

#training and testing data.
data_train <- create_train_test(clean_titanic, 0.76, train = TRUE)
data_test <- create_train_test(clean_titanic, 0.76, train = FALSE)
dim(data_train)

--------------------------------------------------------------------------------------------
                      # Model 1: Decision tree
--------------------------------------------------------------------------------------------
  
#install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

#Setting optimization criteria.
control <- rpart.control(minsplit = 3, maxdepth = 6,cp = 0) 

#fitting the model
fit <- rpart(Survived~., data = data_train, method = 'class', control = control)

#Tree visualization
rpart.plot(fit, extra = 106)

#Predict test data
predict_unseen <-predict(fit, data_test[,2:8], type = 'class')

#complexity parameter
printcp(fit)
plotcp(fit)
table_mat <- table(data_test$Survived, predict_unseen)

#Accuracy
accuracy_Test <- sum(diag(table_mat)) / sum(table_mat)
print(paste('Accuracy of the model is', round(accuracy_Test*100, digits = 2), '% on test data.'))


--------------------------------------------------------------------------------------------
                    # Model 2: Support Vector Machine
--------------------------------------------------------------------------------------------

#install.packages('e1071') 
library(e1071) 

data_train <- create_train_test(clean_titanic, 0.78, train = TRUE)
data_test <- create_train_test(clean_titanic, 0.78, train = FALSE)

#Tuning the model with various cost and gamma values
tune.out = tune(svm, train.x = Survived ~ Age + Pclass + Sex + SibSp + Fare, data = data_train, 
                ranges = list(cost = c(0.1, 1, 5, 10, 100), gamma = c(0.005, 0.5,1,2)), kernel = "radial")

#selecting the best model
bestmodel = tune.out$best.model
summary(tune.out)

#Building the model with derived values
classifier = svm(formula = Survived ~ ., data = data_train, type = 'C-classification', 
                 kernel="radial", cost = 1, gamma = 0.5)

#predict the testing data
y_pred = predict(classifier, data_test)

#Accuracy
table_b = table(y_pred, data_test$Survived)
accuracy_Test <- sum(diag(table_b)) / sum(table_b)
print(paste('Accuracy of the model is', round(accuracy_Test*100, digits = 2), '% on test data.'))
summary(classifier)


