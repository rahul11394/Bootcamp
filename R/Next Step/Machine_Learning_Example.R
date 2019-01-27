library(C50)
library(e1071)
library(corrplot)
library(AppliedPredictiveModeling)
library(caret)
library(pROC)
data(churn)

# Basic box plot
library(ggplot2)
plot1 <- ggplot(churnTrain, aes(x=total_day_minutes, y=churn) , colour = 'red') + geom_jitter(aes(colour = churn))
plot1

plot2 <- ggplot(churnTrain , aes(x=churn) , stat ="count")
plot2 + geom_bar(colour = "blue", fill = "black")

plot3 <- ggplot(churnTrain, aes(y=total_intl_calls, x=churn)) + 
  geom_boxplot(outlier.colour="red", outlier.shape=8, outlier.size=1)
plot3 + scale_color_manual(values = c("red","blue"))

plot4 <- qplot(number_customer_service_calls ,  total_day_calls , data = churnTrain, geom = "jitter" , colour = churn)
plot4


# Prepocessing and data cleaning
# factors

churnTrain$international_plan <- ifelse(churnTrain$international_plan == "yes", 1, 0)
churnTrain$voice_mail_plan <- ifelse(churnTrain$voice_mail_plan == "yes", 1, 0)
churnTest$international_plan <- ifelse(churnTest$international_plan == "yes", 1, 0)
churnTest$voice_mail_plan <- ifelse(churnTest$voice_mail_plan == "yes", 1, 0)

# Drop varaible not required 
churnTrain = churnTrain[,-c(1,3)]
churnTest = churnTest[,-c(1,3)]

# drop the churn response
X = churnTrain[,-18] 
y = churnTrain[,18]

# converting y as numeric from character
Y <- as.numeric(y)
Y[Y==2] <- 0

# correlation Plot
corre <- cor(cbind(X,Y))
corrplot(corre,order = "hclust",tl.cex = .45)

# near zero varaince
zv_cols = nearZeroVar(X)
X = X[,-zv_cols]

# finding linear combination 
l_c <- findLinearCombos(X)

# Check for skewness using apply function
Sk <- apply(X,2,skewness)
total_intl_call_tran <- BoxCoxTrans(X$total_intl_calls)
X$total_intl_calls <- predict(total_intl_call_tran,X$total_intl_calls)

# no missing values - no imputation required
missing <- apply(X,2,is.na)

# removing high correlation
correlation <- cor(X)
highCorr <- findCorrelation(correlation, cutoff = .85)
X <- X[,-highCorr]

# combining the variables
data_1 <- cbind(y,X)

#sampling using stratified sampling for unbalanced data set
rep_sam <- createDataPartition(data_1$y, p=.80,times =1)
str(rep_sam)
train_set <- data_1[rep_sam$Resample1,]
val_set <- data_1[-rep_sam$Resample1,]

# model building
model <-glm(y~.,data = train_set, family = "binomial")
summary(model)
p <- predict(model,val_set[,-1] , type = "response")
p[p > .85] = 1
p[p <= .85] = 0
validation <- ifelse(val_set$y == "yes", 1, 0)
c_m <- confusionMatrix(p,reference = validation)
r <- roc(response = val_set[,1] , predictor = p, auc = TRUE)
plot(r)
