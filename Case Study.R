df = read.csv('C:/Users/limso/Desktop/DSA Data Set.csv')
##################Data Cleaning########################################
#Part 1: Replace unknown with NA
df[df=='unknown'] = NA
#Part 2: Changing some classes
colnames(df)
df[,c(2:10,13,15)] = lapply(df[,c(2:10,13,15)],as.factor)
summary(df)
#Part 3: Remove NAs
df2=df[complete.cases(df),]
1 - nrow(df2)/nrow(df) # 26% data were removed
summary(df2)
#Splitting Data
set.seed(1)
train.rows <- sample(rownames(df2), dim(df2)[1]*0.7)
train.df <- df2[train.rows, ]
valid.rows <- setdiff(rownames(df2), train.rows) 
valid.df <- df2[valid.rows, ]
dim(train.df)
dim(valid.df)

############################Current Model Evaluation##############################
table(train.df$y)
#Imbalanced data, we have more no than yes (I pick AUPRC over AUROC Curve)
#Use AUPRC (Precision Recall Curve) to evaluate the current model
library("PRROC")
fg <- train.df$ModelPrediction[train.df$y == "yes"]
bg <- train.df$ModelPrediction[train.df$y == "no"]

###Training###
#PR Curve
pr <- pr.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(pr)
#AUC: 0.08 is lower than 10% (#positive sample/total).
#This is not a good model

###Valid###
fg.v <- valid.df$ModelPrediction[valid.df$y == "yes"]
bg.v <- valid.df$ModelPrediction[valid.df$y == "no"]
pr.v <- pr.curve(scores.class0 = fg.v, scores.class1 = bg.v, curve = T)
plot(pr.v)
#Valid AUPRC: 0.08 < 0.10, not a good model

##############################Benchmark##########################################
df2$y = as.factor(df2$y)
df.bench = df2[,-21] #remove model prediction from df
logit.reg <- glm(y ~ duration, data = df.bench, family = "binomial") 
options(scipen=999)
summary(logit.reg)
logit.reg.pred.b <- predict(logit.reg, df.bench[,c(11,21)], type = "response") 

# PR Curve
fg.b <- logit.reg.pred.b[df.bench$y=='yes']
bg.b <-logit.reg.pred.b[df.bench$y=='no']
pr.b <- pr.curve(scores.class0 = fg.b, scores.class1 = bg.b, curve = T)
plot(pr.b)
#AUPRC: 0.398, this is our benchmark.
###############################Random Forest#####################################

library(randomForest)
rf.df.train <- train.df[,-c(11,21)] #remove duration and model prediction
rf.df.valid <- valid.df[,-c(11,21)]
rf.df.train$y=as.factor(rf.df.train$y)
rf.train <- randomForest(y ~ ., data = rf.df.train, ntree = 500, 
                         mtry = 4, nodesize = 5, importance = TRUE)  

rf.pred.train <- predict(rf.train,rf.df.train,type = "prob")
rf.pred.valid <- predict(rf.train,rf.df.valid,type = "prob")
forest.df <- as.data.frame(rf.pred.train)
forest.df2 <- as.data.frame(rf.pred.valid)
#PR Curve (train)
fg.f <- forest.df$yes[rf.df.train$y=='yes']
bg.f <- 1-forest.df$no[rf.df.train$y=='no']
pr.f <- pr.curve(scores.class0 = fg.f, scores.class1 = bg.f, curve = T)
plot(pr.f)
# Training AUC: 0.8688 > benchmark 0.398

#PR Curve (valid)
fg.fv <- forest.df2$yes[rf.df.valid$y=='yes']
bg.fv <- 1-forest.df2$no[rf.df.valid$y=='no']
pr.fv <- pr.curve(scores.class0 = fg.fv, scores.class1 = bg.fv, curve = T)
plot(pr.fv)
# Valid AUC: 0.4821 > benchmark 0.398

# Single Threshold 0.5
rf.pred.valid2 <- predict(rf.train,rf.df.valid,type = "class")
library(caret)
confusionMatrix(as.factor(rf.pred.valid2),as.factor(rf.df.valid$y))
# Recall: 0.9747
# Precision: 0.6234
336/(336+203)

#####################Sensitivity Analysis####################
# Duration
library(rminer)
colnames(df2)
df3 = df2[,-c(16:21)] # I removed all economic variables + model prediction
#These variables cannot be controlled by us, and are not useful for making decisions
M=fit(y~., df3, model="randomforest")
I=Importance(M, df3)
S=sort.int(I$imp, decreasing = TRUE, index.return = TRUE)
N=10
L=list(runs=1, sen=t(I$imp[S$ix[1:N]]))
LEG=names(df3)

mgraph(L, graph='IMP', leg = LEG[S$ix[1:N]], col='gray', Grid=10)
vecplot(I, graph = "VEC", xval = 9, main = "Month", Grid = 10, TC=2, sort = "decreasing")
vecplot(I, graph = "VEC", xval = 1, main = "Age", Grid = 10, TC=2, sort = "decreasing")
vecplot(I, graph = "VEC", xval = 15, main = "Previous Outcome", Grid = 10, TC=2, sort = "decreasing")
vecplot(I, graph = "VEC", xval = 11, main = "Duration", Grid = 10, TC=2, sort = "decreasing")
vecplot(I, graph = "VEC", xval = 14, main = "Number of Contacts Performed", Grid = 10, TC=2, sort = "decreasing")


