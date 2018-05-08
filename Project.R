# Uploading the necessary packages
library(dplyr)
library(caret)
library(RWeka)
library(Boruta)
library(data.table)
library(randomForest)
library(lubridate)
library(tidyr)
library(mlbench)
library(pROC)
library(DMwR)

#Let's read the fraud data
FraudData= fread("/Users/farshad/Desktop/Master Folder/IE 583/Project/train.csv",header=T)
table(FraudData$is_attributed)

#The class imbalance was 0.24% have a 1 and 99.75% have a 0


#Too much memory allocation, let's make this better

#Let's randomly sample 50,000 ip addresses 

ChosenIPs<- unique(FraudData$ip)[sample(length(unique(FraudData$ip)),10000)]

#Out of those let's filter through those in our original data set

NewDataSet<- FraudData %>% filter(ip %in% ChosenIPs)

#Let's order them

NewDataSet<- NewDataSet[order(NewDataSet$ip),]

table(NewDataSet$is_attributed)

#We have the same class imbalance, thus we have our new data set that does not kill our memory 

###################################################################################################################################
#Attribute Construction on whole data set that way we can use it when trying to predict the test data set#
###################################################################################################################################

click_time<- NewDataSet$click_time
NewDataSet <- separate(NewDataSet,click_time,sep="\\s", into= c("date","time"), remove=TRUE)
NewDataSet$date<-ymd(NewDataSet$date)
NewDataSet$day<-day(NewDataSet$date)
NewDataSet$hour<-hour(hms(NewDataSet$time))
NewDataSet$minute<-minute(hms(NewDataSet$time))
NewDataSet$second<-second(hms(NewDataSet$time))
NewDataSet <- NewDataSet[, -c("date", "time")]
NewDataSet$click_time<-click_time
NewDataSet <- NewDataSet %>% group_by(ip) %>% mutate(Total.Clicks = length(ip)) %>% ungroup()
NewDataSet<-NewDataSet %>% group_by(ip, day) %>%mutate(diffDate = difftime(click_time, lag(click_time,1))) %>% ungroup()

NewDataSet$IsItEarlyMorning<-ifelse(NewDataSet$hour < 6, 1, 0)
NewDataSet$IsItLateMorning<-ifelse(NewDataSet$hour >= 6 & NewDataSet$hour<12, 1, 0)
NewDataSet$IsItAfternoon<-ifelse(NewDataSet$hour >= 12 & NewDataSet$hour<18, 1, 0)
NewDataSet$IsItNight<-ifelse(NewDataSet$hour >= 18 & NewDataSet$hour< 24, 1, 0)
NewDataSet<-NewDataSet %>% group_by(device) %>% mutate(AreTheyUsingAPopularDevice = length(device)) %>% ungroup()
NewDataSet$PopularDevice<- ifelse(NewDataSet$AreTheyUsingAPopularDevice >=20,1,0)
NewDataSet$attributed_time=NULL
NewDataSet[is.na(NewDataSet)] <- 0


#fwrite(NewDataSet,file="NewDataSet.csv")

NewDataSet<-fread("NewDataSet.csv",header=T)

###################################################################################################################################
#Train and Test data set#
###################################################################################################################################

#Make two list of chosen Ips

samp <- sample(length(unique(NewDataSet$ip)),0.67*length(unique(NewDataSet$ip)))


TrainChosenIPs<- unique(NewDataSet$ip)[samp]
TestChosenIPs<- unique(NewDataSet$ip)[-samp]

#Are they truly independent?`
#intersect(TrainChosenIPs,TestChosenIPs)


#Let us filter the chosen ips for each data set

TrainingDataSet<- NewDataSet %>% filter(ip %in% TrainChosenIPs)
TestDataSet<- NewDataSet %>% filter(ip %in% TestChosenIPs)

#fwrite(TrainingDataSet,file="TrainingDataSet.csv")
#TrainingDataSet<-fread("TrainingDataSet.csv",header=T)


#TestDataSet <- NewDataSet %>% filter(!ip %in% unique(TrainingDataSet$ip))

#fwrite(TestDataSet,file="TestDataSet.csv")
#TestDataSet<-fread("TestDataSet.csv",header=T)


###################################################################################################################################
#Attribute Selection on Training data set that way we can use it when trying to predict the test data set#
###################################################################################################################################

Boruta.Train<-Boruta(is_attributed~.,data=TrainingDataSet,mcAdj = FALSE, maxRuns = 15, 
                     holdHistory = TRUE, doTrace=2,ntree=100)


#Boruta.Train<-readRDS("BorutaObject.rds")


#Let's fix those attributes that are considered Tentative

Final.BorutaTrain<-TentativeRoughFix(Boruta.Train)

#Final.BorutaTrain$ImpHistory


#Let's get the final attributes that boruta picked

borutaList<-getSelectedAttributes(Final.BorutaTrain)

#Subset the training data on those attributes that were picked; be sure to remember 


TrainingDataSet <- as.data.frame(TrainingDataSet)

TestDataSet <- as.data.frame(TestDataSet)


New_trainingData<-TrainingDataSet[c(borutaList,"is_attributed")]

New_trainingData$click_time=NULL


###################################################################################################################################
#Building the Model random forest
###################################################################################################################################

#Let's build the model without any sampling methods

New_trainingData$is_attributed <- as.factor(New_trainingData$is_attributed)
TestDataSet$is_attributed <- as.numeric(TestDataSet$is_attributed)

ranger.model<-ranger(is_attributed~.,data=New_trainingData,num.trees=50,mtry=NULL,importance="impurity", write.forest=TRUE,
                     classification=TRUE , num.threads=8)

pred.data<-predict(ranger.model,data=TestDataSet)


table(pred.data$predictions,TestDataSet$is_attributed)
#Predicting most as 0; clearly wrong! we need to do some biased random sampling to improve the minority class

rocc <- roc(TestDataSet$is_attributed,as.numeric(pred.data$predictions))
plot(rocc)
NormalResult<-auc(rocc)

#Undersample this
down_train<- downSample(x=New_trainingData[,-ncol(New_trainingData)],y=New_trainingData$is_attributed)

table(down_train$Class)

ranger.modelus<-ranger(Class~.,data=down_train,num.trees=50,mtry=NULL,importance="impurity", write.forest=TRUE,
                       classification=TRUE , num.threads=8)

pred.dataus<-predict(ranger.modelus,data=TestDataSet)

table(pred.dataus$predictions,TestDataSet$is_attributed)

undersampling_rocc <- roc(TestDataSet$is_attributed,as.numeric(pred.dataus$predictions))

plot(undersampling_rocc)

USresult<-auc(undersampling_rocc)

#Oversample this
up_train<- upSample(x=New_trainingData[,-ncol(New_trainingData)],y=New_trainingData$is_attributed)

table(up_train$Class)

ranger.modelos<-ranger(Class~.,data=up_train,num.trees=50,mtry=NULL,importance="impurity", write.forest=TRUE,
                       classification=TRUE , num.threads=8)
pred.dataos<-predict(ranger.modelos,data=TestDataSet)

table(pred.dataos$predictions,TestDataSet$is_attributed)

oversampling_rocc <- roc(TestDataSet$is_attributed,as.numeric(pred.dataos$predictions))
plot(oversampling_rocc)
OSresult<- auc(oversampling_rocc)

#SMOTE this

smote_train<- SMOTE(is_attributed~.,data=New_trainingData,perc.over=3000,perc.under=500,k=1000)

table(smote_train$is_attributed)

ranger.modelsmote<-ranger(is_attributed~.,data=smote_train,num.trees=50,mtry=NULL,importance="impurity", write.forest=TRUE,
                          classification=TRUE , num.threads=8)
pred.datasmote<-predict(ranger.modelsmote,data=TestDataSet)

table(pred.datasmote$predictions,TestDataSet$is_attributed)

SMOTE_rocc <- roc(TestDataSet$is_attributed,as.numeric(pred.datasmote$predictions))
plot(SMOTE_rocc)
SMOTEresult<- auc(SMOTE_rocc)

###################################################################################################################################
#Building the Model logistic regression
###################################################################################################################################

#Normal
model_LR<- glm(is_attributed~.,family=binomial(link="logit"),data=New_trainingData)

prediction_LR<- predict(model_LR,TestDataSet,type="response")

table(round(prediction_LR),TestDataSet$is_attributed)

LR<-roc(TestDataSet$is_attributed,round(prediction_LR))

plot(LR)

LRresult<-auc(LR)

#Undersampling
model_LRUS <- glm(Class~.,family=binomial(link="logit"),data=down_train)

prediction_LRUS <- predict(model_LRUS, TestDataSet, type = "response")

table(round(prediction_LRUS),TestDataSet$is_attributed)

USLR<-roc(TestDataSet$is_attributed,round(prediction_LRUS))
plot(USLR)
LRusresult<-auc(USLR)

#Oversampling
model_LROS <- glm(Class~.,family=binomial(link="logit"),data=up_train)
prediction_LROS <- predict(model_LROS, TestDataSet, type = "response")

table(round(prediction_LROS),TestDataSet$is_attributed)

OSLR<-roc(TestDataSet$is_attributed,round(prediction_LROS))
plot(OSLR)
LRosresult<-auc(OSLR)

#Smote
model_LRSMOTE<-glm(is_attributed~., family=binomial(link="logit"),data=smote_train)
prediction_LRSMOTE<-predict(model_LRSMOTE, TestDataSet, type="response")
table(round(prediction_LRSMOTE), TestDataSet$is_attributed)

SMOTELR<-roc(TestDataSet$is_attributed,round(prediction_LRSMOTE))
plot(SMOTELR)
SMOTELRresult<-auc(SMOTELR)


################################Training Errors to see if there is high variance in our models#####################################

predictionforTrainingLR<-predict(model_LRUS,New_trainingData,type="response")
TLR<-roc(New_trainingData$is_attributed,round(predictionforTrainingLR))
plot(TLR)
TLRresult<- auc(TLR)

predictionforTrainingrf<-predict(ranger.modelus,data=New_trainingData)
TRF<-roc(New_trainingData$is_attributed,as.numeric(predictionforTrainingrf$predictions))
plot(TRF)
TRFresult<- auc(TRF)







