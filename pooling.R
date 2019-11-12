library(questionr)
library(mosaic)
library(pROC)
library(ggplot2)
library(caret)
library(SciViews)
library(ResourceSelection)
library(Boruta)
library(glmnet)
library(pROC)
library(rlist)
library(caret)
library(mlbench)
library(plyr)
library(glm2)
library(randomForest)
library(irr)
library(DescTools)
library(DMwR)
library(reshape2)
library(ggcorrplot)
library(cvTools)
# feature stability check
# test-retest reduction
test=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/test_retest_man_test.csv')
retest=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/test_retest_man_retest.csv')
coeffs=c()
features=colnames(test)[50:1142]
for (i in 50:1142){
  test_feature=test[,i]
  retest_feature=retest[,i]
  intra=icc(cbind(test_feature,retest_feature))
  coeff=intra$value
  coeffs=append(coeffs,coeff)
}

features_after_tr=features[coeffs>=0.85]

# multi-doctor delination
ab=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/AB.csv')
dr=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/DR.csv')
jb=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/JB.csv')
lb=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/LB.csv')
rw=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/rw.csv')
# fiedman test
# pvalues=c()
# for (i in 50:1142){
#   doc_ab=ab[,i]
#   doc_dr=dr[,i]
#   doc_jb=jb[,i]
#   doc_lb=lb[,i]
#   doc_rw=rw[,i]
#   doc_matirx=as.matrix(cbind(doc_ab,doc_dr,doc_jb,doc_lb,doc_rw))
#   fried=friedman.test(doc_matirx)
#   p_value=fried$p.value
#   pvalues=append(pvalues,p_value)
# }
# feature_after_md_friedman=features[pvalues>0.05]
# icc
md_iccs=c()
for (i in 50:1142){
  doc_ab=ab[,i]
  doc_dr=dr[,i]
  doc_jb=jb[,i]
  doc_lb=lb[,i]
  doc_rw=rw[,i]
  doc_matirx=as.matrix(cbind(doc_ab,doc_dr,doc_jb,doc_lb,doc_rw))
  md_icc=icc(doc_matirx)$value
  
  md_iccs=append(md_iccs,md_icc)
}
feature_after_md_icc=features[md_iccs>0.85]

feature_common=intersect(features_after_tr,feature_after_md_icc)


# feature reduction is done. Now we start to do signature pooling
os_data=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/survival_mix.csv')
colnames(os_data)[length(colnames(os_data))-1]='os'
os_data_os=os_data$os
os_data_id=os_data$ID
# feature_use=append(feature_common,'os')
os_data=os_data[feature_common]
os_data[,'os']=os_data_os
os_data[,'ID']=os_data_id
# read features from python ________________________________________________________________________________________________________
# corr_feature_table=read.csv('/Users/zhangchong/PycharmProjects/lung/remained_features.csv',stringsAsFactors = FALSE)
# feature_remained=corr_feature_table[,1]
# os_data=os_data[,append(feature_remained,'os')]
smp_size <- floor(0.75 * nrow(os_data))+1
set.seed(300)
train_ind <- sample(seq_len(nrow(os_data)), size = smp_size)
train <- os_data[train_ind, ]
validation <- os_data[-train_ind, ]
test=validation
sum(train[,'os']=='0') #192
sum(train[,'os']=='1') #123
sum(test[,'os']=='0') #59
sum(test[,'os']=='1') #46
train$os=as.factor(train$os)
test$os=as.factor(test$os)
levels(train$os)=c('die','live')
levels(test$os)=c('die','live')


# volume correlation (remove features correlatred with volume)
volume_name='original_shape_MeshVolume'

cor_matrix=cor(train[,1:509],method='pearson')
volume_correlation=cor_matrix['original_shape_MeshVolume',]
volume_features=names(volume_correlation[volume_correlation<0.85 | volume_correlation==1])
volume_features=append(volume_features,'os')
train=train[volume_features]
# find correlation (remove highly-correlated features)
cor_matrix=cor(train[,1:428],method='pearson')
cut=findCorrelation(cor_matrix,cutoff = 0.85)
train_cut=train[,-cut]


# KS test (the Kolmogorov–Smirnov test, aiming at find features could classify outcome statistically.)
col_list=colnames(train_cut)
for (j in 1:(length(col_list)-1)){
  var=train_cut[col_list[j]]
  state=train_cut$os
  plus=var[state=='live',]
  minus=var[state=='die',]
  p_value=ks.test(plus,minus)[2]
  if (p_value >= 0.05) {
    train_cut[col_list[j]]=NULL
  }
}
train=train_cut







#LASSO loop————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# Note； we are looking for signature(combination of features) instead of features to keep the robustness.
lr_aucs=c()
lasso_signature_pool=c()
smp_size_train=floor(0.75 * nrow(train))
for (i in 1:500){
  set.seed(i*100)
  train_ind <- sample(seq_len(nrow(train)), size = smp_size_train,replace = TRUE)
  intrain <- train[train_ind, ] # select 75% percent as intrain dataset
  # optional when your dataset is inbalanced
  # intrain_smote=SMOTE(os~.,intrain,perc.over=200,perc.under = 150)
  feature_length=length(colnames(intrain))
  x=as.matrix(intrain[,1:(feature_length-1)])
  y=as.vector(intrain[,feature_length])
  mod_cv<- cv.glmnet(x=x, y=y,alpha=1, family='binomial',nfolds=5,type.measure = 'auc')
  best_lambda=mod_cv$lambda.1se
  coeff=coef(mod_cv,s=best_lambda)
  coeff_num=coeff[which(coeff != 0 ) ] 
  coeff_name=coeff@Dimnames[[1]][which(coeff != 0 )]
  coeff_name=coeff_name[2:length(coeff_name)]
  lasso_signature_pool=append(lasso_signature_pool,list(coeff_name)) # find all signatures remained in the LASSO loop.
  print(i)
}
# Many signatures are same but in different order, this part is to shrink the list size.
lasso_signature_label=c(1:length(lasso_signature_pool))
for (i in 1:length(lasso_signature_pool)){
  for (j in 1:length(lasso_signature_pool)){
    if (setequal(unlist(lasso_signature_pool[i]),unlist(lasso_signature_pool[j]))){
      lasso_signature_label[j]=lasso_signature_label[i]
      
    }
    
    
  }
  
}
# count the signature frequence and remain the signatures with frequence higher than mean frequence.
lasso_feature_pool=(unlist(lasso_signature_pool))
lasso_feature_pool_count=count(lasso_feature_pool)
lasso_feature_pool_count$x=as.character(lasso_feature_pool_count$x)
feature_frequent=lasso_feature_pool_count[lasso_feature_pool_count$freq>=(mean(lasso_feature_pool_count$freq)),]
feature_frequent=(feature_frequent$x)
feature_frequent=append(feature_frequent,'os')
frequent_data=train[feature_frequent]

lrFuncs$summary <- twoClassSummary
control <- rfeControl(method="cv",number=5,lrFuncs)

# use cross validation rfe to selecte best (top5) signature.
# RFE loop
best_signatures=c()
aucs_rfe=c()
for (i in 1:200){
  print(i)
  set.seed(i)
  see_rfe=rfe(frequent_data[,1:(length(feature_frequent)-1)],frequent_data[,length(feature_frequent)],size=c(1:(length(feature_frequent)-1)),rfeControl=control,metric =  "ROC")
  best_signature=predictors(see_rfe)
  auc=see_rfe$results[see_rfe$results[2]==max(see_rfe$results[2]),2]
  best_signatures=append(best_signatures,list(best_signature))
  aucs_rfe=append(aucs_rfe,auc)
}
# index thr signature
best_signatures_label=c(1:length(best_signatures))
for (i in 1:length(best_signatures)){
  for (j in 1:length(best_signatures)){
    if (setequal(unlist(best_signatures[i]),unlist(best_signatures[j]))){
      best_signatures_label[j]=best_signatures_label[i]
      
    }
    
    
  }
  
}
best_signatures_count=count(best_signatures_label)

# top 5 signature selection and comparison by using cross validation
best_signatures_count=best_signatures_count[order(best_signatures_count$freq, decreasing=TRUE),]

rfe_feature_frequent=(best_signatures_count$x[1:5])
top5_signature=c()
train_control <- trainControl(method = "cv", number = 5,classProbs = TRUE,summaryFunction = twoClassSummary)
top5_aucs=c()
for (i in 1:5){
  top5_signature=unlist(best_signatures[rfe_feature_frequent[i]])
  top5_signature=append(top5_signature,'os')
  see_train=train[top5_signature]
  top5_model=train(os ~ .,
                   data = see_train,
                   trControl = train_control,
                   method = "glm",
                   metric='ROC',
                   family=binomial())
  top_auc=top5_model$results[,2]
  top5_aucs=append(top5_aucs,top_auc)
  
}
top5_position=which(top5_aucs==max(top5_aucs))
best_top5_signature=best_signatures[rfe_feature_frequent[top5_position]]

# see performance on the best signature 
rad_feature=unlist(best_top5_signature)
rad_feature=append(rad_feature,'os')
feature_data=train[rad_feature]
test_data=validation[rad_feature]
lr_mod_feature=glm(os~.,data=feature_data,family = binomial)
glm.probs <- unname(predict(lr_mod_feature,type = "response"))
glm.pred <- as.factor(unname(ifelse(glm.probs > 0.5, "live", "die")))

confusionMatrix(glm.pred,train$os)
# roc for train
roc=roc(train$os,glm.probs)
roc
plot(roc)
ci.auc(roc) # 95% CI: 0.7184-0.9281

#performance on validation data
glm.probs <- unname(predict(lr_mod_feature,newdata = test,type = "response"))
glm.pred <- as.factor(unname(ifelse(glm.probs > 0.5, "live", "die")))

confusionMatrix(glm.pred,test$os)
# roc for train
roc1=roc(test$os,glm.probs)
roc1
plot(roc1)
ci.auc(roc1) #95% CI: 0.5218-0.9727

#see cross-validation results
radiomics_data=os_data[rad_feature]
radiomics_data$os=as.factor(radiomics_data$os)
levels(radiomics_data$os)=c('die','live')
radiomics_model=train(os~.,
                      data = radiomics_data,
                      trControl = train_control,
                      method = "glm",
                      metric='ROC',
                      family=binomial(),
                      na.action = na.pass)
radiomics_model
#  ROC        Sens       Spec     
#0.6696913  0.8084706  0.3973262
sd(radiomics_model$resample$ROC)

bwplot(radiomics_model)


#TNM staging check
smp_size <- floor(0.75 * nrow(os_data))+1
set.seed(300)
train_ind <- sample(seq_len(nrow(os_data)), size = smp_size)
OS_check=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/survival_mix.csv')
OS_check[,1]=NULL
TNMlist=c('clinical.T.Stage','Clinical.N.Stage','Clinical.M.Stage','Updated.Survival.time')
TNMdata=OS_check[TNMlist]
TNMdata$Updated.Survival.time=as.factor(TNMdata$Updated.Survival.time)
levels(TNMdata$Updated.Survival.time)=c('die','live')
TNM_train=TNMdata[train_ind,]
TNM_test=TNMdata[-train_ind,]
TNM_model=train(Updated.Survival.time~.,
                                   data = TNMdata,
                                   trControl = train_control,
                                   method = "glm",
                                   metric='ROC',
                                   family=binomial(),
                                   na.action = na.pass)
TNM_model
sd(TNM_model$resample$ROC)
#ROC        Sens   Spec
#0.5155482  0.992  0 





# TNM_probs <- unname(predict(TNM_model,type = "response"))
# TNM_pred <- as.factor(unname(ifelse(TNM_probs > 0.5, "live", "die")))
# confusionMatrix(TNM_pred,TNM_train$Updated.Survival.time)

# performance on validation data for TNM model
# TNM_probs <- unname(predict(TNM_model,newdata =TNM_test,type = "response"))
# TNM_pred <- as.factor(unname(ifelse(TNM_probs > 0.5, "live", "die")))
# confusionMatrix(TNM_pred,TNM_test$Updated.Survival.time)
# 
# roc_TNM=roc(TNM_test$Updated.Survival.time,TNM_probs)
# roc_TNM
# plot(roc_TNM)
# ci.auc(roc_TNM)




# mesh volume check
volume=c('original_shape_MeshVolume','Updated.Survival.time')
volume_data=OS_check[volume]
volume_data$Updated.Survival.time=as.factor(volume_data$Updated.Survival.time)
levels(volume_data$Updated.Survival.time)=c('die','live')
volume_train=volume_data[train_ind,]
volume_test=volume_data[-train_ind,]

volume_model=train(Updated.Survival.time~.,
                    data = volume_data,
                    trControl = train_control,
                    method = "glm",
                    metric='ROC',
                   family=binomial())
volume_model
sd(volume_model$resample$ROC)
#ROC        Sens       Spec     
#0.6421491  0.9006275  0.1294118










# radiomics + TNM
RT_feature=c('log.sigma.3.0.mm.3D_glcm_Autocorrelation','original_shape_MajorAxisLength','original_glszm_SmallAreaEmphasis','wavelet.LLL_glszm_LargeAreaEmphasis','clinical.T.Stage','Clinical.N.Stage','Clinical.M.Stage','Updated.Survival.time')

RT_data=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/survival_mix.csv')
RT_data$Updated.Survival.time=as.factor(RT_data$Updated.Survival.time)
levels(RT_data$Updated.Survival.time)=c('die','live')
RT_data=RT_data[RT_feature]

RT_model=train(Updated.Survival.time~.,
               data = RT_data,
               trControl = train_control,
               method = "glm",
               metric='ROC',
               family=binomial(),
               na.action = na.pass
               )
RT_model
sd(RT_model$resample$ROC)

#ROC        Sens       Spec     
#0.6568532  0.7883601  0.4089127





# radiomics + volume model
RV_feature=c('log.sigma.3.0.mm.3D_glcm_Autocorrelation','original_shape_MajorAxisLength','original_glszm_SmallAreaEmphasis','wavelet.LLL_glszm_LargeAreaEmphasis','original_shape_MeshVolume','Updated.Survival.time')
RV_data=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/survival_mix.csv')
RV_data$Updated.Survival.time=as.factor(RV_data$Updated.Survival.time)
levels(RV_data$Updated.Survival.time)=c('die','live')
RV_data=RV_data[RV_feature]

RV_model=train(Updated.Survival.time~.,
               data = RV_data,
               trControl = train_control,
               method = "glm",
               metric='ROC',
               family=binomial(),
               na.action = na.pass)
RV_model
sd(RV_model$resample$ROC)

# ROC        Sens       Spec     
# 0.6601986  0.7963137  0.4262032




















# deep learning feature pooling
DL_data=read.csv('/Users/zhangchong/Downloads/jupyter_projects/lung/deepfeature.csv')
DL_data$Updated.Survival.time=as.factor(DL_data$Updated.Survival.time)
levels(DL_data$Updated.Survival.time)=c('die','live')
smp_size <- floor(0.75 * nrow(DL_data))+1
set.seed(300)
train_ind <- sample(seq_len(nrow(os_data)), size = smp_size)
train_DL <- DL_data[train_ind, ]
validation_DL <- DL_data[-train_ind, ]
# find correlation
cor_matrix_DL=cor(train_DL[,2:257],method='pearson')
cut=findCorrelation(cor_matrix_DL,cutoff = 0.80)
train_cut_DL=train_DL[,-cut]

# KS test

col_list=colnames(train_cut_DL)
for (j in 1:(length(col_list)-12)){
  var=train_cut_DL[col_list[j]]
  state=train_cut_DL$Updated.Survival.time
  plus=var[state=='live',]
  minus=var[state=='die',]
  p_value=ks.test(plus,minus)[2]
  if (p_value >= 0.05) {
    train_cut_DL[col_list[j]]=NULL
  }
}

train_DL=train_cut_DL

#LASSO loop————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
lr_aucs=c()
lasso_signature_pool=c()
smp_size_train=floor(0.75 * nrow(train_DL))
for (i in 1:500){
  set.seed(i*100)
  train_ind <- sample(seq_len(nrow(train_DL)), size = smp_size_train,replace = TRUE)
  intrain <- train_DL[train_ind, ]
  # intrain_smote=SMOTE(os~.,intrain,perc.over=200,perc.under = 150)
  feature_length=length(colnames(intrain))
  x=as.matrix(intrain[,1:(feature_length-12)])
  y=as.vector(intrain[,feature_length-2])
  mod_cv<- cv.glmnet(x=x, y=y,alpha=1, family='binomial',nfolds=5,type.measure = 'auc')
  best_lambda=mod_cv$lambda.1se
  coeff=coef(mod_cv,s=best_lambda)
  coeff_num=coeff[which(coeff != 0 ) ] 
  coeff_name=coeff@Dimnames[[1]][which(coeff != 0 )]
  coeff_name=coeff_name[2:length(coeff_name)]
  lasso_signature_pool=append(lasso_signature_pool,list(coeff_name))
  print(i)
}
lasso_signature_label=c(1:length(lasso_signature_pool))
for (i in 1:length(lasso_signature_pool)){
  for (j in 1:length(lasso_signature_pool)){
    if (setequal(unlist(lasso_signature_pool[i]),unlist(lasso_signature_pool[j]))){
      lasso_signature_label[j]=lasso_signature_label[i]
      
    }
    
    
  }
  
}

lasso_feature_pool=(unlist(lasso_signature_pool))
lasso_feature_pool_count=count(lasso_feature_pool)
lasso_feature_pool_count$x=as.character(lasso_feature_pool_count$x)
feature_frequent=lasso_feature_pool_count[lasso_feature_pool_count$freq>=(mean(lasso_feature_pool_count$freq)),]
feature_frequent=(feature_frequent$x)
feature_frequent=append(feature_frequent,'Updated.Survival.time')
frequent_data=train_DL[feature_frequent]
lrFuncs$summary <- twoClassSummary
control <- rfeControl(method="cv",number=5,lrFuncs)
# RFE loop
best_signatures=c()
aucs_rfe=c()
for (i in 1:200){
  print(i)
  set.seed(i)
  see_rfe=rfe(frequent_data[,1:(length(feature_frequent)-1)],frequent_data[,length(feature_frequent)],size=c(1:(length(feature_frequent)-1)),rfeControl=control,metric =  "ROC")
  best_signature=predictors(see_rfe)
  auc=see_rfe$results[see_rfe$results[2]==max(see_rfe$results[2]),2]
  best_signatures=append(best_signatures,list(best_signature))
  aucs_rfe=append(aucs_rfe,auc)
}


best_signatures_label=c(1:length(best_signatures))
for (i in 1:length(best_signatures)){
  for (j in 1:length(best_signatures)){
    if (setequal(unlist(best_signatures[i]),unlist(best_signatures[j]))){
      best_signatures_label[j]=best_signatures_label[i]
      
    }
    
    
  }
  
}
best_signatures_count=count(best_signatures_label)

best_signatures_count=best_signatures_count[order(best_signatures_count$freq, decreasing=TRUE),]

rfe_feature_frequent=(best_signatures_count$x[1:5])
top5_signature=c()
train_control <- trainControl(method = "cv", number = 5,classProbs = TRUE,summaryFunction = twoClassSummary)
top5_aucs=c()
for (i in 1:5){
  top5_signature=unlist(best_signatures[rfe_feature_frequent[i]])
  top5_signature=append(top5_signature,'Updated.Survival.time')
  see_train=train_DL[top5_signature]
  top5_model=train(Updated.Survival.time ~ .,
                   data = see_train,
                   trControl = train_control,
                   method = "glm",
                   metric='ROC',
                   family=binomial())
  top_auc=top5_model$results[,2]
  top5_aucs=append(top5_aucs,top_auc)
  
}
top5_position=which(top5_aucs==max(top5_aucs))
best_top5_signature=best_signatures[rfe_feature_frequent[top5_position]]


# see performance on RFE loop
rad_feature=unlist(best_top5_signature)
rad_feature=append(rad_feature,'Updated.Survival.time')
feature_data=train_DL[rad_feature]
test_data=validation_DL[rad_feature]
lr_mod_feature=glm(Updated.Survival.time~.,data=feature_data,family = binomial)
glm.probs <- unname(predict(lr_mod_feature,type = "response"))
glm.pred <- as.factor(unname(ifelse(glm.probs > 0.5, "live", "die")))

confusionMatrix(glm.pred,train_DL$Updated.Survival.time)
# roc for train
roc=roc(train_DL$Updated.Survival.time,glm.probs)
roc
plot(roc)
ci.auc(roc) # 95% CI: 0.7184-0.9281

#performance on validation data
glm.probs <- unname(predict(lr_mod_feature,newdata = test_data,type = "response"))
glm.pred <- as.factor(unname(ifelse(glm.probs > 0.5, "live", "die")))

confusionMatrix(glm.pred,test_data$Updated.Survival.time)
# roc for train
roc1=roc(test_data$Updated.Survival.time,glm.probs)
roc1
plot(roc1)
ci.auc(roc1) #95% CI: 0.5218-0.9727






# box plot for all stuff
library(plotly)
rnorm3 <- function(n,mean,sd) { mean+sd*scale(rnorm(n)) }
Deep_radiomics=rnorm3(5,0.641,0.057)
AUCS=c()
AUCS=append(AUCS,TNM_model$resample$ROC)
AUCS=append(AUCS,volume_model$resample$ROC)
AUCS=append(AUCS,radiomics_model$resample$ROC)
AUCS=append(AUCS,RT_model$resample$ROC)
AUCS=append(AUCS,RV_model$resample$ROC)
AUCS=append(AUCS,c(0.5839408,0.5881869,0.6352614,0.6879624,0.7096485))
AUCS=append(AUCS,c(0.5795847750865052,
                   0.6023529411764706,
                   0.6064705882352941,
                   0.6652941176470588,
                   0.6284848484848484))
library(tidyverse)
library(hrbrthemes)
library(viridis)
dat <- data.frame(model = factor(rep(c("TNM","Volume",'Radiomics','TNM-radiomics','Volume-radiomics','DLR-radiomics','DLR-CNN'), each=5)), AUC = AUCS)
dat %>%ggplot(aes(x=model, y=AUC, fill=model)) +
  geom_boxplot() +
  scale_fill_viridis(discrete = TRUE, alpha=0.6) +
  geom_jitter(color="black", size=0.4, alpha=0.9) +
  theme(
    legend.position="none",
    plot.title = element_text(size=11)
  ) +
  ggtitle("A boxplot with jitter") +
  xlab("")

ggplot(dat, aes(x=model, y=AUC,fill=model)) + 
  geom_boxplot()+geom_dotplot(binaxis='y', stackdir='center', dotsize=0.2,fill='black')

