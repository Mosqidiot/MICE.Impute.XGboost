# Author: Haozhen Wu
#         Fangzhou Hu

library(Hmisc)
library(xgboost)
library(Matrix)
train.x = data.matrix(homesite_numr_imp[1:5000,])
train.y = as.numeric(train_y[1:5000,])
train.x =as.data.frame(train.x)
train.x$factor = factor(c(rep("1",1000),rep("2",2000),rep("3",2000)))
train.y = as.data.frame(train.y)
train.x.original = train.x
train.y.original = train.y
set.seed(1128)
range = 1:dim(train.x)[2]
for ( i in range[-c(4,26,89,106)]){
  index = sample(dim(train.x)[1], runif(1,0.01,0.1)*dim(train.x)[1])
  train.x[index,i] = NA
}
summary(train.x)

class(train.x)

na_col_index = which(sapply(train.x, function(x) length(which(is.na(x)))>0))

na_matrix = apply(train.x,2, function(x) as.numeric(is.na(x)))
na_size_col = apply(na_matrix,2, sum)
na_size_order = order(na_size_col,decreasing = T)
na_size_order = na_size_order[which(na_size_order %in% which(na_size_col>0))]
col_class = sapply(train.x, function(x) as.numeric(is.numeric(x)))
col.class.impute = function(x) {
  if (class(x) == "numeric") {
    return(as.numeric(impute(x,fun = mean) ))
  }
  else {
    return(as.factor(impute(x,fun = mode)))
  }
  
}
imp_mean_mode = sapply(train.x, col.class.impute)
imp_mean_mode = as.data.frame(imp_mean_mode)
for (i in 1:length(col_class)){
  if (col_class[i] == 0) {
    imp_mean_mode[,i] = as.factor(imp_mean_mode[,i])
  }
}


imp1 = imp_mean_mode

for ( imp_round in 1:3){ 
  # start for loop
  time_start=proc.time()
  number = 1
  for ( j in na_size_order){
    imp1_temp = imp1
    imp1_temp_x = imp1_temp[,-j]
    imp1_temp_y = imp1_temp[,j] # might need to convert to numeric
    #imp1_temp_x = sparse.model.matrix(~., data = imp1_temp_x)
    
    
    train_imp_x = imp1_temp_x[-which(na_matrix[,j]==1),]
    train_imp_y = imp1_temp_y[-which(na_matrix[,j]==1)]
    
    # sample 1% from training set and treat them as validation set.
    vali_index = sample(dim(train_imp_x)[1],round(0.01*dim(train_imp_x)[1]))
    train_imp_x_validation = train_imp_x[vali_index,]
    train_imp_y_validation = train_imp_y[vali_index]
    train_imp_x = train_imp_x[-vali_index,]
    train_imp_y = train_imp_y[-vali_index]
    pred_imp_x = imp1_temp_x[which(na_matrix[,j]==1),]
    #train_imp_x = as.matrix(train_imp_x)
    #train_imp_y = as.matrix(train_imp_y)
    dtrain_imp <- xgb.DMatrix(data = train_imp_x, label= train_imp_y)
    dtest_imp <- xgb.DMatrix(data = train_imp_x_validation, label= train_imp_y_validation)
    watchlist_imp <- list(train=dtrain_imp, test=dtest_imp)
    
    if (col_class[j] == 1) {
      cat("imp_round:",imp_round,"Number:", number,"\n")  
      cat("Variable name:", names(col_class)[j],"\n")
      model_imp <- xgb.train(data=dtrain_imp, max.depth=200, eta=0.03, nround = 1, 
                             eval.metric = "rmse",verbose = 1,watchlist=watchlist_imp, 
                             maximize = FALSE,
                             colsample_bytree = 0.2, 
                             objective = "reg:linear",
                             num_parallel_tree = 500)
      result = predict(model_imp,pred_imp_x)
      imp1[which(na_matrix[,j]==1),j] = result
    }
    else if (col_class[j] == 0){
      if ( length(unique(train.x[,j])) == 3) {
        cat("imp_round:",imp_round,"Number:", number,"\n")   
        cat("Variable name:", names(col_class)[j],"\n")
        model_imp <- xgb.train(data=dtrain_imp, max.depth=200, eta=0.03, nround = 1, 
                               eval.metric = "error",verbose = 1,watchlist=watchlist_imp, 
                               maximize = FALSE,
                               colsample_bytree = 0.2, 
                               objective = "binary:logistic",
                               num_parallel_tree = 500)
        result = predict(model_imp,pred_imp_x)
        imp1[which(na_matrix[,j]==1),j] = result
      } 
      else {
        cat("imp_round:",imp_round,"Number:", number,"\n")    
        cat("Variable name:", names(col_class)[j],"\n")
        num_of_class = length(unique(train.x[,j]))
        model_imp <- xgb.train(data=dtrain_imp, max.depth=200, eta=0.03, nround = 1, 
                               eval.metric = "merror",verbose = 1,watchlist=watchlist_imp, 
                               maximize = FALSE,
                               colsample_bytree = 0.2, 
                               objective = "multi:softmax", num_class = num_of_class,
                               num_parallel_tree = 500)
        result = predict(model_imp,pred_imp_x)
        imp1[which(na_matrix[,j]==1),j] = result
      }
      
      
    }
    number = number + 1
  }
  time = time_start-proc.time()
  time
}



sqrt(sum((imp1[,-c(249,250)]-train.x.original[,-c(249,250)])^2)/(dim(imp1)[1]*(dim(imp1)[2]-2)))
sqrt(sum((imp_mean_mode[,-c(249,250)]-train.x.original[,-c(249,250)])^2)/(dim(imp1)[1]*(dim(imp1)[2]-2)))
which(imp1[,250] != train.x.original[,250])
which(imp_mean_mode[,250] != train.x.original[,250])

head(cbind(imp1[,249],train.x.original[,249]),100)





###################### For real sample imputation ########################
library(Hmisc)
library(xgboost)
library(Matrix)
train.x = homesite_sel
train.y = train_y
#train.x =as.data.frame(train.x)
#train.x$factor = factor(c(rep("1",1000),rep("2",2000),rep("3",2000)))
#train.y = as.data.frame(train.y)
train.x.original = train.x
train.y.original = train.y
set.seed(1128)

#summary(train.x)

class(train.x)

na_col_index = which(sapply(train.x, function(x) length(which(is.na(x)))>0))

na_matrix = apply(train.x,2, function(x) as.numeric(is.na(x)))
na_size_col = apply(na_matrix,2, sum)
na_size_order = order(na_size_col,decreasing = T)
na_size_order = na_size_order[which(na_size_order %in% which(na_size_col>0))]
na_size_order = na_size_order[c(1:8)]
col_class = sapply(train.x, function(x) as.numeric(is.numeric(x)))
col.class.impute = function(x) {
  if (class(x) == "numeric") {
    return(as.numeric(impute(x,fun = mean) ))
  }
  else {
    return(as.factor(impute(x,fun = mode)))
  }
  
}
#imp_mean_mode = sapply(train.x, col.class.impute)
imp_mean_mode = cbind(homesite_numr_imp,homesite_factor_imp)
imp_mean_mode = as.data.frame(imp_mean_mode)
for (i in 1:length(col_class)){
  if (col_class[i] == 0) {
    imp_mean_mode[,i] = as.factor(imp_mean_mode[,i])
  }
}


imp1 = imp_mean_mode

sink('imputation1_out_1201.txt')
for ( imp_round in 1:3){ 
  # start for loop
  time_start=proc.time()
  number = 1
  for ( j in na_size_order){
    imp1_temp = imp1
    imp1_temp_x = imp1_temp[,-j]
    imp1_temp_y = imp1_temp[,j] # might need to convert to numeric
    #
    imp1_temp_y = as.matrix(imp1_temp_y)
    imp1_temp_x_numr = imp1_temp_x[, sapply(imp1_temp_x, is.numeric)]
    imp1_temp_x_factor = imp1_temp_x[, sapply(imp1_temp_x, is.factor)]
    
    #
    cat("Start preparing sparse dataset for imp_round:",imp_round,"number:",number,"\n")
    imp1_temp_x_factor = sparse.model.matrix(~., data = imp1_temp_x_factor)
    imp1_temp_x_numr = as.matrix(imp1_temp_x_numr)
    imp1_temp_x_total = cbind(imp1_temp_x_factor,imp1_temp_x_numr)
    cat("Finished\n")
    
    train_imp_x = imp1_temp_x_total[-which(na_matrix[,j]==1),]
    train_imp_y = imp1_temp_y[-which(na_matrix[,j]==1)]
    
    # sample 1% from training set and treat them as validation set.
    vali_index = sample(dim(train_imp_x)[1],round(0.01*dim(train_imp_x)[1]))
    train_imp_x_validation = train_imp_x[vali_index,]
    train_imp_y_validation = train_imp_y[vali_index]
    train_imp_x = train_imp_x[-vali_index,]
    train_imp_y = train_imp_y[-vali_index]
    pred_imp_x = imp1_temp_x_total[which(na_matrix[,j]==1),]
    
    dtrain_imp <- xgb.DMatrix(data = train_imp_x, label= train_imp_y)
    dtest_imp <- xgb.DMatrix(data = train_imp_x_validation, label= train_imp_y_validation)
    watchlist_imp <- list(train=dtrain_imp, test=dtest_imp)
    
    
    if (col_class[j] == 1) {
      cat("imp_round:",imp_round,"Number:", number,"\n")  
      cat("Variable name:", names(col_class)[j],"\n")
      model_imp <- xgb.train(data=dtrain_imp, max.depth=100, eta=0.03, nround = 1, 
                             eval.metric = "rmse",verbose = 1,watchlist=watchlist_imp, 
                             maximize = FALSE,
                             colsample_bytree = 0.2, 
                             objective = "reg:linear",
                             num_parallel_tree = 100)
      result = predict(model_imp,pred_imp_x)
      imp1[which(na_matrix[,j]==1),j] = result
    }
    else if (col_class[j] == 0){
      if ( length(unique(train.x[,j])) == 3) {
        cat("imp_round:",imp_round,"Number:", number,"\n")   
        cat("Variable name:", names(col_class)[j],"\n")
        model_imp <- xgb.train(data=dtrain_imp, max.depth=100, eta=0.03, nround = 1, 
                               eval.metric = "error",verbose = 1,watchlist=watchlist_imp, 
                               maximize = FALSE,
                               colsample_bytree = 0.2, 
                               objective = "binary:logistic",
                               num_parallel_tree = 100)
        result = predict(model_imp,pred_imp_x)
        imp1[which(na_matrix[,j]==1),j] = result
      } 
      else {
        cat("imp_round:",imp_round,"Number:", number,"\n")    
        cat("Variable name:", names(col_class)[j],"\n")
        num_of_class = length(unique(train.x[,j]))
        model_imp <- xgb.train(data=dtrain_imp, max.depth=100, eta=0.03, nround = 1, 
                               eval.metric = "merror",verbose = 1,watchlist=watchlist_imp, 
                               maximize = FALSE,
                               colsample_bytree = 0.2, 
                               objective = "multi:softmax", num_class = num_of_class,
                               num_parallel_tree = 100)
        result = predict(model_imp,pred_imp_x)
        imp1[which(na_matrix[,j]==1),j] = result
      }
      
      
    }
    number = number + 1
  }
  time = time_start-proc.time()
  print(time)
}
sink()
