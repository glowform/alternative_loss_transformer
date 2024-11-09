#disable gpu devices
#Sys.setenv("CUDA_VISIBLE_DEVICES" = -1)

Sys.setenv(TZ='America/New_York')


library(keras)
library(tensorflow)
#set seed to get reproducible results, seed wyłącza gpu
#set_random_seed(23)
library(tidyverse)
library(dygraphs)
library(lubridate)
library(here)
library(xts)
library(scales)
library(feather)
library(TTR)
#library(kerastuneR)

source(here("SCRIPTS/functions.R"))




#loading 1day snp data
dt_1d <-
  read_csv(here("DATA/eth.csv"))


dt_1d <- dt_1d[!dt_1d$Close == "null", ] 

dt_1d[756,]

dt_1d$Close <- as.numeric(dt_1d$Close)

dt <-
  dt_1d %>%
  select(5) %>%
  mutate(Close = Close/xts::lag.xts(Close) - 1)


dt_l <- diff(log(dt_1d$Close))


data <- data.matrix(dt)
data <- na.omit(data)
rownames(data) <- NULL

summary(data)





#define network hyperparams
filters <- 256
kernel_size <- 3
pool_size = 3
hidden_dims <- 250
dropout_rate <- 0.0 #default 0.0
batch_size <- 5000#default 5000 = test size
l2_reg <-   0.000001
momentum <-0.1

#define sequence length
n_steps <-3

#define number of features (input variables)
n_features <- 1



#custom loss function defintion
made <-function(y_true, y_pred) {
  
  loss <- (-1. * tf$sign(y_true * y_pred) * (tf$abs(y_true)))
  
  
  tf$reduce_mean(loss)
}
keras.losses.made = made

#define weigths initializer
initializer = initializer_glorot_uniform(seed = 0)




#reshape data to match expected layer input shape
input_shape = list(n_steps, n_features)



#define lengths of train and test sets
n_train = 365 # DLA BTC POWINNO BYC 365
n_test = 365
n_records = (length(data))

#create empty signals vector
signal <- vector()
eval_list <- vector()
preds_list <- vector()


#custom act function, relu inverse

relu_inv <-function(x) {
  #zrobić to tensorowo
  tf$where(x < 0 , 
           x,
           0)
}

keras.activations.relu_inv = relu_inv


#main rolling loop
for (i in seq(n_train, n_records, n_test)){
  
  opt = optimizer_adam(lr = 0.5) 
  
  
  #crate train set
  #train <- matrix(data[((i-n_train)+1):i])
  #set beginning to 1 for expanding window
  #train <- matrix(data[1:i])
  
  #stop after x years
  train <- matrix(data[(pmax(i-(n_train*4),0)+1):i])
  
  cat("train length: ", length(train), "\n")
  
  #calculate max and min values and divide by greater of them (normalize)
  #m <- if (abs(max(train))>abs(min(train))) max(train) else min(train)
  #train <- train/m
  
  #create test set and divide by train min/max value
  test <- (matrix(data[((i+1)-n_steps):((i+n_test))])) 
  test <- na.omit(test)
  #test <- test/m
  
  
  splt <- split_sequence(train,n_steps)
  X <- splt[[1]]
  y <- splt[[2]]
  
  #reshape from [samples, timesteps] to [samples, timesteps, features]
  X <- array(X, dim = c(dim(X)[1], dim(X)[2],n_features))
  
  
  
  #early stopping to stop training when it's not improving and model checkpoint to save the best model
  es <- callback_early_stopping(monitor='val_loss', mode='min', patience=6, verbose=1, restore_best_weights = TRUE)
  mc <- callback_model_checkpoint(here(paste("best_model.h5", sep="")), monitor='val_loss', mode='min', save_best_only=TRUE, verbose=1)
  
  
  with_custom_object_scope(c(i_r = relu_inv), {
    
    
    #initialize model
    model <- keras_model_sequential()
    
    #create the model
    model %>%
      
      
      
      layer_lstm(
        units = 512,
        recurrent_activation = "sigmoid",
        return_sequences = TRUE,
        #stateful = TRUE,
        kernel_regularizer = regularizer_l2(l = l2_reg),
        #activity_regularizer=regularizer_l2(0.0001),
        #recurrent_dropout = (0.02),
        kernel_initializer=initializer,
        
      ) %>%
      layer_dropout(rate = dropout_rate) %>%
      
      layer_lstm(
        units = 256,
        recurrent_activation = "sigmoid",
        return_sequences = TRUE,
        #stateful = TRUE,
        kernel_regularizer = regularizer_l2(l = l2_reg),
        #activity_regularizer=regularizer_l2(0.0001),
        #recurrent_dropout = (0.02),
        kernel_initializer=initializer,
        
      ) %>%
      layer_dropout(rate = dropout_rate) %>%
      
      
      layer_lstm(
        units = 128,
        recurrent_activation = "sigmoid",
        #stateful = TRUE,
        kernel_regularizer = regularizer_l2(l = l2_reg),
        kernel_initializer=initializer,
        
        #recurrent_dropout = (0.02),
        #activity_regularizer=regularizer_l2(0.0001),
        input_shape = input_shape
      ) %>%
      layer_dropout(rate = dropout_rate) %>%
      
      
      
      
      layer_dense(1, activation = "relu") 
    #layer_dense(1, activation = "linear")
    # layer_dense(1) %>%
    # layer_activation(activation = c("i_r"))
  
    
    #compile the model
    model %>% compile(
      loss = made,
      optimizer = opt,
    )
    
    
    print("trainin the model...")

    
    hist_RNN <- model%>%
      fit(
        X,
        y,
        batch_size = batch_size,
        validation_split = 0.333,
        epochs = 300,
        verbose = 2,
        shuffle = FALSE, #org było bez tego
        callbacks = list(mc)
      )
    
    
    
    
    #load savel model from mc
    #since we added restore_best_weights = TRUE model = saved_model
    saved_model = load_model_hdf5(here('best_model.h5'), compile=FALSE)
    saved_model %>% compile(
      optimizer=opt,
      loss = made,
    )
    
    
    
    #train and validate the model on train set 
    
    
    
    #split test data into samples and targets
    splt_t <- split_sequence(test,n_steps)
    Xtest <- splt_t[[1]]
    ytest <- splt_t[[2]]
    
    #reshape test data
    Xtest <- array(Xtest, dim = c(dim(Xtest)[1], dim(Xtest)[2],n_features))
    
    
    print("making predictions...")
    #make predictions on test set
    
    preds <- saved_model %>% predict(
      Xtest,
      verbose = 2
    )
    
    ##add preds to list
    preds_list <- c(preds_list, preds)
    
    
    #evaluate model
    
    print("evaluating...")
    eval <- saved_model %>% evaluate(
      Xtest,
      ytest
    )
    
    print(eval)
    
    
    eval_list <- c(eval_list, eval)
    
    
    #generate signal
    
    print("generating signals...")
    
    #print(preds)
    
    for (j in 1:length(preds)){
      if(preds[j] >0.000000) {
        signal <- c(signal,1)
      } else if (preds[j]<0.000000){
        signal <- c(signal,-1)
      } else {
        signal <- c(signal,0)
      }
    }
    
    cat("test length: ",length(test), "\n")
    cat("signal length: ",length(signal), "\n")
    
    
    #write_feather(as.data.frame(signal), here("EXPORTS/signal_snp_1d.feather"))
    
    #Sys.sleep(5)
  }) #end custom scope
  
  
  
  
}


print(eval_list)


#made dla buy and hold
made(as.numeric(tail(data, length(preds_list))),as.numeric(tail(data, length(preds_list))))
#made dla sieci
made(as.numeric(tail(data, length(preds_list))),preds_list)




#replace signal values to test different strategies
signal_bkp <- signal
#signal <- signal_bkp

#save for export
rets_pr <- as.data.frame(preds_list)
rets_pr$signal <- signal



#LONG ONLY
signal_l <-vector()
for (j in 1:length(signal)){
  
  if(signal[j] == 1 ) {
    signal_l[j] <- 1
  } else if (signal[j] == -1){
    signal_l[j] <- 0
  } else {
    signal_l[j] <- 0
  }
  
}
rets_pr$signal_long <- signal_l



#SHORT ONLY
signal_s <-vector()
for (j in 1:length(signal)){
  
  if(signal[j] == 1 ) {
    signal_s[j] <- 0
  } else if (signal[j] == -1){
    signal_s[j] <- -1
  } else {
    signal_s[j] <- 0
  }
  
}
rets_pr$signal_short <- signal_s




dt_1d <-
  read.csv(here("DATA/eth.csv")) %>%
  select(1,5)

dt_1d <- dt_1d[(n_train+1):(n_train + length(signal)),]
dt_1d$signal <- signal
dt_1d$signal_l <- signal_l
dt_1d$signal_s <- signal_s
dt_1d <-xts(dt_1d,as.Date(dt_1d$Date))
dt_1d$Date <- NULL
storage.mode(dt_1d) <- "numeric"
names(dt_1d) <- c("close","signal","signal_l", "signal_s")


rets_pr$Date <- index(dt_1d) 



eql1 <-
  dt_1d %>% 
  get_equity_line(prices_var = "close", 
                  signal_var = "signal")
eql1 %>% head()


eql2 <-
  dt_1d %>% 
  get_equity_line(prices_var = "close", 
                  signal_var = "signal_l")
eql2 %>% head()


eql3 <-
  dt_1d %>% 
  get_equity_line(prices_var = "close", 
                  signal_var = "signal_s")
eql3 %>% head()



#draw eqline with signal, same start
eql1 %>%
  dygraph() %>%
  dyRangeSelector(height = 40) %>%
  dySeries("signal", axis = "y2", strokeWidth=0) %>%
  dyAxis("y2", label = "signal", independentTicks = TRUE) %>%
  dyAxis("y", label = "eql")


#stats LS
st1 <-
  eql1 %>%
  .[, "close"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 365)
st2 <-
  eql1 %>%
  .[, "eql"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 365)
st <- bind_rows(st1, st2)
rownames(st) <- c("btc", "eql")
st %>%
  kableExtra::kable(
    caption = "Performance stats based on daily intervals",
    row.names = T, digits = 2, escape = F
  ) %>%
  kableExtra::kable_styling(
    bootstrap_options = c("striped", "hover", "condensed", "responsive" ),
    full_width = T, font_size = 14
  )



#draw eqline with signal_l, same start
eql2 %>%
  dygraph() %>%
  dyRangeSelector(height = 40) %>%
  dySeries("signal_l", axis = "y2", strokeWidth=0) %>%
  dyAxis("y2", label = "signal_l", independentTicks = TRUE) %>%
  dyAxis("y", label = "eql")


#stats LO
st1 <-
  eql2 %>%
  .[, "close"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 365)
st2 <-
  eql2 %>%
  .[, "eql"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 365)
st <- bind_rows(st1, st2)
rownames(st) <- c("btc", "eql")
st %>%
  kableExtra::kable(
    caption = "Performance stats based on daily intervals",
    row.names = T, digits = 2, escape = F
  ) %>%
  kableExtra::kable_styling(
    bootstrap_options = c("striped", "hover", "condensed", "responsive" ),
    full_width = T, font_size = 14
  )


#draw eqline with signal_s, same start
eql3 %>%
  dygraph() %>%
  dyRangeSelector(height = 40) %>%
  dySeries("signal_s", axis = "y2", strokeWidth=0) %>%
  dyAxis("y2", label = "signal_l", independentTicks = TRUE) %>%
  dyAxis("y", label = "eql")


#stats LS
st1 <-
  eql3 %>%
  .[, "close"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 365)
st2 <-
  eql3 %>%
  .[, "eql"] %>%
  as.numeric() %>%
  getPerformanceStats(scale = 365)
st <- bind_rows(st1, st2)
rownames(st) <- c("btc", "eql")
st %>%
  kableExtra::kable(
    caption = "Performance stats based on daily intervals",
    row.names = T, digits = 2, escape = F
  ) %>%
  kableExtra::kable_styling(
    bootstrap_options = c("striped", "hover", "condensed", "responsive" ),
    full_width = T, font_size = 14
  )



rets_pr$eql_bh <- as.numeric(eql1$close)
rets_pr$eql_ls <- as.numeric(eql1$eql)

rets_pr$eql_lo <- as.numeric(eql2$eql)




write_feather(as.data.frame(rets_pr), here("EXPORTS/eth_madl_lstm_train1y_lo.feather"))


#rets_pr <- arrow::read_feather(here('EXPORTS/from_2015/eth_madl_lstm_train1y_lo.feather'))


