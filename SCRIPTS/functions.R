#split data into samples and targets
split_sequence <- function(sequ, n_stpes) {
  X <- matrix(ncol=n_steps)
  y <- matrix(ncol=n_features)
  
  for(i in 1:length(sequ)) {
    #find the end of the pattern
    end_ix <- i + n_steps -1
    
    #check if it's beyond the sequence
    if (end_ix > length(sequ)-1) {
      break
    }
    
    #assing input and outputs
    seq_x <- sequ[i:end_ix]
    seq_y <- sequ[end_ix+1]
    
    X <- rbind(X,seq_x)
    y <- rbind(y,seq_y)
    
  }
  X<-na.omit(X)
  y<-na.omit(y)
  
  #return samples and targets
  list(X,y)
}


split_sequences <- function(sequ, n_stpes) {
  X <- matrix(ncol=n_steps)
  X2 <- matrix(ncol=n_steps)
  y <- matrix(ncol=1)
  
  for(i in 1:nrow(sequ)) {
    #find the end of the pattern
    end_ix <- i + n_steps -1
    
    #check if it's beyond the sequence
    if (end_ix > nrow(sequ)-1) {
      break
    }
    
    #print(sequ[i:end_ix,2])
    
    #assing input and outputs
    seq_x <- sequ[i:end_ix,1]
    seq_x2 <- sequ[i:end_ix,2]
    seq_y <- sequ[end_ix+1,1]
    
    
    
    X <- rbind(X,seq_x)
    X2 <- rbind(X2,seq_x2)
    y <- rbind(y,seq_y)
    
  }
  X<-na.omit(X)
  X2<-na.omit(X2)
  y<-na.omit(y)
  
  
  
  #return samples and targets
  list(X,X2,y)
}






get_equity_line <- function(dt, 
                            prices_var = "close", 
                            signal_var = "signal", 
                            normalize_to = NULL,
                            TC = NULL,
                            DFL = 1
) {
  prices <- dt[, prices_var]
  signal <- dt[, signal_var]
  ret <- (prices / xts::lag.xts(prices)) - 1
  names(ret) <- "ret"
  
  
  # calculating costs
  # costs <- -(abs(signal) * 2 * TC)
  costs <- -(abs(signal - xts::lag.xts(signal)) * 1 * 0.0005) 
  # costs[(signal - xts::lag.xts(signal)) == 0] <- 0
  costs[is.na((signal - xts::lag.xts(signal)))] <- 0
  
  #view(costs)
  
  # adding costs
  # eql <- max(-1, ret * as.numeric(lag.xts(signal)) + costs)
  eql <- DFL * ret * as.numeric(lag.xts(signal)) + costs
  eql[eql < -1] <- -1
  names(eql) <- "eql"
  eql[is.na(eql), ] <- 0
  eql <- zoo::na.locf(eql)
  eql <- cumprod(1 + eql)
  
  ## without costs
  #eql <- ret * as.numeric(lag.xts(signal))
  #names(eql) <- "eql"
  #eql[is.na(eql), ] <- 0
  #eql <- zoo::na.locf(eql)
  #eql <- cumprod(1 + eql)
  
  result <- merge(dt[, c(prices_var, signal_var)], eql, costs) 
  result[, 3] <- result[, 3] * as.numeric(result[1, 1])
  
  if (!is.null(normalize_to)) {
    result[, 3] <- result[, 3] / as.numeric(result[1, 1]) * normalize_to
  }
  
  return(result)
}


## ---------------------------------------------------------------------------------------------------------------------
# aRC
getARC <- function(x, scale = 365){
  nInt <- length(x)
  ARC  <- (x[length(x)] / x[1]) ^ (scale/nInt) - 1
  return(ARC)
}

# aSD
getASD <- function(x, scale = 365){
  r <- xts::diff.xts(x)/xts::lag.xts(x)
  annSdtDev <- sd(r, na.rm = T) * sqrt(scale)
  return(annSdtDev)
}

# MD
getMD <- function(x){
  r <- xts::diff.xts(x)/xts::lag.xts(x)
  return(suppressWarnings(PerformanceAnalytics::maxDrawdown(r)))
}

# IR
getIR <- function(x, scale = 365){
  getARC(x, scale)/getASD(x, scale)
}

# LD
getLD2 <- function(x, scale = 365) {
  currentMax <- x[1]
  LD         <- rep(NA, length(x))
  LD[1]      <- 0
  
  for (i in 2:length(x)) {
    if (x[i] >= currentMax) {
      currentMax <- x[i]
      LD[i]      <- 0
    } else {
      LD[i]      <- LD[i - 1] + 1
    }
  }
  
  LD <- LD / scale
  
  # return(round(LD, 2))
  return(LD)
}

# get all performance stats
getPerformanceStats <- function(x, scale = 365){
  
  aRC     <- getARC(x, scale)
  aSD     <- getASD(x, scale)
  MD      <- getMD(x)
  MLD     <- max(getLD2(x, scale))
  IR1     <- getIR(x, scale)
  IR2     <- IR1 * aRC * sign(aRC)/ MD
  IR3     <- aRC ^ 3 / (aSD * MD * MLD)
  nObs    <- length(x)
  result <- c(100 * aRC, 100 * aSD, 100 * MD, MLD,
              IR1, IR2, IR3, nObs)
  names(result) <- c('aRC', "aSD", "MD", "MLD",
                     "IR*", "IR**", "IR***", "nObs")
  return(result)
}


# transform xts objects to tibbles ==========================
xts2tbl <- function(x, closeVar = "BTC") {
  
  ind <- index(x)
  
  output <-
    x %>%
    as.data.frame(x) %>%
    as_tibble() %>%
    mutate(datetime = ind) %>%
    rename(
      # !!closeVar := close,
      signalLS = signal, 
      eqlLS = eql
    ) %>%
    select(datetime, everything())
  
  return(output)
}