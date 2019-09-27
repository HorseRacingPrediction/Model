
# basic settings
rm(list=ls())
options("scipen"=99, "digits"=5, stringAsFactors=F)

# set the directory which has the data and the prediction dataframe
setwd("F:/dropbox/Dropbox/MSBD")

library(dplyr)

# read the saved result dataframe from your prediction (rds format)
data <- readRDS("data_winpla.rds")

# check the dataframe (!!! need to include the 4 columns: 'winprob', 'plaprob', 'winstake', 'plastake')
data %>% print(n=10, width=Inf)
summary(data)

# compute the return rate of odds
data.returnrate <- data %>% 
  group_by(rdate, rid) %>%
  summarise( place.returnrate = 3/sum(1/place), win.returnrate = 1/sum(1/win)) %>%
  ungroup

# return rate of odds of win & place (around 82%)
mean(data.returnrate$place.returnrate, na.rm = T)
mean(data.returnrate$win.returnrate, na.rm = T)


###### 1 & 2. compute average RMSE of win prediction & place prediction over games
data.sum <- data %>% 
  group_by(rdate, rid) %>% 
  mutate( win.rmse = (winprob-ind_win)**2,
          pla.rmse = (plaprob-ind_pla)**2 ) %>%
  summarise( wrmse.pg = sqrt(mean(win.rmse)), prmse.pg = sqrt(mean(pla.rmse)) ) %>%
  ungroup %>%
  print(n=20, width=Inf)

# average RMSE of win & place of games
avg.RMSEwin <- mean(data.sum$wrmse.pg, na.rm = T)
avg.RMSEpla <- mean(data.sum$prmse.pg, na.rm = T)


###### 3. compute and summarize return by REAL odds
data <- data %>% 
  mutate( win.ret = winstake * (ind_win*win - 1),
          pla.ret = plastake * (ind_pla*place - 1), 
          plastake0 = ifelse(is.na(plastake), 0, plastake),
          winstake0 = ifelse(is.na(winstake), 0, winstake),
          total.stake = winstake0 + plastake0 ) %>%
  print(n=20, width=Inf)

data.ret <- data %>% 
  group_by(rdate, rid) %>% 
  # sum return and betting ratio per game
  summarise( ret.pg = sum(win.ret,na.rm = T) + sum(pla.ret,na.rm = T), 
             ratio.pg = sum(winstake,na.rm = T) + sum(plastake,na.rm = T) ) %>%
  ungroup %>%
  # compute the cumulative wealth
  mutate( cum.wealth = cumprod(1+ret.pg) ) %>%
  # compute betting amount per game
  mutate( cost.pg = ratio.pg * cum.wealth/(1+ret.pg) ) %>%
  print(n=20, width=Inf)

# save some statistics
## define 1 function to compute values with NA
mean.narm <- function(x) return(mean(x[!is.na(x)]))

# cumulative wealth
wealth.ro <- data.ret$cum.wealth
# final wealth
finalwealth.ro <- rev(wealth.ro)[1]
# total profit
totalprofit.ro <- finalwealth.ro - 1
# mean return per dollar
meanret.ro <- (totalprofit.ro)/sum(data.ret$cost.pg)
# number of betting games
no.games <- sum(data.ret$ratio.pg != 0)
# number of betting horses
no.horses <- sum(data$total.stake != 0)

###### 4. compute and summarize return by FAIR odds 
###### (Actually, the following steps can be combined with the part 3 cause they are similar, I just split into 2 parts here)
data <- data %>% 
  # amplify the real odds by dividing the return rate to get 'fair' odds
  mutate( win.ret2 = winstake * (ind_win*win/0.82 - 1),
          pla.ret2 = plastake * (ind_pla*place/0.82 - 1)) %>%
  print(n=20, width=Inf)

data.ret <- data %>% 
  group_by(rdate, rid) %>% 
  # sum return and betting ratio per game
  summarise( ret.pg2 = sum(win.ret2,na.rm = T) + sum(pla.ret2,na.rm = T),
             ratio.pg = sum(winstake,na.rm = T) + sum(plastake,na.rm = T) ) %>%
  ungroup %>%
  # compute the cumulative wealth
  mutate( cum.wealth2 = cumprod(1+ret.pg2) ) %>%
  # compute betting amount per game
  mutate( cost.pg2 = ratio.pg * cum.wealth2/(1+ret.pg2) ) %>%
  print(n=20, width=Inf)

# cumulative wealth
wealth.fo <- data.ret$cum.wealth2
# final wealth
finalwealth.fo <- rev(wealth.fo)[1]
# total profit
totalprofit.fo <- finalwealth.fo - 1
# mean return per dollar
meanret.fo <- (totalprofit.fo)/sum(data.ret$cost.pg2)


######################## plot wealth changes per games
plot(log(wealth.ro), type="l", ylab= "Cumulative Wealth", xlab="Games", col='blue')
lines(log(wealth.fo), type="l", col="orange", lty=2) # dashed line
grid()
legend("bottomleft", legend=c("Real Odds", "Fair Odds"), col=c('blue','orange'), lty=1:2, cex=0.8)


######################## combine summarized results
output <- c(avg.RMSEwin, avg.RMSEpla, 
            meanret.ro, totalprofit.ro, finalwealth.ro, 
            meanret.fo, totalprofit.fo, finalwealth.fo, 
            no.horses, no.games)

names(output) <- c('AverageRMSEwin','AverageRMSEpalce',
                   'MeanRetPerDollar(Real odds)','TotalProfit(Real odds)','FinalWealth(Real odds)',
                   'MeanRetPerDollar(Fair odds)','TotalProfit(Fair odds)','FinalWealth(Fair odds)',
                   'No.Horses','No.Games')
print(output)

# save the backtesting summary
write.csv(output, "backtesting_summary_R.csv")

