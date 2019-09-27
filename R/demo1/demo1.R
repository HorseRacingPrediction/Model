# =============================================================================
# I. Here you can
# 1. import necessary R packages for your prediction
# 2. load your own files containing trained models, engineered features, extra data etc. for prediction
# 3. set some global constants

# Note:
# 1. You should put your used files in the same folder as this R file
# 2. When load files, ALWAYS use relative path such as load("model1.RData")
#    DO NOT use absolute path such as load("C:/Users/Peter/Documents/project/MSBD/model1.RData")
# =============================================================================

library(dplyr)

# =============================================================================
# II. Here are your predictions

# Note:
# 1. Need to add necessary comments to help us understand your program. 
#    MUST give the description of inputs and outputs of functions as the following example. 

# 2. The prediction should OUTPUT the data dataframe with 4 new columns (must saved in following column names!)
#   1) winprob: winning probabilities of horses
#   2) plaprob: top-three probabilities of horses
#   3) winstake: betting ratios of the bankroll on horses to be winners
#   4) plastake: betting ratios of the bankroll on horses to finish within top three places


# Here you should explain the idea of your predictions briefly in the form of R comment.
# You can also attach related files such as a document & image & table in your team folder to show your idea

# The idea of this sample prediction:
# 1) make use of rating (column rating) of horses to predict winning probabilities of them 
# 2) then use the Plackett–Luce model to transfer winning probabilities to top-three probabilities
# 3) fix a stake and bet by finding merits based on odds 5 minutes before the start of matches

# =============================================================================

# basic settings
rm(list=ls())
options("scipen"=99, "digits"=5, stringAsFactors=F)

# set the directory which has the data
#setwd("F:/dropbox/Dropbox/MSBD")
setwd("/Users/Roger/Dropbox/MSBD")

data <- read.csv("HR200709to201901.csv")
head(data)
data$rdate <- as.Date(data$rdate)
summary(data)

### get the winning probabilities and top 3 probabilities
print("Getting winning probabilities...")

# use dplyr to transfer rating values to winning probabilities and name it as 'wp'
datawp <- data %>% 
  group_by(rdate, rid) %>% 
  mutate( wp = rating/sum(rating, na.rm = T) ) %>%
  ungroup
# print 10 rows to check results
datawp %>% print(n=10, width=Inf)

print("Getting place probabilities...")

# use dplyr to get probability of 2nd place given the winner probability and name it as 'p2nd'
datawp <- datawp %>% 
  group_by(rdate, rid) %>% 
  mutate(p2nd1 = wp/(1-wp)) %>%
  mutate(p2nd = wp*(sum(p2nd1) - p2nd1)) %>%
  ungroup %>%
  print(n=10, width=Inf)

# the function to get a vector of probabilities of 3rd place given the vector of winner probabilities     
Place3rd <- function(wp){
  p3s = c()
  for (k in 1:length(wp)){
    p3 = 0
    wpx = wp[-k]
    ## the following computation is due to Luce model
    # choose the 1st 
    for (i in 1:length(wpx)){
      # then choose the 2nd
      x = wpx[i]
      for (y in wpx[-i]){
        p3 = p3 + x * y * wp[k]/((1-x)*(1-x-y))
      }
    }
    p3s = c(p3s, p3)
  }
  return(p3s)
}

# use the following loop to get a vector of probabilities of 3rd place from wp column
rdates <- unique(datawp$rdate)
p3col <- c()
for (rd in rdates){
  # data of the same date
  rd.data <- datawp[datawp$rdate==rd, ]
  rids <- unique(rd.data$rid)
  for (ri in rids){
    # data of the same race 
    wpi <- rd.data[rd.data$rid==ri, ]$wp
    p3col <- c(p3col, Place3rd(wpi))
  }
}

# add a column 'p3rd' to the dataframe
datawp <- datawp %>% 
  mutate(p3rd = p3col) %>% 
  print(n=10, width=Inf)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
# sum 3 probabilities together to get the probability of top 3
datawp <- datawp %>% 
  mutate(plaprob = wp+p2nd+p3rd) %>% 
  print(n=10, width=Inf)


### choose a fixed ratio and merit threshold to get betting stake vectors of win and place
## you should control the sum of betting ratios per week is less than 1, otherwise you may end up bankrupting!
## Higher ratio means bigger risk
fixratio = 1/10000
mthresh = 9

print("Getting win stake...")

# rename wp column as winprob and find the betting stakes of win
datawp <- datawp %>% 
  rename(winprob = wp) %>%
  mutate(winstake = fixratio*(winprob*win_t5 > mthresh))
  
datawp %>% print(n=20, width=Inf)
summary(datawp$winstake)

print("Getting place stake...")

# rename wp column as winprob and find the betting stakes of win
datawp <- datawp %>% 
  mutate(plastake = fixratio*(plaprob*place_t5 > mthresh))

datawp %>% print(n=20, width=Inf)
summary(datawp$plastake)

# save the dataframe inlcuding 4 new columns as rds file
saveRDS(datawp, 'data_winpla.rds')



