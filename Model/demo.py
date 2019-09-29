# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:52:58 2019

@author: Roger
"""
# =============================================================================
# I. Here you can
# 1. import necessary python packages for your prediction
# 2. load your own files containing trained models, engineered features, extra data etc. for prediction
# 3. set some global constants

# Note:
# 1. You should put your used files in the same folder as this python file
# 2. When load files, ALWAYS use relative path such as "MSBD/model1.pickle"
#    DO NOT use absolute path such as "C:/Users/Peter/Documents/project/MSBD/model1.pickle"

# =============================================================================
import os
import numpy as np
import pandas as pd


# =============================================================================
# II. Here are your predictions

# Note:
# 1. Need to add necessary comments to help us understand your program. 
#    MUST give the description of inputs and outputs of functions as the following example. 

# 2. The prediction should OUTPUT the data dataframe including 4 new columns (must saved in following column names!)
#   1) winprob: winning probabilities of horses
#   2) plaprob: top-three probabilities of horses
#   3) winstake: betting ratios of the bankroll on horses to be winners
#   4) plastake: betting ratios of the bankroll on horses to finish within top three places


# Here you should explain the idea of your predictions briefly in the form of Python comment.
# You can also attach related files such as a document & image & table in your team folder to show your idea

# The idea of this sample prediction:
# 1) make use of rating (column rating) of horses to predict winning probabilities of them 
# 2) then use the Plackett–Luce model to transfer winning probabilities to top-three probabilities
# 3) fix a stake and bet by finding merits based on odds 5 minutes before the start of matches

# =============================================================================

# the function to be applied on the dataframe to transfer rating value to winning probability
def rating2wp(data, df):
    '''
    Input: 
        data: the data dataframe
        df: the data dataframe (a trick of apply function)
    Output:
        wp: winning probability     
    '''
    rdate = data['rdate']
    rid = data['rid']
    #print(rdate, rid)
    ratingsum = df.loc[(df['rdate']==rdate) & (df['rid']==rid), 'rating'].sum()
    wp = data['rating']/ratingsum
    return wp

# the function to apply the function rating2wp on dataframe to get a vector of winning probability
def WinProb(datadf):
    '''
    Input: 
        datadf: the data dataframe
    Output:
        wp: a series of winning probabilities     
    '''
    wp = datadf.apply(rating2wp, axis=1, df=datadf)
    return wp

# get probability of 2nd place given winner probability    
def Place2nd(wp):
    '''
    Input: 
        wp: an array of winning probabilities
    Output:
        p2s: a list of probabilities of 2nd place     
    '''
    p2s = []
    for k, w in enumerate(wp):
        p2 = 0
        # due to Luce model, choose the 1st from the rest
        for ww in np.delete(wp, k):
            p2 += ww * w/(1-ww)
        p2s.append(p2)
    return p2s

# get probability of 3rd place given winner probability      
def Place3rd(wp):
    '''
    Input: 
        wp: an array of winning probabilities
    Output:
        p3s: a list of probabilities of 3rd place     
    '''
    p3s = []
    for k,w in enumerate(wp):
        p3 = 0
        wpx = np.delete(wp, k)
        # choose the 1st 
        for i,x in enumerate(wpx):
            # then choose the 2nd
            for y in np.delete(wpx, i):
                p3 += x * y * w/((1-x)*(1-x-y))
        p3s.append(p3)
    return p3s

# get probability of place (top3) given winner probability    
def WinP2PlaP(datawp, wpcol):
    '''
    Input: 
        datawp: the dataframe with column of winning probability
        wpcol: colunm name of the winning probability of datawp
    Output:
        top3: an array of probabilities of top 3    
    '''
    p2nds = []
    p3rds = []
    for (rd, rid), group in datawp.groupby(['rdate', 'rid']):
        wp = group[wpcol].values
        p2nds += Place2nd(wp)
        p3rds += Place3rd(wp)
    
    top3 = datawp[wpcol].values + np.array(p2nds) + np.array(p3rds)
    return top3

if __name__ == '__main__':
    ### read data
    # os.chdir('/Users/Roger/Dropbox/MSBD')

    # data = pd.read_csv('HR200709to201901.csv')
    data = pd.read_csv('../Data/HR200709to201901.csv')
    # infer the right date format
    data['rdate'] = pd.to_datetime(data['rdate'], infer_datetime_format=True)

    ### get the winning probabilities and top 3 probabilities (may take some time)
    print("Getting winning probabilities...")
    data['winprob'] = WinProb(datadf=data)
    # tansfer the winning probabilities to top 3 probabilities
    print("Getting place probabilities...")
    data['plaprob'] = WinP2PlaP(data, wpcol='winprob')

    ### choose a fixed ratio of bankroll and merit threshold to get betting stake vectors of win and place
    ## you should control the sum of betting ratios per week is less than 1, otherwise you may end up bankrupting!
    ## Higher ratio means bigger risk
    fixratio = 1/10000
    mthresh = 9
    print("Getting win stake...")
    data['winstake'] = fixratio * (data['winprob'] * data['win_t5'] > mthresh)
    print("Getting place stake...")
    data['plastake'] = fixratio * (data['plaprob'] * data['place_t5'] > mthresh)

    data.to_csv('test.csv')











