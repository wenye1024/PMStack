'''
Created on May 13, 2019

@author: yewen
'''

import pandas as pd
import numpy as np




def macd(df, n_fast=12, n_slow=26, n_signal=9):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    
    EMAfast = df.ewm(span=n_fast, min_periods=n_slow).mean()
    EMAslow = df.ewm(span=n_slow, min_periods=n_slow).mean()
    MACD = EMAfast - EMAslow
    MACDsign = MACD.ewm(span=n_signal, min_periods=n_signal).mean()
    MACDdiff = MACD - MACDsign
    
    return MACD, MACDsign, MACDdiff

def action_macd(df, n_fast=12, n_slow=26, n_signal=9):
    MACD, MACDsign, MACDdiff = macd(df.iloc[-(n_slow + n_signal + 6):], n_fast, n_slow, n_signal)
    
    #print(MACDdiff)    
    a = (MACDdiff.iloc[-1]>0) & (MACDdiff.iloc[-2]<=0)
    b = (MACDdiff.iloc[-1]>0) & (MACDdiff.iloc[-2]>0) & (MACDdiff.iloc[-3]<=0)
    c = (MACDdiff.iloc[-1]>0) & (MACDdiff.iloc[-2]>0) & (MACDdiff.iloc[-3]>0) & (MACDdiff.iloc[-4]<=0)

    buy = a | b | c

    a = (MACDdiff.iloc[-1]<0) & (MACDdiff.iloc[-2]>=0)
    b = (MACDdiff.iloc[-1]<0) & (MACDdiff.iloc[-2]<0) & (MACDdiff.iloc[-3]>=0)
    c = (MACDdiff.iloc[-1]<0) & (MACDdiff.iloc[-2]<0) & (MACDdiff.iloc[-3]<0) & (MACDdiff.iloc[-4]>=0)

    sell = a | b | c
    
    return buy, sell



if __name__ == '__main__':
    pass