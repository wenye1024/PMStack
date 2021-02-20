'''
Created on Sep 30, 2018

@author: yewen
'''


import os
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from scipy import optimize
from _ast import operator
import Equity_port.historical_price as historical_price
import Equity_port.finance as finance
import Crypto.technical_indicators as ti


class Alert:
    def __init__(self, today):
        xl = pd.ExcelFile('watchlist.xlsx')
        self.watchlist = xl.parse('watchlist')
        self.watchlist['Index'] = self.watchlist['Ticker']
        self.watchlist.set_index('Index', inplace = True)
        print(self.watchlist)

        self.today = today
        self.last_year = pd.Series(pd.date_range(end=self.today, periods=365)) #total calendar days
    
    def get_price_history (self, history = None, workbook='Historical price.xlsx', sheet='Historical price'):        
        
        if history is None:
            r = historical_price.get_price_history(workbook=workbook, sheet=sheet)
        else:
            r = history
        #self.forex = r['forex']
        #self.company_name = r['company_name']
        #self.equity_currency = r['equity_currency']
        self.equity_prices_local = r['equity_price_local'][self.watchlist.index] 
        #self.equity_prices_USD = r['equity_price_USD']
        self.equity_daily_return_local = r['equity_daily_return_local'][self.watchlist.index]
        #self.equity_daily_return_USD = r['equity_daily_return_USD']



    def sharp(self):
        
        df = pd.DataFrame(index=self.last_year, columns=['Sharp10','SharpSigma'])
        df = df[df.index.dayofweek < 5]
            
        for ticker in self.watchlist.index:
            for end_date in df.index:
                #start_date = end_date - pd.Timedelta(days = 6)
                #df.loc[end_date,'Sharp5'] = finance.sharp(equity_daily_return_local.loc[start_date:end_date, ticker])
                start_date = end_date - pd.Timedelta(days = 13)
                df.loc[end_date,'Sharp10'] = finance.sharp(self.equity_daily_return_local.loc[start_date:end_date, ticker])
            
            mean = np.mean(df['Sharp10'])
            std = np.std(df['Sharp10'])
            sharp10 = df.loc[self.today,'Sharp10']
            sigma = (sharp10 - mean)/std
            self.watchlist.loc[ticker,'Sharp10Sigma'] = sigma
    
    def price_action(self):
        dp = self.equity_prices_local.loc[self.today - pd.Timedelta(days = 365):self.today]
        dmin = pd.DataFrame(dp.min())
        dmin.columns = ['Min']
        dmax = pd.DataFrame(dp.max())
        dmax.columns = ['Max']
        dlast = pd.DataFrame(dp.iloc[-1])
        dlast.columns = ['Last']
        self.watchlist = self.watchlist.join(dmin)
        self.watchlist = self.watchlist.join(dmax)
        self.watchlist = self.watchlist.join(dlast)
        self.watchlist['SinceMin'] = self.watchlist['Last'] / self.watchlist['Min'] -1
        self.watchlist['SinceMax'] = self.watchlist['Last'] / self.watchlist['Max'] -1

        self.watchlist['7dReturn'] = self.equity_prices_local.loc[self.today] / self.equity_prices_local.loc[self.today - pd.Timedelta(days = 7)] - 1
        self.watchlist['28dReturn'] = self.equity_prices_local.loc[self.today] / self.equity_prices_local.loc[self.today - pd.Timedelta(days = 28)] - 1
        
        self.watchlist.drop(columns = ['Min','Max','Last'], inplace = True)


    def moving_average_crossing(self, short=50, medium=100, long=200):
        dp = self.equity_prices_local.loc[self.today - pd.Timedelta(days = 365+280):self.today]
        ma = {}
        ma['MA50'] = ti.moving_average(dp, short)
        ma['MA100'] = ti.moving_average(dp, medium)
        ma['MA200'] = ti.moving_average(dp, long)
        
        #print(ma['MA50'])
        
        for i in ['MA50','MA100','MA200']:
            df = ti.moving_average_crossing(dp, ma[i])
            df.columns = ['Pvs{0}'.format(i)]
            self.watchlist = self.watchlist.join(df)

    
    
    
    
    def generate_alerts(self):
        self.price_action()
        self.sharp()
        self.moving_average_crossing()
        
        self.watchlist.drop(columns = ['Ticker'], inplace = True)
        #print(self.watchlist)
        
        writer = pd.ExcelWriter('watchlist_output.xlsx')
        self.watchlist.to_excel(writer, 'Output')
        writer.save()


if __name__ == '__main__':
    
    a = Alert(today = pd.to_datetime('2018-10-1'))
    
    mywb = 'C:\\Users\\yewen\\OneDrive\\Documents\\Sheets\\Historical price.xlsx'
    a.get_price_history(workbook_name=mywb)
    a.generate_alerts()

    