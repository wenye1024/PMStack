'''
Created on May 13, 2019

@author: yewen
'''


import pandas as pd
import numpy as np
import PMStack.Equity_port.historical_trades as trade_history
import PMStack.Equity_port.historical_price as historical_price
import PMStack.Tools.technical_indicators as ti
import PMStack.Tools.helpers as helpers
import warnings
import PMStack.Tools.excel_writer as excel_writer
import PMStack.Equity_port.stock_info as stock_info




class Price_Actions:
    def __init__(self, equity_prices, equity_daily_return, benchmark = 'SPX', watchlist = None, writer = None):
        self.equity_prices = equity_prices
        self.equity_daily_return = equity_daily_return
        
        if watchlist == None:
            watchlist = list(self.equity_prices.columns)
        else:
            watchlist = watchlist.copy()
            if benchmark not in watchlist:
                watchlist.append(benchmark)
            self.equity_prices = self.equity_prices[watchlist]
            self.equity_daily_return = self.equity_daily_return[watchlist]
            
        
        self.benchmark = benchmark
        self.equity_daily_excess_return = self.equity_daily_return.subtract( self.equity_daily_return[benchmark], axis = 0)
        self.writer = writer
        self.reset_calculation()

    def reset_calculation(self):
        self.__excess_return__ = None
        self.__information_ratio__ = None

    def get_excess_return(self, duration = [1,3,5,20,200]):
        if self.__excess_return__ == None:
        
            def temp(duration):
                for f in duration:
                    d = self.equity_prices.iloc[-1] / self.equity_prices.iloc[-f-1] - 1
                    d.name = '{:d}D_excess_r'.format(f)
                    yield d        
    
            dd = pd.concat(temp(duration), axis =1)
            self.__excess_return__ = dd - dd.loc[self.benchmark]
            self.__excess_return_duration__ = duration

        return self.__excess_return__


    def get_information_ratio(self, duration = [5,20,200]):
        if self.__information_ratio__ == None:
    
            def temp(duration):
                for f in duration:
                    d = self.equity_daily_excess_return.iloc[-f : -1]
                    dd = d.mean() / d.std(ddof = 0)
                    dd.name = '{:d}D_info_ratio'.format(f)
                    #writer.print('test.xlsx', [d, d.mean(), d.std(), dd], ['a','mean','std','ir'])
                    dd[self.benchmark] = 0
                    yield dd        
            
            self.__information_ratio__ = pd.concat(temp(duration), axis =1)
            self.__information_ratio_duration__ = duration
            

        return self.__information_ratio__


    def get_MACD_signal(self, print_out=False, output_file = None):
        buy, sell = ti.action_macd(self.equity_prices)
        if print_out:
            print('Buy and sell signals from MACD')
            print('    Stocks with buy signal:', buy[buy].index.values)
            print('    Stocks with sell signal:', sell[sell].index.values)

        if output_file != None:
            self.writer.print(output_file, [buy, sell], ['MACD_buy','MACD_sell'])
        
        return buy, sell

    def get_price_actions(self, include_MACD_signal = True, print_out=False, output_file = None):
        

        d = pd.concat([self.get_excess_return(),self.get_information_ratio()], axis = 1)
        
        for c in d.columns.values:
            d = d.sort_values([c])
            d.loc[3:-3, c] = False
        
        if include_MACD_signal:
            d['MACD_Buy'], d['MACD_Sell'] = self.get_MACD_signal(print_out = False)

        d['Price_Action'] = False

        
        def temp(str, duration):
            for i in duration:
                yield str.format(i)
            
        cols = list(temp('{:d}D_excess_r', self.__excess_return_duration__))
        cols.extend(temp('{:d}D_info_ratio', self.__information_ratio_duration__))
        
        for row in d.index:
            
            r = ''
            for col in cols:
                if d.loc[row, col] != False:
                    r = r + col + " is {:,.2%}, ".format(d.loc[row,col])
            
            if d.loc[row, 'MACD_Buy']: r = r + 'MACD_Buy' + ", "
            if d.loc[row, 'MACD_Sell']: r = r + 'MACD_Sell'            

            if r!='':
                d.loc[row, 'Price_Action'] = r   
           
        d = d[d['Price_Action']!=False]

        
        if output_file != None:
            self.writer.print(output_file, [d], ['PriceActions'])
            
        if print_out:
            print(d)
                
        return d



if __name__ == '__main__':

    mywb = 'C:\\Users\\ye.wen\\Downloads\\Paper portfolio tracking v1.1.xlsx'
    
    stock_info = stock_info.get_stock_info(mywb, 'Latest price')
    watchlist = stock_info['OnWatchList'][stock_info['OnWatchList']>0.1]
    watchlist = watchlist.index.values

    selected = ['SPX', 'COMP']
    selected.extend(watchlist)
    selected.remove('PLACEHOLDER')
    #print(selected)
    
    
    r = historical_price.get_price_history(workbook_name = 'C:\\Users\\ye.wen\\Downloads\\Historical price.xlsx', print_out = False)

    output_dir = 'C:\\Users\\ye.wen\\Downloads\\'
    writer = excel_writer.ExcelWriter(output_dir)
    output_file='price_actions'
    
    pa = Price_Actions(r['equity_price_local'][selected], r['equity_daily_return_local'][selected], writer=writer)

    pa.get_price_actions(output_file=output_file)
    
    print()
    
    #pa.get_MACD_signal(output_file=output_file)
    #writer.save()
