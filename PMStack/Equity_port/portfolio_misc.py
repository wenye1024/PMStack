'''
Created on Mar 3, 2019

author: yewen
'''
import pandas as pd
import numpy as np
from PMStack.Equity_port import portfolio_write, historical_price

import warnings
import inspect

class Portfolio_Misc(portfolio_write.Portfolio_Write):

    def __init__(self, file_price, file_trades, start_date, end_date=None, capital=None, benchmark = 'SPX', benchmark2=[], base_currency = 'USD', ignore_ticker_list = [], file_stock_info = None, output_dir = '' ):
        super().__init__(file_price, file_trades, start_date, end_date, capital, benchmark, benchmark2, base_currency, ignore_ticker_list, file_stock_info, output_dir)
    


    def define_equity_group(self, new_ticker, tickers, long_short = 'Long'):

        historical_price.define_equal_weight_daily_rebalance_port(new_ticker, tickers, self.equity_prices_local_full_history, self.equity_daily_return_local_full_history, self.company_name)
        historical_price.define_equal_weight_daily_rebalance_port(new_ticker, tickers, self.equity_prices_USD_full_history, self.equity_daily_return_USD_full_history, self.company_name)
                
        self.equity_PnL[long_short][new_ticker] = self.equity_PnL[long_short][tickers].sum(axis = 1)
        self.equity_PnL_cumulative[long_short][new_ticker] = self.equity_PnL_cumulative[long_short][tickers].sum(axis = 1)
        self.equity_value_USD[long_short][new_ticker] = self.equity_value_USD[long_short][tickers].sum(axis = 1)
        self.equity_cumulative_cost_USD[long_short][new_ticker] = self.equity_cumulative_cost_USD[long_short][tickers].sum(axis = 1)


    def sanity_check(self, date=None):
        if date is None: date = self.end_date
        for long_short in self.long_short_list:
            print('{} side has {:,.0f} shares'.format(long_short, self.equity_balance[long_short].loc[date].sum()))

        
    
