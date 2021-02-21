'''
Created on Oct 8, 2020
author: yewen


'Balance_EOD' includes equities, options and cash
'Cash' is the cash balance that includes dividends received
'Daily_Return' is the Daily_P&L over Balance_EOD
'Time_Weighted_Port_Return' is the compounded Daily_Return

'''


import pandas as pd
import numpy as np
from PMStack.Tools import finance, helpers
from PMStack.Equity_port import historical_price, historical_trades, stock_info
import warnings
import inspect
from datetime import timedelta
from IPython.core.display import display

class Portfolio_Core:

    
    def __balance_roll_over__(self, current_index, current_date):
        for long_short in ['Long','Short','Options']:
            self.equity_balance[long_short].iloc[current_index] = self.equity_balance[long_short].iloc[current_index-1]
            self.equity_cumulative_cost_USD[long_short].iloc[current_index] = self.equity_cumulative_cost_USD[long_short].iloc[current_index-1]
            self.currency_balance[long_short].iloc[current_index] = self.currency_balance[long_short].iloc[current_index-1]
            
        #self.option_balance.iloc[current_index] = self.option_balance.iloc[current_index-1]
        #self.option_cumulative_cost_USD.iloc[current_index] = self.option_cumulative_cost_USD.iloc[current_index-1]
        #self.currency_balance_options.iloc[current_index] = self.currency_balance_options.iloc[current_index-1]
        self.currency_balance_other.iloc[current_index] = self.currency_balance_other.iloc[current_index-1]
        self.PA_snapshots.iloc[current_index] = self.PA_snapshots.iloc[current_index-1]
        self.PA_snapshots_other.iloc[current_index] = self.PA_snapshots_other.iloc[current_index-1]
        self.PA_snapshots_pure_cap_return.iloc[current_index] = self.PA_snapshots_pure_cap_return.iloc[current_index-1]

    def __get_trade_history__ (self, workbook_name, sheet_name, accounts = None):
        self.trades = historical_trades.get_trade_history(workbook_name, sheet_name, accounts)
        #print(self.trades)   
        
    def __get_price_history__ (self, file_price, base_currency):
           
        r = historical_price.get_price_history(**file_price, base_currency=base_currency)
        self.forex = r['forex']
        self.company_name = r['company_name']
        self.equity_currency = r['equity_currency']
        self.equity_prices_local = r['equity_price_local'] 
        self.equity_prices_USD = r['equity_price_USD']
        self.equity_daily_return_local = r['equity_daily_return_local']
        self.equity_daily_return_USD = r['equity_daily_return_USD']
        self.end_date = r['end_date']



        
    '''
    going through transactions (self.trades) to update the following
        equity_balance_long and equity_balance_short
        equity_value_USD
        currency_balance_long and currency_balance_short
        PA_snapshots_pure_cap_return['Equity_Long','Currency_Long','Equity_Short','Currency_Short']
        PA_Snapshot['Cumulative_Capital_Flow', 'Cumulative_Dividends']
    '''
    def __process_trades__(self, ignore_ticker_list = []):
        current_index = 0
        current_date = self.PA_snapshots.index.values[current_index]
                
        index_count = len(self.PA_snapshots.index)
        
        #print(self.PA_snapshots.index)
        

        for idx in self.trades.index:
            trade = self.trades.loc[idx]
            #print(trade['Date'])
            tdate = np.datetime64(trade['Date'])

            
            
            #print('tdate =', tdate)
            #print('current_date =', current_date)
            if (current_date > tdate):
                raise ValueError('Error in trades file: Trades not in order of time, or trades predate historical price data. Current transaction is at', tdate, ' However the system is already processing', current_date)
            
            
            
            while current_date != tdate:
                #print(tdate, current_date, current_index)
                current_index += 1
                if (current_index >= len(self.PA_snapshots.index)):
                    raise ValueError('Error in trades file: maybe a trade is happening on a weekend -', tdate)
                last_date = current_date
                #print(current_index, current_date)
                current_date = self.PA_snapshots.index.values[current_index]
                #print('current_date = ', current_date)
                
                
                self.__balance_roll_over__(current_index, current_date)
            

            
            if (trade['TransactionType'] == 'DEPOSIT'):
                self.currency_balance_other.at[current_date, trade['Currency']] += trade['TransactionAmount']
                self.PA_snapshots.at[current_date, 'Cumulative_Capital_Flow'] += trade['TransactionAmount'] * self.forex.at[current_date, trade['Currency']]
                #print('Deposit', current_date, self.PA_snapshots.at[current_date, 'Cumulative_Capital_Flow'])
            elif (trade['TransactionType'] in ['DIVIDENDS', 'DIVIDEND']):

                self.currency_balance_other.at[current_date, trade['Currency']] += trade['TransactionAmount']
                usd_amount = trade['TransactionAmount'] * self.forex.at[current_date, trade['Currency']]
                self.PA_snapshots.at[current_date, 'Cumulative_Dividends'] += usd_amount
                if usd_amount>0.0:
                    self.PA_snapshots_other.loc[current_date, 'Cumulative_Dividends_Long'] += usd_amount
                else:
                    self.PA_snapshots_other.loc[current_date, 'Cumulative_Dividends_Short'] += usd_amount
            elif (trade['TransactionType'] == 'INTEREST'):
                self.currency_balance_other.at[current_date, trade['Currency']] += trade['TransactionAmount']
                usd_amount = trade['TransactionAmount'] * self.forex.at[current_date, trade['Currency']]
                if trade['Ticker'] == 'SHORT INTEREST':
                    self.PA_snapshots_other.loc[current_date, 'Cumulative_Short_Interest'] += usd_amount
                else:
                    self.PA_snapshots_other.loc[current_date, 'Cumulative_Credit_Interest'] += usd_amount
            elif (trade['TransactionType'] == 'TRANSFERS'):
                continue
            elif (trade['TransactionType'] == 'FEE'):
                continue
            elif (trade['TransactionType'] == 'BORROW FEE'):
                continue
            elif (trade['SecurityType'] in ['Equity', 'ETF']):
                
                if trade['Ticker'] in ignore_ticker_list:
                    #print('Trades intentionally ignored: ', pd.Timestamp(tdate).strftime('%Y-%m-%d'), trade['TransactionType'], trade['Ticker']  )
                    continue

                
                long_short = 'Long'
                
                
                
                if (self.equity_balance_short.at[current_date, trade['Ticker']] != 0):
                    long_short = 'Short'
                elif (self.equity_balance_long.at[current_date, trade['Ticker']] == 0 and trade['Quantity'] < 0.0 ):
                    long_short = 'Short'
                
                    
                self.trades.loc[idx,'Long_Short'] = long_short
                
                #if (long_short == 'Short'):
                #if (t.type in ['SHORT','COVER', 'Short', 'Cover']):
                #    print(long_short, current_date, t.ticker, t.quantity, self.equity_balance_short.at[current_date, t.ticker])
            
                if (trade['TransactionType'] in ['BUY', 'SELL', 'SHORT', 'LONG','COVER']):
                    amount = trade['TransactionAmount']
                    self.trades.loc[idx,'TransactionAmountUSD'] = amount * self.forex.at[current_date, trade['Currency']]
                elif (trade['TransactionType'] in ['OPTION ASSIGNMENT', 'OPTION EXECUTION']):
                    amount = (-trade['Quantity'] * self.equity_prices_local.at[current_date, trade['Ticker']])
                elif (trade['TransactionType'] in ['STOCK DIVIDEND', 'STOCK DIVIDENDS']):
                    amount = 0.0
                    
                self.equity_balance[long_short].at[current_date, trade['Ticker']] += trade['Quantity']
                self.currency_balance[long_short].at[current_date, trade['Currency']] += amount  
                self.equity_cumulative_cost_USD[long_short].at[current_date, trade['Ticker']] += amount * self.forex.at[current_date, trade['Currency']]
                
                
                
                if (trade['TransactionType'] in ['BUY', 'LONG','COVER']) and (long_short == 'Long'):
                    self.lastBuyPrice[trade['Ticker']] = amount * self.forex.at[current_date, trade['Currency']] / -trade['Quantity']
                    if (self.lastBuyPrice[trade['Ticker']] > self.hightestBuyPrice[trade['Ticker']]): 
                        self.hightestBuyPrice[trade['Ticker']] = self.lastBuyPrice[trade['Ticker']]                       
                elif (trade['TransactionType'] in ['SELL', 'SHORT']) and (long_short == 'Short'):
                    self.lastSellPrice[trade['Ticker']] = amount * self.forex.at[current_date, trade['Currency']] / -trade['Quantity']
                    if (self.lastSellPrice[trade['Ticker']]<self.lowestSellPrice[trade['Ticker']]):
                        self.lowestSellPrice[trade['Ticker']] = self.lastSellPrice[trade['Ticker']]
                    

                if helpers.isZero(self.equity_balance[long_short].at[current_date, trade['Ticker']]): # clear out highest buy / lowest sell price if the position is back to zero
                    
                    self.equity_balance[long_short].at[current_date, trade['Ticker']] = 0.0 # clear out fraction shares
                    
                    if (long_short == 'Long'):
                        self.hightestBuyPrice[trade['Ticker']] = 0.0
                    elif (long_short == 'Short'):
                        self.lowestSellPrice[trade['Ticker']] = 9999999.0

                
                
                
            elif(trade['SecurityType'] == 'Put' or trade['SecurityType'] == 'Call'):

                #print('Option trades not processed: ', pd.Timestamp(tdate).strftime('%Y-%m-%d'), trade['TransactionType'], trade['Ticker']  )
                #continue

                #print(t.ticker, t.quantity, t.transaction_amount, t.currency)
                #ticker, expiration, strike, tp = finance.optionDecompose(trade['Ticker'])
                if(not(trade['Ticker'] in self.equity_balance['Options'].columns)):
                    self.equity_balance['Options'][trade['Ticker']] = 0
                    self.equity_cumulative_cost_USD['Options'][trade['Ticker']] = 0

                self.equity_balance['Options'].loc[current_date, trade['Ticker']] += trade['Quantity']

                if (trade['TransactionType'] in ['BUY', 'LONG', 'SELL','SHORT','COVER']):
                    amount = trade['TransactionAmount']
                elif (trade['TransactionType'] in ['Option Assignment', 'Option Execution']):
                    print ('Code not implemented - Option assignment or execution')

                self.currency_balance_options.at[current_date, trade['Currency']] += amount
                self.equity_cumulative_cost_USD['Options'].at[current_date, trade['Ticker']] += amount * self.forex.at[current_date, trade['Currency']]
                   
            elif(trade['SecurityType'] == 'Forex'):
                #print('Forex trades not processed: ', pd.Timestamp(tdate).strftime('%Y-%m-%d'), trade['TransactionType'], trade['Ticker']  )
                continue
            else:
                print('Trades not processed: ', pd.Timestamp(tdate).strftime('%Y-%m-%d'), trade['TransactionType'], trade['Ticker']  )
                continue
        # end of for loop: for t in self.trades
        
        #print(self.equity_cumulative_cost_USD['Long']['002415-CN'])
        
        
        while True:
            current_index += 1
            if current_index == index_count: break
            last_date = current_date
            current_date = self.PA_snapshots.index.values[current_index]
            
            self.__balance_roll_over__(current_index, current_date)
            

        self.trades.set_index('Date', inplace=True)
        x = ((pd.DataFrame(self.trades['TransactionType'])=='BUY') | (pd.DataFrame(self.trades['TransactionType'])=='SELL') | (pd.DataFrame(self.trades['TransactionType'])=='LONG') | (pd.DataFrame(self.trades['TransactionType'])=='SHORT')| (pd.DataFrame(self.trades['TransactionType'])=='COVER')).any(axis = 1)
        x = self.trades[x].copy()
        y = ((pd.DataFrame(x['SecurityType'])=='Equity') | (pd.DataFrame(x['SecurityType'])=='ETF')).any(axis = 1)        
        self.equity_and_ETF_trades = x[y].copy()
                               

    
    def __init__(self, file_price, file_trades, start_date, end_date=None, capital=None, benchmark = 'SPX', benchmark2=[], base_currency = 'USD', ignore_ticker_list = [], file_stock_info = None):
        
        self.__debug_mode__ = False
        if self.__debug_mode__: print('Entering', inspect.stack()[0][3]) 

        
        self.option_contract_size = 100
        
        #self.file_output = True
        
        
        self.__get_price_history__(file_price, base_currency=base_currency)
        self.__get_trade_history__(**file_trades) #file_trades['workbook_name'], file_trades['sheet_name'])

        self.capital = capital
        self.benchmark = benchmark
        self.benchmark2 = [benchmark]
        self.benchmark2.extend(benchmark2)
        self.base_currency = base_currency
        
        if file_stock_info == None:
            self.stock_info = None
        else:
            self.stock_info = stock_info.get_stock_info(file_stock_info['workbook_name'], file_stock_info['sheet_name'])

        #equity_balance keeps track of number of shares for each position (columns) on each day (row)
        self.equity_balance_long = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.equity_prices_USD.columns.values).fillna(0.0)
        self.equity_balance_short = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.equity_prices_USD.columns.values).fillna(0.0)
        self.equity_balance = {'Long': self.equity_balance_long, 'Short': self.equity_balance_short}

        #equity_cumulative_cost keeps track of cumulative cashflows involved with each position
        self.equity_cumulative_cost_long_USD = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.equity_prices_USD.columns.values).fillna(0.0)
        self.equity_cumulative_cost_short_USD = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.equity_prices_USD.columns.values).fillna(0.0)
        self.equity_cumulative_cost_USD = {'Long': self.equity_cumulative_cost_long_USD, 'Short': self.equity_cumulative_cost_short_USD}
        
        #option related
        self.equity_balance_options = pd.DataFrame(index = self.equity_prices_USD.index.copy())
        self.equity_balance['Options'] = self.equity_balance_options
        self.equity_cumulative_cost_options_USD = pd.DataFrame(index = self.equity_prices_USD.index.copy())
        self.equity_cumulative_cost_USD['Options'] = self.equity_cumulative_cost_options_USD
        
        
        self.currency_names = self.forex.columns.tolist()

        
        self.currency_balance_long = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.currency_names).fillna(0.0) # equity longs
        self.currency_balance_short = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.currency_names).fillna(0.0) # equity shorts
        self.currency_balance_options = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.currency_names).fillna(0.0) # options
        self.currency_balance_other = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=self.currency_names).fillna(0.0) # interest, dividends, etc
        self.currency_balance = {'Long': self.currency_balance_long, 'Short': self.currency_balance_short, 'Options':self.currency_balance_options}
        
        self.PA_snapshots = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=['Cumulative_Capital_Flow', 'Cumulative_Dividends', 'Balance_EOD', 'Daily_P&L','Daily_Return','Cum_P&L','Port_NAV','Benchmark_Return',  'Benchmark_Normalized','Gross_Exposure','Net_Exposure','Capital_Deployment']).fillna(0.0)
        self.PA_snapshots_other = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=['Cumulative_Dividends_Long', 'Cumulative_Dividends_Short', 'Cumulative_Credit_Interest', 'Cumulative_Short_Interest']).fillna(0.0)
        self.PA_snapshots_pure_cap_return = pd.DataFrame(index = self.equity_prices_USD.index.copy(), columns=['Equity_Long', 'Currency_Long', 'Capital_Base_Long', 'P&L_Long','Return_Long','Equity_Short','Currency_Short','Capital_Base_Short','P&L_Short','Return_Short', 'Options', 'Currency_Options', 'P&L_Options', 'Total_P&L', 'Total_Return_on_Capital_Base_Long','NAV','NAV_Long','Cum_P&L','Cum_P&L_Long','Cum_P&L_Short']).fillna(0.0)

        self.lastBuyPrice = {}
        self.lastSellPrice = {}
        self.hightestBuyPrice = pd.Series(index = self.equity_prices_USD.columns.copy()).fillna(0.0)
        self.lowestSellPrice = pd.Series(index = self.equity_prices_USD.columns.copy()).fillna(99999.0)
        
        
        '''
        Going through self.trades to reflect transactions
        '''
        
        self.__process_trades__(ignore_ticker_list)
        
        
        self.start_date = helpers.string_to_date(start_date)
        if end_date != None: self.end_date = helpers.string_to_date(end_date)
        
        
        self.has_option_book = (abs(self.equity_balance_options[self.start_date:self.end_date].values).sum()>0)

        if helpers.isZero(self.equity_balance['Short'].values.sum()):
            self.long_only = True
            self.long_short_list = ['Long']
        else:
            self.long_only = False
            self.long_short_list = ['Long', 'Short']

        self.long_short_options = self.long_short_list.copy()
        if self.has_option_book: self.long_short_options.append('Options')


        self.__setAnalysisScope__(self.start_date, self.end_date)


        #print(self.equity_value_USD)
        

        self.metrics = {}
        self.__calculate_portfolio_PnL__()
        self.__calculated_option_intrinsic_values__()
        self.__prepare_current_holdings_and_exposure__()

        self.__calculate_dividends__()
        self.__calculate_interest__()
        self.__calculate_vol_corr__()

        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3]) 
    


    # start_date should be 1 day before your desired period, so that you can establish a baseline
    # end_date is inclusive, i.e., you do NOT need to use the day after your desired period        
    def __setAnalysisScope__(self, start_date, end_date):

        if self.__debug_mode__: print('Entering', inspect.stack()[0][3]) 

        
        # balance-related scope: start_date to end_date
        # P&L-related scope: start_date_plus_one to end_date
        end_date_plus_one = end_date + timedelta(1)

        self.equity_prices_USD_full_history = self.equity_prices_USD
        self.equity_prices_local_full_history = self.equity_prices_local 
        self.equity_daily_return_local_full_history = self.equity_daily_return_local
        self.equity_daily_return_USD_full_history = self.equity_daily_return_USD
        
        self.equity_prices_USD = self.equity_prices_USD.loc[start_date : end_date_plus_one]
        self.equity_prices_local = self.equity_prices_local.loc[start_date : end_date_plus_one] 
        
        self.__init_date_indexing__()
        
        self.benchmark_return = self.equity_prices_local.loc[end_date, self.benchmark] / self.equity_prices_local.loc[start_date, self.benchmark] - 1
        
        self.equity_daily_return_USD = self.equity_daily_return_USD.loc[start_date : end_date_plus_one]
        self.equity_daily_return_USD = self.equity_daily_return_USD.iloc[1:]
        
        self.start_date_plus_one = self.equity_daily_return_USD.index[0]
        start_date_plus_one = self.start_date_plus_one
        
        self.equity_daily_return_local = self.equity_daily_return_local.loc[start_date_plus_one : end_date_plus_one]
        
        
        self.PA_snapshots = self.PA_snapshots.loc[start_date : end_date_plus_one]
        self.PA_snapshots_pure_cap_return = self.PA_snapshots_pure_cap_return.loc[start_date : end_date_plus_one]
        self.forex = self.forex[start_date : end_date_plus_one]
        
        self.trades_full_history = self.trades
        self.trades = self.trades[start_date_plus_one : end_date_plus_one]
        self.equity_and_ETF_trades = self.equity_and_ETF_trades[start_date_plus_one : end_date_plus_one]



        'Equity and Options common part'

        self.equity_value_USD = {}
        self.equity_cashflows = {}

        for long_short in ['Long','Short','Options']:
            self.equity_balance[long_short] = self.equity_balance[long_short].loc[start_date : end_date_plus_one]
            self.equity_value_USD[long_short] = (self.equity_balance[long_short] * self.equity_prices_USD).fillna(0.0)

            self.equity_cashflows[long_short] = self.equity_cumulative_cost_USD[long_short].diff()
            self.equity_cashflows[long_short] = self.equity_cashflows[long_short].loc[start_date : end_date_plus_one]
            self.equity_cashflows[long_short].iloc[0] = 0.0



        
        '''
        Option Specific
        '''
        self.equity_value_USD['Options'] *= self.option_contract_size
        
        



        self.currency_balance_USD = {}
        self.equity_investment_cashflows = {}
        self.equity_investment_cashflows_shift_minus_one = {}
        self.equity_payment_cashflows = {}


        '''
        equity_holding_window = 0 on the day the position is opened, =1 on the day the position is closed
        equity_holding_and_trading_window = 1 on both position open and close day
        '''
        self.equity_holding_window = {}
        self.equity_trading_window = {}
        self.equity_holding_and_trading_window = {}
        self.traded_equity_list = {}


        for long_short in ['Long', 'Short', 'Options']:
            #traded_equities is the list of tickers that has been traded during the period, by descending orders of position size
            owned_equities = set(helpers.non_zero_dataFrame_columns_or_rows(self.equity_balance[long_short], 'column'))
            traded_equities = set(helpers.non_zero_dataFrame_columns_or_rows(self.equity_cashflows[long_short], 'column'))
            traded_equities = list(traded_equities | owned_equities)
            self.traded_equity_list[long_short] = traded_equities
            


            last_row = self.equity_value_USD[long_short].loc[end_date, traded_equities].sort_values(axis = 'index', ascending = (long_short == 'Short'))
            traded_equities = last_row.index



            self.equity_balance[long_short] = self.equity_balance[long_short].loc[:, traded_equities]
            self.equity_value_USD[long_short] = self.equity_value_USD[long_short].loc[:, traded_equities]

            df = self.equity_value_USD[long_short]
            l = df.columns[df.isna().any()].tolist()
            if len(l) > 0:
                warnings.warn('Missing pricing data for '+str(l))





            self.currency_balance[long_short] = self.currency_balance[long_short].loc[start_date : end_date_plus_one]
            
            
            self.equity_cumulative_cost_USD[long_short] = self.equity_cumulative_cost_USD[long_short].loc[:, traded_equities]
            self.equity_cashflows[long_short] = self.equity_cashflows[long_short].loc[:, traded_equities]
            
            '''
            Negative cashflow means investment, i.e., putting in capital to buy equities
            Positive cashflow means payment, i.e., selling and cashing out
            '''
            df = self.equity_cashflows[long_short]
            if (long_short == 'Short'): df = df*-1 # for shorts, inverse cash flow so that investment and payment are consistent in signs (+/-) with longs
            window = df.apply(lambda x: x<0)
            self.equity_investment_cashflows[long_short] = df*window
            self.equity_payment_cashflows[long_short] = df - self.equity_investment_cashflows[long_short]
            
            self.equity_cumulative_cost_USD[long_short] = self.equity_cumulative_cost_USD[long_short].loc[start_date : end_date_plus_one]
            self.equity_payment_cashflows[long_short] = self.equity_payment_cashflows[long_short].loc[start_date : end_date_plus_one]
            self.equity_investment_cashflows[long_short] = self.equity_investment_cashflows[long_short].loc[start_date : end_date_plus_one]
            self.equity_investment_cashflows_shift_minus_one[long_short] = self.equity_investment_cashflows[long_short].shift(-1)
            self.equity_investment_cashflows_shift_minus_one[long_short].iloc[-1] = 0
            

            # for the days there is holding, return 1, else 0; 0 for the entry day, 1 for exit day
            self.equity_holding_window[long_short] = self.equity_balance[long_short].shift(1).iloc[1:].applymap(lambda x: 1*helpers.isNotZero(x)) 
            self.equity_trading_window[long_short] = self.equity_cashflows[long_short].iloc[1:].applymap(lambda x: 1*helpers.isNotZero(x)) 
            self.equity_holding_and_trading_window[long_short] = 1*(self.equity_holding_window[long_short] | self.equity_trading_window[long_short])

            
            '''
            
            # this is testing 
            if (long_short == 'Short'):
                test_start_date = pd.to_datetime('2018-4-30')
                test_end_date = pd.to_datetime('2018-5-3')
                print(self.equity_cumulative_cost_USD[long_short].loc[test_start_date:test_end_date,'AAPL'])
                print(df.loc[test_start_date:test_end_date,'AAPL'])
                print(window.loc[test_start_date:test_end_date,'AAPL'])
                print(a.loc[test_start_date:test_end_date,'AAPL'])
                print(b.loc[test_start_date:test_end_date,'AAPL'])
            '''
        # end of for loop : for long_short in ['Long', 'Short']:

        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3]) 



    def __calculated_option_intrinsic_values__(self):
        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])

        self.option_intrinsic_value = pd.DataFrame(index = self.equity_balance['Options'].index.values, columns=self.equity_balance['Options'].columns.values).fillna(0.0)
        self.option_dict = pd.DataFrame(index = self.equity_balance['Options'].columns.values, columns = ['Underlying', 'Exp', 'Strike', 'Type'])
        for ticker in self.equity_balance['Options'].columns:            
            tk, expiration, strike, tp = finance.optionDecompose(ticker)
            self.option_dict.loc[ticker,'Underlying'] = tk
            self.option_dict.loc[ticker,'Exp'] = expiration
            self.option_dict.loc[ticker,'Strike'] = strike
            self.option_dict.loc[ticker,'Type'] = tp
            for current_date in self.equity_balance['Options'].index:
                iv = finance.optionIntrinsicValue(strike, tp, self.equity_prices_local.loc[current_date, tk])
                self.option_intrinsic_value.loc[current_date, ticker] = iv * self.equity_balance['Options'].loc[current_date, ticker] * self.option_contract_size
                
        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])


    def __calculate_portfolio_PnL__(self):        
        
        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])
        

        pas = self.PA_snapshots_pure_cap_return #to shorten the code
        pas['Options'] = self.equity_value_USD['Options'].sum(axis = 1)
        pas['Currency_Options'] = (self.currency_balance_options * self.forex).sum(axis = 1)      
        pas['Total_Options'] = pas['Options'] + pas['Currency_Options']

        #self.option_PnL = (self.option_intrinsic_value + self.equity_cumulative_cost_USD['Options']).diff().iloc[1:]
        
        '''
        Equity, Currency, Options analysis
        '''

        for long_short in ['Long', 'Short']:
            pas['Equity_'+long_short] = self.equity_value_USD[long_short].sum(axis = 1)
            self.currency_balance_USD[long_short] = self.currency_balance[long_short] * self.forex
            pas['Currency_'+long_short] = self.currency_balance_USD[long_short].sum(axis = 1)



        pas['Total_Long'] = pas['Equity_Long'] + pas['Currency_Long']
        pas['Total_Short'] = pas['Equity_Short'] + pas['Currency_Short']
        
        
        

        self.equity_PnL = {}
        self.equity_daily_capital_base = {} # base line to calculate return. The diff is PnL
        self.equity_PnL_cumulative = {}
        for long_short in ['Long', 'Short', 'Options']:
            df = self.equity_value_USD[long_short] + self.equity_cumulative_cost_USD[long_short]
            self.equity_PnL[long_short] = df.diff().iloc[1:]
            self.equity_daily_capital_base[long_short] = self.equity_value_USD[long_short].shift(1) - self.equity_cashflows[long_short]
            
        for long_short in self.long_short_options:
            self.equity_PnL_cumulative[long_short] = self.equity_PnL[long_short].cumsum()
        

            
        
        '''
        Calculate P&L and return
        '''
        previous_date = pas.index[0]
        for current_date in pas.index[1:]:
            pas.loc[current_date, 'P&L_Long'] = pas.loc[current_date, 'Total_Long'] - pas.loc[previous_date, 'Total_Long']
            pas.loc[current_date, 'Capital_Base_Long'] = pas.loc[previous_date, 'Equity_Long']
            if(pas.loc[previous_date,'Currency_Long'] > pas.loc[current_date, 'Currency_Long']):
                pas.loc[current_date, 'Capital_Base_Long'] +=  pas.loc[previous_date,'Currency_Long'] - pas.loc[current_date, 'Currency_Long']
            
            if (helpers.isZero(pas.loc[current_date, 'Capital_Base_Long'])):
                pas.loc[current_date, 'Return_Long'] = 0
            else:
                pas.loc[current_date, 'Return_Long'] = pas.loc[current_date, 'P&L_Long'] / pas.loc[current_date, 'Capital_Base_Long']
            
            pas.loc[current_date, 'P&L_Short'] = pas.loc[current_date, 'Total_Short'] - pas.loc[previous_date, 'Total_Short']
            pas.loc[current_date, 'Capital_Base_Short'] = pas.loc[previous_date, 'Equity_Short']
            if(pas.loc[previous_date,'Currency_Short'] < pas.loc[current_date, 'Currency_Short']):
                pas.loc[current_date, 'Capital_Base_Short'] +=  pas.loc[previous_date,'Currency_Short'] - pas.loc[current_date, 'Currency_Short']

            if (helpers.isZero(pas.loc[current_date, 'Capital_Base_Short'])):
                pas.loc[current_date, 'Return_Short'] = 0
            else:
                pas.loc[current_date, 'Return_Short'] = pas.loc[current_date, 'P&L_Short'] / pas.loc[current_date, 'Capital_Base_Short']
            
            
            
            
            
            
            pas.loc[current_date, 'P&L_Options'] = pas.loc[current_date, 'Total_Options'] - pas.loc[previous_date, 'Total_Options']
            pas.loc[current_date, 'Total_P&L'] = pas.loc[current_date, 'P&L_Long'] + pas.loc[current_date, 'P&L_Short'] + pas.loc[current_date, 'P&L_Options']

            if (helpers.isZero(pas.loc[current_date, 'Capital_Base_Long'])):
                pas.loc[current_date, 'Total_Return_on_Capital_Base_Long'] = 0
            else:
                pas.loc[current_date, 'Total_Return_on_Capital_Base_Long'] = pas.loc[current_date, 'Total_P&L'] / pas.loc[current_date, 'Capital_Base_Long']
            
            previous_date = current_date
        

        '''
        Calculate daily currency balance p&l due to FX change
        Daily currency balance P&L is included in the Long_P&L, Short_P&L, Options_P&L, but not in the P&L by each equities
        As a result, the discrepancy between Long_P&L and the sum of each long positions's P&L is the FX change
        '''        
        a = self.forex.diff(1)
        b_long = (self.currency_balance['Long'].shift(1) * a).sum(axis = 1) 
        b_short = (self.currency_balance['Short'].shift(1) * a).sum(axis = 1) 
        b_options = (self.currency_balance['Options'].shift(1) * a).sum(axis = 1) 
        self.currency_PnL = pd.concat([b_long, b_short, b_options], axis = 1).iloc[1:]
        self.currency_PnL.columns = ['Long','Short','Options']



        
 
        pas['NAV'] = (1 + pas['Total_Return_on_Capital_Base_Long']).cumprod()
        pas['NAV_Long'] = (1 + pas['Return_Long']).cumprod()
        pas['Cum_P&L'] = pas['Total_P&L'].cumsum()
        pas['Cum_P&L_Long'] = pas['P&L_Long'].cumsum()
        pas['Cum_P&L_Short'] = pas['P&L_Short'].cumsum()
        
        self.PA_snapshots_pure_cap_return = pas.drop(columns = ['Total_Long', 'Total_Short', 'Total_Options'])
        
        paspcr = self.PA_snapshots_pure_cap_return
        pas = self.PA_snapshots
        
        pas['Cash'] = pas['Cumulative_Capital_Flow'] + pas['Cumulative_Dividends'] + paspcr['Currency_Long'] + paspcr['Currency_Short']  + paspcr['Currency_Options']
        
        pas['Balance_EOD'] = paspcr['Equity_Long'] + paspcr['Equity_Short'] + paspcr['Options'] + pas['Cash']
        pas['Daily_P&L'] = self.PA_snapshots_pure_cap_return['Total_P&L']
        pas['Daily_Return'] = pas['Daily_P&L'] / (pas['Balance_EOD'].shift(1))
        pas.loc[self.start_date, 'Daily_Return'] = 0
        pas['Port_NAV'] = (1 + pas['Daily_Return']).cumprod()
        pas['Cum_P&L'] = pas['Daily_P&L'].cumsum()
            
         
        pas['Benchmark_Return'] = self.equity_daily_return_local[self.benchmark]
        pas['Benchmark_Normalized'] = self.equity_prices_local[self.benchmark] / self.equity_prices_local.loc[self.start_date, self.benchmark]

        if len(self.benchmark2)>1:
            pas['Benchmark2_Return'] = self.equity_daily_return_local[self.benchmark2[1]]
            pas['Benchmark2_Normalized'] = self.equity_prices_local[self.benchmark2[1]] / self.equity_prices_local.loc[self.start_date, self.benchmark2[1]]


        
        self.__sampling_portfolio_return__()
        

        self.net_deposit = self.PA_snapshots.iloc[-1]['Cumulative_Capital_Flow'] - self.PA_snapshots.iloc[0]['Cumulative_Capital_Flow']

        if self.capital == None:
            self.capital = pas.loc[self.start_date, 'Balance_EOD'] + self.net_deposit
            print()
            print("Capital unspecified. Using {} {:,.2f}, which is beginning assets plus net deposit.".format(self.base_currency, self.capital))

        pas['Gross_Exposure'] = (paspcr['Equity_Long'] - paspcr['Equity_Short'] + paspcr['Options']) / (self.capital + paspcr['Cum_P&L'])
        pas['Net_Exposure'] = (paspcr['Equity_Long'] + paspcr['Equity_Short'] + paspcr['Options']) / (self.capital + paspcr['Cum_P&L'])
        
        
        pas['Capital_Deployment'] = - (self.equity_and_ETF_trades.groupby(['Date'])['TransactionAmountUSD'].sum())        
        pas.loc[self.start_date, 'Capital_Deployment'] = pas.loc[self.start_date, 'Balance_EOD']
        pas['Capital_Deployment'] = pas['Capital_Deployment'].fillna(0)
        pas['Capital_Deployment'] = pas['Capital_Deployment'].cumsum()
        

        self.metrics['P&L composition'] = {}
        
        
        self.metrics['P&L composition']['Long'] = self.PA_snapshots_pure_cap_return['P&L_Long'].iloc[1:].sum(axis=0)
        self.metrics['P&L composition']['Short'] = self.PA_snapshots_pure_cap_return['P&L_Short'].iloc[1:].sum(axis=0)
        self.metrics['P&L composition']['Options'] = self.PA_snapshots_pure_cap_return['P&L_Options'].iloc[1:].sum(axis=0)
        self.metrics['P&L'] = self.metrics['P&L composition']['Long'] + self.metrics['P&L composition']['Short'] + self.metrics['P&L composition']['Options']
        self.latest_capital = self.capital + self.metrics['P&L']
        
        self.portfolio_PnL_FX = self.currency_PnL.values.sum()
        
        self.metrics['Return'] = self.metrics['P&L'] / self.capital
        self.metrics['Return composition'] = {}
        
        
        for long_short in ['Long', 'Short', 'Options']:
            self.metrics['Return composition'][long_short] = self.metrics['P&L composition'][long_short] / self.capital
        
                

        
        
        #Checksum Logic        
        '''
        Total equity long P&L in PA_Snapshots_Pure_Cap should be equal to the sum of each long equity's P&L, plus the FX-caused P&L from the currency balance 
        '''
        for long_short in ['Long', 'Short']:
            a = pd.DataFrame(self.equity_PnL[long_short].sum(axis = 1))
            a.columns = ['P&L_sum_of_equities']
            b = a.join(self.PA_snapshots_pure_cap_return['P&L_' + long_short])
            b['Err'] = b['P&L_sum_of_equities'] + self.currency_PnL[long_short] - b['P&L_' + long_short]
            if helpers.isZero(b['Err'].values.sum()) == False:
                warnings.warn('Total P&L is not equal to sum of individual stocks and FX-caused currency P&L')
                print(b)                    
        #End of checksum logic
                
        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])
    
    
    def __sampling_portfolio_return__(self):
        if len(self.benchmark2) <=1:
            pas = self.PA_snapshots[['Daily_Return','Benchmark_Return']]
        else:
            pas = self.PA_snapshots[['Daily_Return','Benchmark_Return','Benchmark2_Return']]
        pas = pas.iloc[1:]

        pnl = self.PA_snapshots[['Daily_P&L']]
        pnl = pnl.iloc[1:]

        fs = ['A', 'Q', 'M', 'W-SAT']
        def pas_resample(pas,fs):
            for f in fs:
                p = pas.resample(f).apply(helpers.growth_resampler)
                p['PnL'] = pnl.resample(f).sum()
                p['Frequency'] = f
                yield p
                
        self.PA_snapshots_sampled_return = pd.concat(pas_resample(pas, fs))
        self.PA_snapshots_sampled_return.rename(columns = {'Daily_Return':'Time_Weighted_Port_Return'}, inplace=True)
                
    
    def __prepare_current_holdings_and_exposure__(self):        

        if self.__debug_mode__: print('Entering', inspect.stack()[0][3]) 
        
        '''
        Calculate current holdings and exposure
        '''
        self.current_holdings = self.holdings_snapshot_of_date(self.end_date)
        self.current_exposure = {}


        for long_short in ['Long', 'Short']: #self.long_short_list:
           
            
            self.current_exposure[long_short] = self.current_holdings[long_short].copy()
            
            
            
            '''
            "SinceMin" and "SinceMax" are within the target time period for the analysis
            "SinceHighestBuy" and "SinceLatestBuy" are based on trade history, thus the trades may happen before the target time period for the analysis
            '''
            
            
            dp = self.equity_prices_USD.loc[:, self.current_holdings[long_short].index]
            dmin = pd.DataFrame(dp.min())
            dmin.columns = ['Min']
            dmax = pd.DataFrame(dp.max())
            dmax.columns = ['Max']
            dlast = pd.DataFrame(dp.iloc[-1])
            dlast.columns = ['Last']
            self.current_holdings[long_short] = self.current_holdings[long_short].join(dmin)
            self.current_holdings[long_short] = self.current_holdings[long_short].join(dmax)
            self.current_holdings[long_short] = self.current_holdings[long_short].join(dlast)
            self.current_holdings[long_short]['SinceMin'] = self.current_holdings[long_short]['Last'] / self.current_holdings[long_short]['Min'] -1
            self.current_holdings[long_short]['SinceMax'] = self.current_holdings[long_short]['Last'] / self.current_holdings[long_short]['Max'] -1
            
            
        #end of for loop: for long_short in self.long_short_list
        
        
        dlastBuyPrice = (pd.Series(self.lastBuyPrice))[self.current_holdings['Long'].index]
        dlastBuyPrice.name = 'LatestBuy'
        self.current_holdings['Long'] = self.current_holdings['Long'].join(dlastBuyPrice)
        
        self.hightestBuyPrice = self.hightestBuyPrice[self.current_holdings['Long'].index]
        self.hightestBuyPrice.name = 'HighestBuy'
        self.current_holdings['Long'] = self.current_holdings['Long'].join(self.hightestBuyPrice)
        
        
        self.current_holdings['Long']['SinceLatestBuy'] = self.current_holdings['Long']['Last'] / self.current_holdings['Long']['LatestBuy'] - 1
        self.current_holdings['Long']['SinceHighestBuy'] = self.current_holdings['Long']['Last'] / self.current_holdings['Long']['HighestBuy'] - 1
        
        
        dlastSellPrice = (pd.Series(self.lastSellPrice))[self.current_holdings['Short'].index]
        dlastSellPrice.name = 'LatestSell'
        self.current_holdings['Short'] = self.current_holdings['Short'].join(dlastSellPrice)
        
        self.lowestSellPrice = self.lowestSellPrice[self.current_holdings['Short'].index]
        self.lowestSellPrice.name = 'LowestSell'
        self.current_holdings['Short'] = self.current_holdings['Short'].join(self.lowestSellPrice)
        
        self.current_holdings['Short']['SinceLatestSell'] = self.current_holdings['Short']['Last'] / self.current_holdings['Short']['LatestSell'] - 1
        self.current_holdings['Short']['SinceLowestSell'] = self.current_holdings['Short']['Last'] / self.current_holdings['Short']['LowestSell'] - 1


            
            
        
        
        '''
        Deal with options
        '''
        if self.has_option_book:
            '''
            last_row = self.option_balance.loc[self.end_date]
            option_holdings = pd.DataFrame(last_row[last_row!=0])
            option_holdings['Share#'] = option_holdings[self.end_date]  # don't know how to rename the column
            option_holdings['Size'] = self.option_intrinsic_value.loc[self.end_date,option_holdings.index]
            option_holdings = option_holdings.drop(columns = [self.end_date])
            
            option_exposure = option_holdings.copy()
            option_exposure['Share#'] *= 100
            for ticker in option_exposure.index.values:
                tk, expiration, strike, tp = finance.optionDecompose(ticker)
                if (tp in ['PUT', 'Put', 'put', 'p', 'C']): option_exposure.at[ticker, 'Share#'] *= -1
                option_exposure.at[ticker, 'Size'] = option_exposure.at[ticker, 'Share#'] * self.equity_prices_USD.at[self.end_date, tk]
                option_exposure.at[ticker, 'Ticker'] = tk+'*'
                                
            
            option_long = (option_holdings[option_holdings['Share#']>0]).sort_values(by=['Size'], ascending = False)
            option_short = (option_holdings[option_holdings['Share#']<0]).sort_values(by=['Size'], ascending = True)


            
            self.current_holdings['Long'] = pd.concat([self.current_holdings['Long'], option_long], sort=False)
            self.current_holdings['Short'] = pd.concat([self.current_holdings['Short'], option_short], sort=False)
                
            
            if( option_exposure.shape[0] >0): # make sure it's not empty; shape = (row_count, column_count)
                option_exposure = option_exposure.groupby(['Ticker']).sum()
            
            option_long = (option_exposure[option_exposure['Share#']>0]).sort_values(by=['Size'], ascending = False)       
            option_short = (option_exposure[option_exposure['Share#']<0]).sort_values(by=['Size'], ascending = True)
            self.current_exposure['Long'] = pd.concat([self.current_exposure['Long'], option_long])
            self.current_exposure['Short'] = pd.concat([self.current_exposure['Short'], option_short])

            for long_short in ['Long', 'Short']:
                self.current_holdings[long_short].index.name = 'Ticker'
                self.current_exposure[long_short].index.name = 'Ticker'
            '''
        
        
        #end if self.has_option_book:

        #the below 2 sections will need to be incorporated into the above options section
        '''
        for long_short in ['Long', 'Short']: #self.long_short_list:
            self.current_holdings[long_short] = self.__add_company_name__(self.current_holdings[long_short])
       
            df = self.current_holdings[long_short]
            x = (df.isna()).any(axis = 1)
            df.loc[x,'Name'] = df[x].index
        


                    
        for long_short in ['Long', 'Short']: #self.long_short_list:
            self.current_holdings[long_short]['Weight'] = self.current_holdings[long_short]['Size'] / self.capital
            self.current_exposure[long_short]['Weight'] = self.current_exposure[long_short]['Size'] / self.capital
            self.current_holdings[long_short]['CumPnL'] = self.equity_PnL_cumulative[long_short].iloc[-1]

        '''
        
        # re-order columns
        self.current_holdings['Long'] = self.current_holdings['Long'][['Share#', 'Size', 'Weight', 'Name','SinceMin','SinceMax','SinceLatestBuy','SinceHighestBuy', 'CumPnL', 'LastPx', 'PxChg']]        
        self.current_holdings['Short'] = self.current_holdings['Short'][['Share#', 'Size', 'Weight','Name', 'SinceMin','SinceMax','SinceLatestSell','SinceLowestSell','CumPnL', 'LastPx','PxChg']]
        

        for long_short in ['Long', 'Short']:#self.long_short_list:
            self.current_holdings[long_short] = self.current_holdings[long_short].fillna(0.0)
            self.current_exposure[long_short] = self.current_exposure[long_short][['Share#', 'Size', 'Weight']]                
        
        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])

    
    def holdings_snapshot_of_date(self, date, exited_positions = True, zeroPnLPositions = False, weight = True, last_px = True, px_chg = True, name = True, cum_pnl = True, start_date = None, sort_by = None):        
        holdings = {}

        if start_date is None: start_date = self.start_date

        if last_px:
            px = pd.DataFrame(self.equity_prices_local.loc[date])
            px.columns = ['LastPx']
        
        if px_chg:
            pxchg = pd.DataFrame(self.equity_prices_local.loc[date] / self.equity_prices_local.loc[start_date] - 1)
            pxchg.columns = ['PxChg']


        for long_short in self.long_short_options:
            row = self.equity_balance[long_short].loc[date]
            
            
            if exited_positions:
                holdings[long_short] = pd.DataFrame(row)
            else:
                holdings[long_short] = pd.DataFrame(row[row!=0])
            
            holdings[long_short].rename(columns = {holdings[long_short].columns[0]: 'Share#'}, inplace = True)
            holdings[long_short]['Size'] = self.equity_value_USD[long_short].loc[date, holdings[long_short].index]
            holdings[long_short].index.name = 'Ticker'

            if weight:
                holdings[long_short]['Weight'] = holdings[long_short]['Size'] / self.PA_snapshots.loc[date,'Balance_EOD']

            if last_px:
                holdings[long_short] = pd.merge(holdings[long_short],px, left_index = True, right_index = True, how = 'left')

            if px_chg:
                holdings[long_short] = pd.merge(holdings[long_short], pxchg, left_index = True, right_index = True, how = 'left')


            if name:
                holdings[long_short] = self.__add_company_name__(holdings[long_short])
            
            if cum_pnl:
                holdings[long_short]['CumPnL'] = self.equity_PnL_cumulative[long_short].loc[date]
                if start_date != self.start_date:
                    holdings[long_short]['CumPnL'] -= self.equity_PnL_cumulative[long_short].loc[start_date]

                if zeroPnLPositions is not True:
                    holdings[long_short] = holdings[long_short][holdings[long_short]['CumPnL'] != 0.0]

        
        if sort_by is None:
            holdings['Long'].sort_values('Size', ascending = False, inplace = True)
            holdings['Short'].sort_values('Size', ascending = True, inplace = True)
        else:
            holdings['Long'].sort_values(sort_by, ascending = False, inplace = True)
            holdings['Short'].sort_values(sort_by, ascending = False, inplace = True)

        
        if self.has_option_book:
            a = pd.DataFrame(self.option_intrinsic_value.loc[date]).rename(columns={date:'Intrinsic Value'})
            b = pd.concat([holdings['Options'],self.option_dict,a],axis = 1)
            b = b.sort_values(['Underlying','Exp']).reset_index().set_index(['Underlying','Ticker'])
            holdings['Options'] = b.drop(columns = ['Exp','Strike','Type'])
        

        return holdings

    
    def __init_date_indexing__(self):
        self.date_list = list(self.equity_prices_USD.index)
        self.date_dict = {}
        for i, d in enumerate(self.date_list):
            self.date_dict[d] = i

    
    def __calculate_dividends__(self, start_date = None, end_date = None):
        start_date, end_date = self.assign_start_and_end_date(start_date, end_date)
        self.metrics['Dvd long'] = self.PA_snapshots_other.loc[end_date, 'Cumulative_Dividends_Long'] - self.PA_snapshots_other.loc[start_date, 'Cumulative_Dividends_Long']
        self.metrics['Dvd short'] = self.PA_snapshots_other.loc[end_date, 'Cumulative_Dividends_Short'] - self.PA_snapshots_other.loc[start_date, 'Cumulative_Dividends_Short']

    
    def __calculate_interest__(self, start_date = None, end_date = None):
        start_date, end_date = self.assign_start_and_end_date(start_date, end_date)
        self.metrics['Short interest'] = self.PA_snapshots_other.loc[end_date, 'Cumulative_Short_Interest'] - self.PA_snapshots_other.loc[start_date, 'Cumulative_Short_Interest']
        self.metrics['Credit interest'] = self.PA_snapshots_other.loc[end_date, 'Cumulative_Credit_Interest'] - self.PA_snapshots_other.loc[start_date, 'Cumulative_Credit_Interest']

    def assign_start_and_end_date(self, start_date, end_date):
        if start_date is None:
            start_date = self.start_date
        else:
            start_date = helpers.string_to_date(start_date)
        
        if end_date is None:
            end_date = self.end_date
        else:
            end_date = helpers.string_to_date(end_date)
            
        return start_date, end_date


    def __add_company_name__(self, df, ticker_is_index=True):
        if ticker_is_index:
            return df.join(self.company_name)
        else:
            return df.join(self.company_name, on='Ticker')

    def __replace_company_name__(self, df, keep_Ticker_as_index_name = True):
        r = self.__add_company_name__(df)
        r.reset_index(drop=True, inplace=True)
        r.set_index(['Name'], inplace=True)
        if keep_Ticker_as_index_name: r.index.name = 'Ticker'
        return r

    def getBookBalance(self, date, print_out = False):
        
        b_equity_long = self.PA_snapshots_pure_cap_return.loc[date,'Equity_Long']
        b_equity_short = self.PA_snapshots_pure_cap_return.loc[date,'Equity_Short']
        b_options = self.PA_snapshots_pure_cap_return.loc[date,'Options']
        b_cash = self.PA_snapshots.loc[date,'Cash']
        b_total = self.PA_snapshots.loc[date,'Balance_EOD']
        
        if print_out:
            print()
            print('Balance on', date.strftime('%Y-%m-%d'))
            print('  Equity long: {:,.2f}'.format(b_equity_long))
            print('  Equity short: {:,.2f}'.format(b_equity_short))
            print('  Options: {:,.2f}'.format(b_options))
            print('  Cash: {:,.2f}'.format(b_cash))
            print('  Total asset: {:,.2f}'.format(b_total))
        
        return b_equity_long, b_equity_short, b_options, b_cash, b_total


    def __calculate_vol_corr__(self):
        s = self.PA_snapshots[["Daily_Return","Benchmark_Return","Benchmark2_Return"]].iloc[1:]
        s['Valid'] = True
        for i in s.index:
            if s.loc[i, 'Daily_Return'] == 0.0 and s.loc[i, 'Benchmark_Return'] == 0.0 and s.loc[i, 'Benchmark2_Return'] == 0.0:
                s.loc[i, 'Valid'] = False
        s = s[s['Valid'] == True]

        v = s.std()*np.sqrt(len(s.index))
        
        self.metrics['Vol'] = v['Daily_Return']

        self.metrics['Benchmark vol'] = {}
        self.metrics['Benchmark vol'][self.benchmark2[0]] = v['Benchmark_Return']
        self.metrics['Benchmark vol'][self.benchmark2[1]] = v['Benchmark2_Return']
        
        self.metrics['Corr'] = {}
        self.metrics['Corr'][self.benchmark2[0]] = s['Daily_Return'].corr(s['Benchmark_Return'])
        self.metrics['Corr'][self.benchmark2[1]] = s['Daily_Return'].corr(s['Benchmark2_Return'])
    
    def print_metrics(self, metric):
        if metric == 'Vol':
            if len(self.benchmark2)>1:
                print('The annualized vol of the portfolio is {:.1%}, while {} and {} are at {:.1%} and {:.1%}, respectively'.format(self.metrics['Vol'], self.benchmark2[0], self.benchmark2[1], self.metrics['Benchmark vol'][self.benchmark2[0]], self.metrics['Benchmark vol'][self.benchmark2[1]]))
            else:
                print('The annualized vol of the portfolio is {:.1%}, while {} is at {:.1%} '.format(self.metrics['Vol'], self.benchmark2[0],  self.metrics['Benchmark vol'][self.benchmark2[0]]))
            return True
        elif metric == 'Corr':
            if len(self.benchmark2)>1:
                print('The correlation with {} and {} are {:.1%} and {:.1%}, respectively'.format(self.benchmark2[0], self.benchmark2[1], self.metrics['Corr'][self.benchmark2[0]], self.metrics['Corr'][self.benchmark2[1]]))
            else:
                print('The correlation with {} is {:.1%}'.format(self.benchmark2[0],  self.metrics['Corr'][self.benchmark2[0]]))
            return True
        elif metric == 'P&L composition':
            self.__print_PnL_composition__(**(self.metrics['P&L composition']))
            return True
        else:
            return False

    def __print_PnL_composition__(self, Long, Short, Options, date=None):
        if date is None: 
            capital = self.capital
        else:
            capital = self.PA_snapshots.loc[date,'Balance_EOD']
        
        print('Total P&L: {} {:,.0f}, or {:.2%}, where:'.format(self.base_currency, Long+Short+Options, (Long+Short+Options)/capital))
        print('  Long P&L: {} {:,.0f}, or {:.2%}'.format(self.base_currency, Long, Long/capital))
        if helpers.isNotZero(Short): print('  Short P&L: {} {:,.0f}, or {:.2%}'.format(self.base_currency, Short, Short/capital))
        if helpers.isNotZero(Options): print('  Option P&L: {} {:,.0f}, or {:.2%}'.format(self.base_currency, Options, Options/capital))
