'''
Created on May 18, 2019

author: yewen
'''

import pandas as pd
import numpy as np

from PMStack.Tools import finance, helpers, excel_writer
import inspect
from PMStack.Equity_port import portfolio_print


class Portfolio_Write(portfolio_print.Portfolio_Print):
    '''
    Adding capability to export to file
    '''

    def __init__(self, file_price, file_trades, start_date, end_date=None, capital=None, benchmark = 'SPX', benchmark2=[], base_currency = 'USD', ignore_ticker_list = [], file_stock_info = None, output_dir = '' ):
        super().__init__(file_price, file_trades, start_date, end_date, capital, benchmark, benchmark2, base_currency, ignore_ticker_list, file_stock_info)
        self.writer = excel_writer.ExcelWriter(output_dir = output_dir)

    def save_analysis_to_file(self, porfolio_filename = 'output', perf_by_equity_filename = 'output1', perf_attribution_filename = None):
        self.write_portfolio(porfolio_filename)
        self.write_perf_by_equity(perf_by_equity_filename)
        
        if perf_attribution_filename != None:
            self.write_perf_attribution(perf_attribution_filename)
        
        self.writer.save()
            

    def write_portfolio(self, filename='output'):        
                
    
        dfs = [self.PA_snapshots, self.PA_snapshots_sampled_return, self.PA_snapshots_pure_cap_return ]
        sheetnames = ['Snapshots', 'Snapshots_Sampled_R', 'Snapshots_Pure_Cap']
        
        for long_short in ['Long', 'Short']: #self.long_short_list:    
            
            dfs.extend([self.current_holdings[long_short], self.current_exposure[long_short]])
            sheetnames.extend(['Holdings_'+long_short, 'Exposure_'+long_short])
            
            dfs.extend([self.equity_balance[long_short], self.equity_value_USD[long_short], self.equity_PnL[long_short], self.currency_balance[long_short]])
            sheetnames.extend(['Equity_balance_' + long_short, 'Equity_value_USD_' + long_short, 'Equity_P&L_' + long_short, 'Currency_balance_' + long_short])
                
        # end of for loop for long_short in self.long_short_list:
                    
        dfs.extend([self.option_balance, self.option_intrinsic_value, self.option_PnL, self.currency_balance_options, self.currency_PnL])
        sheetnames.extend(['Option_Balance', 'Option_Intrinsic_Value', 'Option_P&L', 'Currency_balance_options', 'Currency_P&L'])
        
        self.__print_to_excel__(filename, dfs, sheetnames)
        

        dfs = []
        sheetnames = []
        for long_short in self.long_short_list:

            df = helpers.excel_tables_to_relational_tables(tables=[self.equity_PnL_cumulative[long_short]], table_names=['PnL_cum'], index_name='Date', column_name='Ticker', new_index_name='Date')
            

            df = self.__add_company_name__(df, ticker_is_index=False)
            dfs.extend([df])
            sheetnames.extend(['PnL_Cum_'+long_short])
        self.__print_to_excel__(filename, dfs, sheetnames, mimic_short=self.long_only)
        
        
    def write_perf_by_equity(self, filename = 'output1'):
        
        if self.statusAnalyzedPnLbyEquity == False:
            self.__analyze_perf_by_equity__()
        
        for key, value in self.equity_PnL_periodic_summary.items():
            self.__print_to_excel__(filename, [value], [key], mimic_short = self.long_only)
                    
        self.__print_to_excel__(filename, [self.bats], ['Bats'])

    def write_perf_attribution(self, filename = 'output_perf_attribution'):
        
        if self.statusPerformanceAttributed == False:
            self.__attribute_performance__()

        pcbe = self.positioning_contribution_by_equity
        scbe = self.selection_contribution_by_equity
        abe = self.attribution_by_equity
        papa = self.PA_PerformanceAttribution
        pra = self.portfolio_return_attribution

        self.writer.print(filename,[pra, papa],['Summary', 'Portfolio'])

        for long_short in ['Long', 'Short']:  
            dfs = [abe[long_short], pcbe[long_short], scbe[long_short]]
            sheetnames = ['Attribution_'+long_short, 'Positioning_'+long_short, 'Selection_'+long_short]
            self.writer.print(filename, dfs, sheetnames)

        

    def __print_to_excel__(self, filename, dfs, sheetnames, mimic_short=False):

        if mimic_short:
            names = sheetnames.copy()
            for idx, name in enumerate(names):
                if 'Long' in name:
                    new_name = name.replace('Long', 'Short')
                    df = pd.DataFrame(columns = dfs[idx].columns.values)
                    df.index.name = dfs[idx].index.name
                    dfs.append(df)
                    sheetnames.append(new_name)
            
        self.writer.print(filename, dfs, sheetnames)

    def export_port_for_BBU_from_date(self, portfolio_name, start_date, separate_long_short = True):
        dates = list(self.equity_prices_USD.loc[start_date:].index)
        return self.export_port_for_BBU(portfolio_name, dates, separate_long_short)
    
    
    def export_port_for_BBU(self, portfolio_name, dates = None, separate_long_short = True):
        if dates == None: dates = [self.end_date]
        filename = "BBU upload.xls"
        print("writing portfolio to", filename)
        
        cc = []
        
        for date in dates:
            
            hh = self.holdings_snapshot_of_date(date)
            
            a = hh['Long'][['Share#']].join(self.stock_info[['BBG Ticker']])
            b = hh['Short'][['Share#']].join(self.stock_info[['BBG Ticker']])
            
            a['Portfolio'] = portfolio_name
            b['Portfolio'] = portfolio_name
            
            if separate_long_short:
                aa = a.copy()
                bb = b.copy()
                aa['Portfolio'] = portfolio_name + '-LONG'
                bb['Portfolio'] = portfolio_name + '-SHORT'
                c = pd.concat([a,b,aa,bb], axis=0)
            else:                 
                c = pd.concat([a,b], axis=0)
            c['Date'] = helpers.date_to_string(date)
            cc.append(c)
        
        c = pd.concat(cc, axis=0)    
        c.reset_index(inplace = True)
        c.drop(columns=['Ticker'], inplace = True)
        c.set_index('BBG Ticker', inplace = True)
        c.to_excel(filename)
