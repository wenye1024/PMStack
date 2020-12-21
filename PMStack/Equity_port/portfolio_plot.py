'''
Created on Feb 16, 2019

author: yewen
'''

import pandas as pd
import numpy as np
import PMStack.Tools.finance as finance
import PMStack.Tools.helpers as helpers
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pandas.plotting import register_matplotlib_converters
from datetime import timedelta
from PMStack.Equity_port import portfolio_perf, historical_price

class Portfolio_Plot(portfolio_perf.Portfolio_Perf):
    '''
    Adding plotting capabilities on top of Portfolio_Perf
    '''
    def __init__(self, file_price, file_trades, start_date, end_date=None, capital=None, benchmark = 'SPX', benchmark2=[], base_currency = 'USD', ignore_ticker_list = [], file_stock_info = None ):
        super().__init__(file_price, file_trades, start_date, end_date, capital, benchmark, benchmark2, base_currency, ignore_ticker_list, file_stock_info)
        register_matplotlib_converters()
        plt.rcParams['figure.figsize'] = [16, 9]
        plt.rcParams['lines.linewidth'] = 1.0
        #plt.style.use('dark_background')


    def plotPrice(self, ticker, start_date = None, end_date = None, color = 'C0', show = False, plot_type = 'line', annotate_trades = 0 ):

        if start_date is None:
            start_date = self.start_date
        
        if end_date is None:
            end_date = self.end_date
        
        if annotate_trades>0:
            annotationss = historical_price.generate_trade_annotation(self.trades_full_history.loc[start_date:end_date+timedelta(1)], [ticker], annotate_trades, equity_currency=self.equity_currency, forex = self.forex)
        else:
            annotationss = None
        
        historical_price.plotPrice(self.equity_prices_local_full_history, [ticker], names=self.company_name, start_date = start_date, end_date = end_date, colors = [color], show = show, plot_type = plot_type, annotationss=annotationss)


        '''
        if start_date == None:
            x = self.equity_prices_local.index
            y = self.equity_prices_local[ticker]            
        else:
            if end_date == None:
                end_date = self.end_date
                
            x = self.equity_prices_local_full_history.loc[start_date:end_date].index
            y = self.equity_prices_local_full_history.loc[start_date:end_date, ticker]    
        
        if plot_type == 'line':
            
            plt.plot(x, y, color = color, label = ticker)
        else:
            low = min(y)
            high = max(y)
            plt.ylim([np.ceil(low-0.5*(high-low)), np.ceil(high+0.5*(high-low))])
            
            if plot_type == 'area': plt.fill_between(x, y, color = color, alpha = 0.2, label = ticker)
            elif plot_type == 'bar': plt.bar(x, y, color = color, alpha = 0.2, label = ticker)
        
        if show:
            plt.show()
        '''

    def plotEquityPriceAsBackground(self, ticker, start_date = None, end_date = None, color = 'C9', show = False):
        self.plotPrice(ticker, start_date, end_date, color, show, plot_type = 'area')


    def plotBenchmarkAsBackground(self, start_date = None, end_date = None, color = 'C9', show = False):
         self.plotEquityPriceAsBackground(self.benchmark, start_date, end_date, color, show)


    def plotPortfolioConcentration(self, long_short = 'Long', x = 8):
        ev = self.equity_value_USD[long_short]
        if long_short == 'Short': ev = -ev
        t = ev.sum(1)
        t.name='Total'
        r = pd.DataFrame(t)
            
        top_x = np.partition(ev.values, -x)[:, -x:]
        str_top_x = 'Top '+ str(x)
        str_top_x_pct = str_top_x +' %'

        r[str_top_x] = np.sum(top_x, 1)
        r[str_top_x_pct] = r[str_top_x] / r['Total']

        ax = r[str_top_x_pct].plot()

        ymax = max(1.1, r[str_top_x_pct].max() + 0.1)
        ymin = min(0.5, r[str_top_x_pct].min() - 0.1)

        plt.ylim(ymin, ymax)
        plt.title('Concentration of ' + str_top_x + ' among all Positions in ' + long_short + ' Book')
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

        plt.show()





    def plotPortfolioExposure(self):
        self.__plot_equity_line_chart_with_price_background__(self.benchmark, self.PA_snapshots, 'Portfolio exposure', tickers = ['Gross_Exposure','Net_Exposure'], labels = ['Gross','Net'])


    def plotPortfolioNAV(self, include_long_only_NAV = False):
        
        plt.title('Portfolio NAV vs. Benchmark')
        plt.plot(self.PA_snapshots.index, self.PA_snapshots['Port_NAV'], color='C0', label = 'Portfolio')
        
        if include_long_only_NAV:
            plt.plot(self.PA_snapshots_pure_cap_return.index, self.PA_snapshots_pure_cap_return['NAV_Long'], linewidth=1, color='C1', label = 'Long book')
       
        plt.plot(self.PA_snapshots.index, self.PA_snapshots['Benchmark_Normalized'], color='C2', label = self.benchmark)

        plt.legend(loc='upper left')
        plt.show()
        

    def plotPortfolioPnL(self):
                
        plt.title('Cumulative P&L')
        plt.fill_between(self.PA_snapshots_pure_cap_return.index, self.PA_snapshots_pure_cap_return['Cum_P&L_Long'], color='C0', label = 'Long', alpha = 0.2)
        plt.fill_between(self.PA_snapshots_pure_cap_return.index, self.PA_snapshots_pure_cap_return['Cum_P&L_Short'], color='C1', label = 'Short', alpha = 0.2)
        plt.plot(self.PA_snapshots.index, self.PA_snapshots['Cum_P&L'], color='C2', label = 'Total')
        plt.legend(loc='upper left')
                
        plt.show()


    def plotPortfolioCapitalDeployment(self):

        a = self.equity_and_ETF_trades.groupby(['Long_Short','Date'])['TransactionAmountUSD'].sum()
        d = pd.DataFrame(self.PA_snapshots['Capital_Deployment'])
        d['Cumulative'] = d['Capital_Deployment']
        for long_short in ['Long','Short']:
            d[long_short + ' flow'] = -a[long_short]
            d[long_short + ' flow'] = d[long_short + ' flow'].fillna(0)
            d[long_short + ' flow'] = d[long_short + ' flow'].cumsum()
            
        self.__plot_equity_line_chart_with_price_background__(self.benchmark, d, 'Capital Deployment', tickers = ['Cumulative','Long flow', 'Short flow'])

  



    def plotPriceLTM(self, ticker, end_date = None, color = 'C0', show = False, annotate_trades = 0  ):
        if end_date == None: end_date = self.end_date
        start_date = end_date + pd.DateOffset(-365)
        self.plotPrice(ticker, start_date, end_date, color, show, annotate_trades = annotate_trades)



    def __plot_equity_line_chart_with_price_background__(self, background_ticker, series, chart_name, tickers = None, labels = None):
        plt.title(chart_name)
        self.plotEquityPriceAsBackground(background_ticker)
        
        ax2 = plt.twinx()
        if tickers is None:
            ax2.plot(series.index, series, color='C1', label = background_ticker)
        else:
            for i, ticker in enumerate(tickers):
                ax2.plot(series.index, series[ticker], color='C'+str(i), label = ticker if labels is None else labels[i])
        plt.legend(loc='upper right')
        plt.show()
        
    def plotEquityPnL(self, ticker, long_short = 'Long'):
        self.__plot_equity_line_chart_with_price_background__(ticker, self.equity_PnL_cumulative[long_short][ticker], 'Cumulative P&L')
        
    
    def plotEquityExposure(self, ticker, long_short = 'Long'):
        self.__plot_equity_line_chart_with_price_background__( ticker, self.equity_value_USD[long_short][ticker], 'Equity Exposure')
        
    
    def plotEquityCapitalDeployment(self, ticker, long_short = 'Long'):
        d = self.equity_value_USD[long_short][ticker].iloc[0] + self.equity_cumulative_cost_USD[long_short][ticker].iloc[0]
        d = - self.equity_cumulative_cost_USD[long_short][ticker] + d
        self.__plot_equity_line_chart_with_price_background__( ticker, d, 'Capital deployment')
    

    def plotEquity(self, ticker, long_short = 'Long', plot_trades = False):
        self.plotEquityPnL(ticker, long_short)
        self.plotEquityCapitalDeployment(ticker, long_short)
        self.plotEquityExposure(ticker, long_short)
        if plot_trades:
            self.plotPrice(ticker, annotate_trades = 10)

    
    def plotEquityPortfolioTrades(self):
        self.plotEquityTrades(['Equity Portfolio'])
    
    def plotEquityTrades(self, tickers):
        c = len(tickers)
        if c == 1:
            a,b = 1,1
        elif c == 2:
            a,b = 2,1
        elif c<=4:
            a,b = 2,2
        elif c<=6:
            a,b = 2,3
        else:
            print("Too many tickers")
            return

        i = 0
        for ticker in tickers:
            i += 1
            plt.subplot(a,b,i)
            
            
            if ticker == 'Equity Portfolio':
                plt.title('Equity portfolio trades')
                plt.plot(self.equity_prices_local.index, self.equity_prices_local[self.benchmark], linewidth=1, color='white')
            else:
                plt.title(ticker)
                plt.plot(self.equity_prices_local.index, self.equity_prices_local[ticker], linewidth=1, color='white')
                
            

            
            if ticker == 'Equity Portfolio':
                trades = self.equity_and_ETF_trades[self.equity_and_ETF_trades['SecurityType'] != 'ETF']
            else:
                trades = self.equity_and_ETF_trades[self.equity_and_ETF_trades['Ticker']==ticker]
                #print(trades)
                
                
            trades = trades.groupby(['Date'])['TransactionAmountUSD','Quantity'].sum()
            trades['BuyAmount'] = -trades[trades['TransactionAmountUSD']<0]['TransactionAmountUSD']
            trades['SellAmount'] = trades[trades['TransactionAmountUSD']>0]['TransactionAmountUSD']
                    
            #print(trades)
            ax2 = plt.twinx()
            ax2.bar(trades.index, trades['BuyAmount'], width=1, color='green')
            ax2.bar(trades.index, trades['SellAmount'], width=1, color='red')
                

        #end for ticker in tickers
        plt.show()
        
    def plotEquityTradesMarketTimingHist(self):
        if self.statusAnalyzedMarketTiming == False:
            self.__analyzeEquityPortfolioMarketTiming__()

        
        t = self.equity_market_timing
        bins = int(round((t[self.benchmark].max() - t[self.benchmark].min())/(t.iloc[0][self.benchmark]) * 100))

        if self.long_only:
            l = ['Total']
        else:
            l = ['Total', 'Long', 'Short']
        for idx, long_short in enumerate(l):
            plt.subplot(2,3,idx+1)
            plt.title(long_short)
            plt.hist(t[self.benchmark], bins=bins, weights=t[long_short+'BuyAmount'], color='green')
            plt.subplot(2,3,idx+4)
            plt.hist(t[self.benchmark], bins=bins, weights=t[long_short+'SellAmount'], color='red')
        
        plt.show()



