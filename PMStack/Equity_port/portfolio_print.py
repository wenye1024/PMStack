'''
Created on Feb 17, 2019

author: yewen
'''


import pandas as pd
import numpy as np
import PMStack.Tools.finance as finance
import PMStack.Tools.helpers as helpers
import PMStack.Equity_port.price_actions as price_actions
import inspect
from PMStack.Equity_port import portfolio_plot
from IPython.display import display


class Portfolio_Print(portfolio_plot.Portfolio_Plot):
    '''
        Adding more printing capabilities on top of Portfolio_Plot
    '''


    def __init__(self, file_price, file_trades, start_date, end_date=None, capital=None, benchmark = 'SPX', benchmark2=[], base_currency = 'USD', ignore_ticker_list = [], file_stock_info = None):
        super().__init__(file_price, file_trades, start_date, end_date, capital, benchmark, benchmark2, base_currency, ignore_ticker_list, file_stock_info)

    
    


    def printExposure(self, date = None):
        if date is None: date = self.end_date
        print('Gross exposure: {:.1%}'.format(self.PA_snapshots.loc[date]['Gross_Exposure']))
        print('Net exposure: {:.1%}'.format(self.PA_snapshots.loc[date]['Net_Exposure']))
        print()



    def printHoldings(self, date = None, printSizeByClassifiers = [], lastPx = True, pxChg = True, sinceLastTrx = False, cumPnL = True, exposure = True, exitedPositions = False, zeroPnLPositions = False, aggreAH = False, start_date = None, sort_by = None):

        if date is None: 
            date = self.end_date
            hldgs = self.current_holdings
            if exitedPositions is False:
                hldgs['Long'] = hldgs['Long'][hldgs['Long']['Share#']!=0]
                hldgs['Short'] = hldgs['Short'][hldgs['Short']['Share#']!=0]
        else: 
            # disable certain features when date is not self.end_date
            sinceLastTrx = False
            aggreAH = False
            printSizeByClassifiers = []
        
            hldgs = self.holdings_snapshot_of_date(date, exited_positions = exitedPositions, zeroPnLPositions = zeroPnLPositions, last_px = lastPx, px_chg = pxChg, cum_pnl = cumPnL, start_date = start_date, sort_by = sort_by)       


        print()
        print('======================================================================')
        print('Holdings as of', helpers.date_to_string(date))
        print('======================================================================')
        print()

        if exposure: self.printExposure(date)

        def __printHoldings__(table, lastPx, pxChg, sinceLastTrx, cumPnL, name = True, intrinsicValue = False, reset_index = ''):
            params = dict(table=table, format2="{:,.0f}", columns2=['Size'], format3="{:.2%}", columns3=['Weight'], only_print_formatted_columns=True, reset_index=reset_index)
            
            if name:
                params['format'] ="{}"
                params['columns']=['Name']
            
            if lastPx:
                params['format4'] = '{:,.2f}'
                params['columns4'] = ['LastPx']
            
            if pxChg:
                params['format5'] = '{:,.1%}'
                params['columns5'] = ['PxChg']

            if sinceLastTrx:    
                if long_short == 'Long':
                    params['format6'] = '{:,.1%}'
                    params['columns6'] = ['SinceLatestBuy']
                if long_short == 'Short':
                    params['format6'] = '{:,.1%}'
                    params['columns6'] = ['SinceLatestSell']

            if cumPnL:
                params['format7'] = '{:,.0f}'
                params['columns7'] = ['CumPnL']
            
            if intrinsicValue:
                params['format8'] = '{:,.0f}'
                params['columns8'] = ['Intrinsic Value']
                
                
            helpers.printFormattedTable(**params)
 


        for long_short in self.long_short_list:
            print()
            print(long_short, " book:")
            
            a = hldgs[long_short]

            if aggreAH:
                k = lambda x: x[-2:]=='CH' or x[-2:]=='HK'
                not_k = lambda x: not(k(x))

                b = a[a.index.map(not_k)].copy()
                c = a[a.index.map(k)]
                c_sum = c.sum()
                cname = 'Chinese A/H Shares'
                c_sum.name = cname
                b = b.append(c_sum)
                b.loc[cname, 'Name'] = cname
                a = b

            __printHoldings__(a, lastPx, pxChg, sinceLastTrx, cumPnL)

            if aggreAH:
                print()
                print('... where Chinese A/H share holdings are:')
                __printHoldings__(c, lastPx, pxChg, sinceLastTrx, cumPnL)

 
            exp = a['Size'].sum()
            print('Total size = {:,.0f}, {:,.1%} of capital'.format(exp, exp/self.latest_capital))
            print()
        
        if self.has_option_book:
            print()
            print("Option book:")

            
            __printHoldings__(hldgs['Options'], lastPx, False, False, cumPnL, intrinsicValue = True, name = False, reset_index = None)

            exp = hldgs['Options']['Size'].sum()
            print('Total size = {:,.0f}, {:,.1%} of capital'.format(exp, exp/self.latest_capital))
            print()    
        
        for classifier in printSizeByClassifiers:
            r = self.getCurrentHoldingByClassifier(classifier)
            helpers.printPercentagePointTable(r, print_zero_as = '')

    def printPerfAttribution(self):
        if self.statusPerformanceAttributed == False:
            self.__attribute_performance__()        
        
        self.__printPerfAttrToPositioning__()
        self.printEquityMarketTiming()
        self.__printPerfAttrToSelection__()

    
    def printHistoricalReturns(self, period = 'Q', periodic_benchmark_return = False):
        annual_returns = self.PA_snapshots_sampled_return[self.PA_snapshots_sampled_return['Frequency'] == 'A'].copy()
        annual_returns['Year'] = annual_returns.index.year
        annual_returns.set_index('Year', inplace = True)

        period_returns = self.PA_snapshots_sampled_return[self.PA_snapshots_sampled_return['Frequency'] == period].copy()
        period_returns['Year'] = period_returns.index.year
        
        if period == 'Q':
            m2p = {3: 'Q1', 6: 'Q2', 9: 'Q3', 12: 'Q4'}
        elif period == 'M':
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            m = range(1,13)
            m2p = dict(zip(m, months))
        
        get_p = lambda x: m2p[x.month]
        period_returns['Period'] = period_returns.index.map(get_p)

        def periodic_returns(column, column_name):
            r = pd.pivot_table(period_returns, values=column, index=['Year'], columns=['Period'], aggfunc=np.sum)
            r['Annual'] = annual_returns[column]
            r.columns.name = column_name
            return r
        
        returns = periodic_returns('Time_Weighted_Port_Return', 'Portfolio Return')
        if periodic_benchmark_return:
            benchmark_r = periodic_returns('Benchmark_Return', '{} Return'.format(self.benchmark))
            if len(self.benchmark2)>1:
                benchmark2_r = periodic_returns('Benchmark2_Return', '{} Return'.format(self.benchmark2[1]))
        else:
            returns[self.benchmark] = annual_returns['Benchmark_Return']
            if len(self.benchmark2)>1:
                returns[self.benchmark2[1]] = annual_returns['Benchmark2_Return']

        if period == 'M':
            mm = []
            for emm in months:
                if emm in returns.columns:
                    mm.append(emm)
            if periodic_benchmark_return:
                returns = returns[mm + ['Annual']]
                benchmark_r = benchmark_r[mm + ['Annual']]
                if len(self.benchmark2)>1:
                    benchmark2_r = benchmark2_r[mm + ['Annual']]
            else:
                if len(self.benchmark2)>1:
                    returns = returns[mm + ['Annual', self.benchmark, self.benchmark2[1]]]
                else:
                    returns = returns[mm + ['Annual', self.benchmark]]
        
        helpers.printPercentagePointTable(returns)        
        if periodic_benchmark_return:
            helpers.printPercentagePointTable(benchmark_r)     
            if len(self.benchmark2)>1:
                helpers.printPercentagePointTable(benchmark2_r)


    def __printTimeWeightedReturnOnInvestedCapital__(self, snapshots_pure_cap_r):

        rl = snapshots_pure_cap_r['Return_Long'].iloc[1:]
        rl1 = rl + 1
        print()
        print('Long book time-weighted return on invested capital is {:.2%}'.format(rl1.product(axis=0)-1))

        if not(self.long_only):        
            rs = snapshots_pure_cap_r['Return_Short'].iloc[1:]
            rs1 = rs + 1
            print('Short book time-weighted return on invested capital is {:.2%} (positive return means negative P&L)'.format(rs1.product(axis=0)-1))


    def printTickerReturn(self, ticker, start_date = None, end_date = None):
        start_date, end_date = self.assign_start_and_end_date(start_date, end_date)
        r = self.equity_prices_local.loc[end_date, ticker] / self.equity_prices_local.loc[start_date, ticker] -1
        print('{} has returned {:.2%}'.format(ticker, r))


    def printPerfOfLastNumOfTradingDays(self, days = 5, printPerfClassifiers = []):
        start_date = self.__offset_days__(self.end_date, -days)
        self.printPerfBetweenDates(start_date, self.end_date, printPerfClassifiers)

    def printPerfBetweenDates(self, start_date, end_date, printPerfClassifiers = []):

        day_one = self.__next_day__(start_date)
        pas = self.PA_snapshots_pure_cap_return.loc[start_date:end_date] # including Day Zero
        idx_plus_one = pas.index
        idx = idx_plus_one[1:] # removing day zero

        self.__print_PnL_composition__(pas.iloc[1:]['P&L_Long'].sum(), pas.iloc[1:]['P&L_Short'].sum(), pas.iloc[1:]['P&L_Options'].sum(), date = start_date)   

        self.__printTimeWeightedReturnOnInvestedCapital__(pas)
        
        print()
        for bm in self.benchmark2:   
            self.printTickerReturn(bm, start_date, end_date)

        print()

        self.printHoldings(date=end_date, start_date=start_date, exitedPositions = True, sort_by = 'CumPnL')


        print()
        print('==========================================================================================================')
        s_day_one = helpers.date_to_string(day_one)
        s_end_date = helpers.date_to_string(end_date)
        if s_day_one != s_end_date:
            print('Performance Summary of', s_day_one, '-', s_end_date)
        else:
            print('Performance Summary of', s_end_date)
        print('==========================================================================================================')
        
        
        print()
        print('By classifiers')
        self.printPerfByClassifers(printPerfClassifiers, start_date, end_date)


    def printPerfByClassifers(self, classifiers, start_date = None, end_date = None):            
        for classifier in classifiers:
            r = self.analyzePerfByClassfier(classifier, start_date, end_date)
            helpers.printFormattedTable(r,'{:,.0f}', [('Long','Size')], '{:.2%}', [('Long','Avg_Exposure'), ('Long','TWRR')], '{:,.0f}', [('Long','P&L'),('Short','Size')],'{:.2%}', [('Short','Avg_Exposure'),('Short','TWRR')],  '{:,.0f}', [('Short','P&L'),('Net','Size'), ('Net', 'P&L')], only_print_formatted_columns = True, print_zero_as = '', print_nan_as = '')
            '''
            for long_short in ['Long', 'Short']:
                print()
                print('On the', long_short, 'side:')
                helpers.printFormattedTable(r[long_short], '{:,.0f}', ['P&L'], '{:.2%}', ['TWRR','Exposure*','Positioning', 'Selection', 'Total'], only_print_formatted_columns = True, print_zero_as = '')
            '''
        

    def printDailySummary(self, printPerfClassifiers = []):
        self.printPerfOfLastNumOfTradingDays(days=1, printPerfClassifiers= printPerfClassifiers)

    def printHeadlineSummary(self):
        b_equity_long, b_equity_short, b_options, b_cash, b_total = self.getBookBalance(self.start_date, print_out = False)
        
        print()
        print('==========================================================================================================')
        print('For the period between', self.start_date.strftime('%Y-%m-%d'), 'and', self.end_date.strftime('%Y-%m-%d'))
        print('==========================================================================================================')

               
        capital_deployed = b_total + self.net_deposit
        print()
        print('Capital deployed: {} {:,.2f}, where:'.format(self.base_currency, capital_deployed))
        print('  Beginning assets: {} {:,.2f}'.format(self.base_currency, b_total))
        print('  Net deposit: {} {:,.2f}'.format(self.base_currency, self.net_deposit))
        
        if helpers.isZero(self.capital - capital_deployed) == False:
            print("However, return calculation using specified capital {} {:,.2f} instead of calculated capital deployed".format(self.base_currency, self.capital))

        print()       
        self.print_metrics('P&L composition')

    


    
    def printPerformanceSummary(self, printPerfClassifiers = []):

        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])

        self.printHeadlineSummary()
        print()

        self.print_metrics('Vol')
        self.print_metrics('Corr')

        self.printHoldings(printSizeByClassifiers = printPerfClassifiers, exitedPositions = True, sinceLastTrx = True)

        print()
        print('==========================================================================================================')
        print('Beginning balance')
        print('==========================================================================================================')

        self.getBookBalance(self.start_date, print_out = True)
        


        print()
        print('==========================================================================================================')
        print('Ending balance')
        print('==========================================================================================================')

        self.getBookBalance(self.end_date, print_out = True)


        
        print()
        print('==========================================================================================================')
        print('Performance attribution 1')
        print('==========================================================================================================')
        
        
        self.printPerfAttribution()
                
        
        print()
        print('==========================================================================================================')
        print('Performance attribution 2')
        print('==========================================================================================================')
        
        
        
        print()
        print('P&L attribution :')
        self.printPerfByClassifers(printPerfClassifiers)
        

        print()
        print('==========================================================================================================')
        print('Time weighted returns on invested capital (excl. uninvested cash)')
        print('==========================================================================================================')
        

        self.__printTimeWeightedReturnOnInvestedCapital__(self.PA_snapshots_pure_cap_return)
 
        
        rl = self.PA_snapshots_pure_cap_return['Return_Long'].iloc[1:] # removing the Day Zero return
        rt = self.PA_snapshots['Daily_Return'].iloc[1:] # removing the Day Zero return
        
        for bm in self.benchmark2:   
            b = self.equity_prices_local[bm]
            rb = self.equity_daily_return_local[bm]
            print()
            benchmark_r = b.iloc[-1]/b.iloc[0]-1
            print('{} has returned {:.2%}'.format(bm, benchmark_r))
            print('Long portfolio (ex. unused cash) beta = {:.2f}'.format(finance.beta(rl, rb)))
            
            beta = finance.beta(rt, rb)
            alpha = self.metrics['Return'] - benchmark_r * beta
            
            print('Total portfolio (incl. long, short and unused cash) beta = {:.2f}, alpha = {:.2%}'.format(beta, alpha))



        print()
        print('==========================================================================================================')
        print('Batting Average')
        print('==========================================================================================================')
        print()

        self.print_metrics('Batting avg')
        print()
        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])

    def printHistoricalPerformance(self):
        print('Historical returns:')
        self.printHistoricalReturns(period = 'Q', periodic_benchmark_return = True)
        print()
        self.printHistoricalReturns(period = 'M', periodic_benchmark_return = True)
        print()
        print('Portfolio NAV:')
        self.plotPortfolioNAV()
        print()
        print('Portfolio concentration:')
        self.plotPortfolioConcentration()
        


    def printPriceActionsWithList(self, selected, print_stock_charts_without_action = True):
        pa = price_actions.Price_Actions(self.equity_prices_local_full_history, self.equity_daily_return_local_full_history, watchlist=selected)

        d = pa.get_price_actions()
        
        def last_px(ticker):
            px =self.equity_prices_local.loc[self.end_date, ticker]
            return 'last price = {:.2f},'.format(px)
        
        for idx in d.index:
            print(idx+',', last_px(idx), d.loc[idx, 'Price_Action'])
            self.plotPriceLTM(idx, show = True, annotate_trades = 10)

        if print_stock_charts_without_action:        
            for x in selected:
                if x not in list(d.index):
                    print(x+',', last_px(x))
                    self.plotPriceLTM(x, show = True, annotate_trades = 10)



    def printPriceActions(self, long_short = True, watchlist = False, print_stock_charts_without_action = True):

        if long_short:
            for ls in self.long_short_list:
                print(ls,'side:')
                self.printPriceActionsWithList(self.traded_equity_list[ls], print_stock_charts_without_action)

        if watchlist:
            watchlist = self.stock_info['OnWatchList'][self.stock_info['OnWatchList']>0.1]
            watchlist = watchlist.index.values

            selected = ['SPX', 'QQQ']
            selected.extend(watchlist)
            if 'PLACEHOLDER' in selected: selected.remove('PLACEHOLDER')

            self.printPriceActionsWithList(selected, print_stock_charts_without_action)


    def printWinnersLosers(self, n=5, details = False):
        if self.statusPerformanceAttributed == False:
            self.__attribute_performance__()
        
        abe = self.attribution_by_equity

        for long_short in self.long_short_list:
            print(long_short, "book top {:.0f} winners and losers:".format(n))
            if details:
                helpers.printFormattedTable(abe[long_short], format = '{:.2%}', columns=['Selection', 'Positioning', 'Total', 'TWRR'], format2 = '{:,.0f}', columns2 = ['P&L'], head=n, tail=n, reset_index='Name')
            else:
                helpers.printFormattedTable(abe[long_short], format = '{:,.0f}', columns = ['P&L'], head=n, tail=n, reset_index='Name')

            print()



    def __printPerfAttrToSelection__(self):
        if self.statusPerformanceAttributed == False:
            self.__attribute_performance__()

        
        pra = self.portfolio_return_attribution
        abe = self.attribution_by_equity

        print()
        print('Selection contributed {:.2%} return, including {:.2%} from long and {:.2%} from short'.format(pra.loc['Selection','Net'], pra.loc['Selection','Long'], pra.loc['Selection','Short']))
        print()
        
        '''
        for long_short in self.long_short_list:
            print(long_short, "book top 6 winners and losers:")
            helpers.printFormattedTable(abe[long_short], format = '{:.2%}', columns=['Selection', 'Positioning', 'Total', 'TWRR'], format2 = '{:,.0f}', columns2 = ['P&L'], head=6, tail=6, reset_index='Name')
            print()
        '''
        self.printWinnersLosers(n=6, details = True)

    def __printPerfAttrToPositioning__(self):
        if self.statusPerformanceAttributed == False:
            self.__attribute_performance__()


        pra = self.portfolio_return_attribution
        papa = self.PA_PerformanceAttribution
                
        
        print()
        print('Positioning contributed {:.2%} return, including {:.2%} from long and {:.2%} from short'.format(pra.loc['Positioning','Net'], pra.loc['Positioning','Long'], pra.loc['Positioning','Short']))
        print('    * Performance attribution to positioning does not include P&L from options')
        
        
        equiv_start_long_exposure = pra.loc['Positioning','Long'] /self.benchmark_return
        equiv_start_short_exposure = pra.loc['Positioning','Short'] /self.benchmark_return
        equiv_start_exposure = equiv_start_long_exposure + equiv_start_short_exposure
        print()
        print('    * Positioning implies {:.2%} equivalent starting net exposure, or {:.2%} long and {:.2%} short at start'.format(equiv_start_exposure,equiv_start_long_exposure,equiv_start_short_exposure))

        avg_long_exposure = np.average(papa['Long_Exposure_Percent_EoD'])
        avg_short_exposure = np.average(papa['Short_Exposure_Percent_EoD'])
        
        print('    * Throughout the period, average long exposure is {:.2%}, short exposure is {:.2%}'.format(avg_long_exposure, avg_short_exposure))

        print()
        print('    Allocation contributed {:.2%} return, including {:.2%} from long and {:.2%} from short'.format(pra.loc['Allocation', 'Net'], pra.loc['Allocation', 'Long'], pra.loc['Allocation', 'Short']))
        print('        * Allocation contribution assumes that total capital that has been put into long and short sides, respectively, is deployed on Day 0')

        print()
        print('    Market timing contributed {:.2%} return, including {:.2%} from long and {:.2%} from short'.format(pra.loc['Timing', 'Net'], pra.loc['Timing', 'Long'], pra.loc['Timing', 'Short']))
