'''
Created on Feb 16, 2019

author: yewen
'''
import pandas as pd
import numpy as np
import PMStack.Tools.finance as finance
import PMStack.Tools.helpers as helpers
import warnings
import inspect
from datetime import timedelta
from PMStack.Equity_port import portfolio_core
from IPython.display import display


class Portfolio_Perf(portfolio_core.Portfolio_Core):
    '''
    Adding performance analysis on top of Portfolio_Core
    '''

    def __init__(self, file_price, file_trades, start_date, end_date=None, capital=None, benchmark = 'SPX', benchmark2=[], base_currency = 'USD', ignore_ticker_list = [], file_stock_info = None ):
        super().__init__(file_price, file_trades, start_date, end_date, capital, benchmark, benchmark2, base_currency, ignore_ticker_list, file_stock_info)

        self.statusAnalyzedPnLbyEquity = False
        self.statusPerformanceAttributedToPositioning = False
        self.statusPerformanceAttributed = False
        self.statusAnalyzedMarketTiming = False

        

        self.equity_PnL_by_classifiers = {}
        self.current_holding_by_classifiers = {}
        self.equity_daily_return_USD_by_classifiers = {}
        self.positioning_contribution_by_classifiers = {}
        self.selection_contribution_by_classifiers = {}
        self.size_by_classifiers = {}



        self.attribution_by_equity = None #This is not neccessary, but for clearing an error where the variable appears in the code before explicitly initalized (but no issue in the runtime)
        self.__attribute_performance__()
        self.__analyze_perf_by_equity__()
        self.__analyze_batting_average__()

        
        




    def __attribute_performance__(self, start_date = None, end_date = None):

        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])

        if self.statusPerformanceAttributed and (start_date is None) and (end_date is None):
            print()
            warnings.warn('Exit portfolio_perf.attributePerformance() without execution as it is done before.')
            return self.attribution_by_equity

        if (self.statusPerformanceAttributed is False) and ((start_date is not None) or (end_date is not None)):
            self.__attribute_performance__()
            return self.__attribute_performance__(start_date, end_date)
            
        if self.statusPerformanceAttributed is False:
            self.positioning_contribution_by_equity = {}
            self.selection_contribution_by_equity = {}
            pcbe = self.positioning_contribution_by_equity
            scbe = self.selection_contribution_by_equity

            for long_short in ['Long', 'Short']:  
                x = (self.equity_value_USD[long_short] / self.capital).shift(1).iloc[1:]  # starting from self.start_date_plus_one
                pcbe[long_short] = x.mul(self.equity_daily_return_USD[self.benchmark], axis = 0)
                scbe[long_short] = self.equity_PnL[long_short] / self.capital - pcbe[long_short]

            self.attribution_by_equity = {}
            abe = self.attribution_by_equity
            self.TWRR_by_equity = {}
            tbe = self.TWRR_by_equity
        else:
            start_date, end_date = self.assign_start_and_end_date(start_date, end_date)
            start_date = start_date + timedelta(1)

            pcbe = {}
            scbe = {}
            for long_short in ['Long', 'Short']:  
                pcbe[long_short] = self.positioning_contribution_by_equity[long_short].loc[start_date:end_date]
                scbe[long_short] = self.selection_contribution_by_equity[long_short].loc[start_date:end_date]

            abe = {}
            tbe = {}

        
        #for long_short in self.long_short_list:  
        for long_short in ['Long', 'Short']:  
                        

            a = scbe[long_short].sum(axis = 0).rename('Selection')
            b = pcbe[long_short].sum(axis = 0).rename('Positioning')
            abe[long_short] = pd.concat([a,b],axis = 1)
            abe[long_short].index.name = 'Ticker'
            abe[long_short]['Total'] = abe[long_short]['Selection'] + abe[long_short]['Positioning']
            abe[long_short].sort_values('Total', ascending = False, inplace = True)
            

            '''
            Calculate Time Weighted Rate of Return (TWRR) by equity
            '''
   
            daily_equity_return = self.equity_daily_return_USD.loc[self.equity_holding_window[long_short].index,self.equity_holding_window[long_short].columns]

            daily_equity_return = daily_equity_return.loc[start_date:end_date]

            tbe[long_short] = (daily_equity_return*self.equity_holding_window[long_short].loc[start_date:end_date]).apply(lambda x: np.prod(x+1)-1).rename('TWRR') 

            
            abe[long_short] = abe[long_short].join(tbe[long_short])
            abe[long_short]['P&L'] = abe[long_short]['Total'] * self.capital
            abe[long_short] = self.__add_company_name__(abe[long_short])
                                                
        #End of For loop

        # If just for a specific date range, no need to go on below
        if self.statusPerformanceAttributed:               
            if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])
            return abe



        a = self.positioning_contribution_by_equity['Long'].sum(axis=1).rename('Positioning_Long')
        b = self.positioning_contribution_by_equity['Short'].sum(axis=1).rename('Positioning_Short')
        c = self.selection_contribution_by_equity['Long'].sum(axis=1).rename('Selection_Long')
        d = self.selection_contribution_by_equity['Short'].sum(axis=1).rename('Selection_Short')
        e = self.equity_daily_return_USD[self.benchmark].rename('Benchmark_Return')
    
        # by date returns
        self.PA_PerformanceAttribution = pd.concat([a,b,c,d, e],axis=1)
        
        papa = self.PA_PerformanceAttribution
        paspcr = self.PA_snapshots_pure_cap_return

        papa['Long_Exposure_Percent_EoD'] = paspcr['Equity_Long'] / self.capital
        papa['Short_Exposure_Percent_EoD'] = paspcr['Equity_Short'] / self.capital
        papa['Net_Exposure_Percent_EoD'] = papa['Long_Exposure_Percent_EoD'] + papa['Short_Exposure_Percent_EoD']

        self.portfolio_return_attribution = pd.DataFrame(index = ['Total','Selection','Positioning','Allocation','Timing'], columns=['Description', 'Net', 'Long', 'Short'])
        pra = self.portfolio_return_attribution
        pra.loc['Total', 'Description'] = 'Position + Selection (residue is Options + FX on currency balance)'
        pra.loc['Positioning', 'Description'] = 'Allocation + Timing'
        pra.loc['Selection', 'Description'] = ''
        pra.loc['Allocation', 'Description'] = 'Assuming all capital deployed on Day 0'
        pra.loc['Timing', 'Description'] = ''
        pra.loc['Total','Net'] = self.metrics['Return']
        pra.loc['Total','Long'] = self.metrics['Return composition']['Long']
        pra.loc['Total','Short'] = self.metrics['Return composition']['Short']


        pra.loc['Positioning','Long'] = papa['Positioning_Long'].sum()
        pra.loc['Positioning','Short'] = papa['Positioning_Short'].sum()
        pra.loc['Positioning','Net'] = pra.loc['Positioning','Long'] + pra.loc['Positioning','Short']


        
        allocatedCapital_long = paspcr.loc[self.start_date, 'Equity_Long'] - paspcr['Currency_Long'].min()
        allocatedCapital_short = paspcr.loc[self.start_date, 'Equity_Short'] - paspcr['Currency_Short'].max()
        
        pra.loc['Allocation', 'Long'] = allocatedCapital_long / self.capital * self.benchmark_return
        pra.loc['Allocation', 'Short'] = allocatedCapital_short / self.capital * self.benchmark_return
        pra.loc['Allocation', 'Net'] = pra.loc['Allocation', 'Long'] + pra.loc['Allocation', 'Short']

        
        pra.loc['Timing', 'Long'] = pra.loc['Positioning','Long'] - pra.loc['Allocation', 'Long']
        pra.loc['Timing', 'Short'] = pra.loc['Positioning','Short'] - pra.loc['Allocation', 'Short']
        pra.loc['Timing', 'Net'] = pra.loc['Timing', 'Long'] + pra.loc['Timing', 'Short']

        pra.loc['Selection','Long'] = scbe['Long'].values.sum()
        pra.loc['Selection','Short'] = scbe['Short'].values.sum()
        pra.loc['Selection','Net'] = pra.loc['Selection','Long'] + pra.loc['Selection','Short']
                
        # Checksum logic        
        r = scbe[long_short].values.sum() + pra.loc['Positioning', long_short]
        rr = self.metrics['Return composition'][long_short]
        err = r + self.currency_PnL[long_short].sum(axis = 0)/self.capital - rr
        if helpers.isZero(err) == False:
            print()
            warnings.warn('Error calculating selection attribution:')
            print(long_short, r, rr)
        # End of Checksum logic

        self.statusPerformanceAttributed = True         

                
        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])      
        return abe

    
        
    '''
    Designed for analyze trading pre-emptive trading decisions ahead of market volatility, ideally a V-shape correction-recovery
    '''
    def analyzeTrading(self, start_date, end_date):
        if self.statusPerformanceAttributed == False:
            self.__attribute_performance__()

        b_return = self.equity_prices_local.loc[end_date, self.benchmark]/self.equity_prices_local.loc[start_date, self.benchmark] -1
        print('From {} to {}, {} returned {:,.3%}'.format(helpers.date_to_string(start_date), helpers.date_to_string(end_date), self.benchmark, b_return))
        print()

       
        papa = self.PA_PerformanceAttribution.loc[start_date:end_date]
        r = papa['Positioning_Long'].sum() + papa['Positioning_Short'].sum()
        print('Positioning returned {:,.3%}, and the profits were {} {:,.2f}'.format(r, self.base_currency, r*self.capital))
        #print(self.PA_snapshots.loc[start_date, 'Capital_Deployment'])
        #print(self.PA_snapshots.loc[end_date, 'Capital_Deployment'])
        leverage = self.PA_snapshots.loc[end_date, 'Capital_Deployment'] / self.capital
        print('Return would have been {:,.3%} if all capital allocated and deployed on Day 1.'.format(b_return * leverage))

        self.printEquityMarketTiming(total_only = True, start_date = start_date, end_date = end_date)         

    def printEquityMarketTiming(self, total_only = False, start_date = None, end_date = None):
        if self.statusAnalyzedMarketTiming == False:
            if self.equity_and_ETF_trades.empty:
                print('No trades happened. So not performing market timing analysis')
                return
            self.__analyzeEquityPortfolioMarketTiming__()
        
        start_date, end_date = self.assign_start_and_end_date(start_date, end_date)
        
        t = self.equity_market_timing.loc[start_date:end_date]
        print_prefix = ['Weighted average', '    Long side:', '    Short side:']
        
        print()
        if self.long_only or total_only:
            l = ['Total']
        else:
            l = ['Total', 'Long', 'Short']
        for idx, long_short in enumerate(l):
            b_level = t[long_short+'BuyAmount'].sum() / t[long_short+'BuyQ'].sum()
            s_level = t[long_short+'SellAmount'].sum() / t[long_short+'SellQ'].sum()
            print(print_prefix[idx], 'buy trades happened when {:} was at {:.0f}, while sell happened at {:.0f}, {:.2%} from buy'.format(self.benchmark, b_level, s_level, s_level/b_level-1))



    def peak(self, indices=None):
        
        if indices == None:
            indices = [self.benchmark]
        
        for index in indices:        
            peak_date = self.equity_prices_local[index].idxmax()
            str_peak_date = pd.Timestamp(peak_date).strftime('%Y-%m-%d') 
            peak_price = self.equity_prices_local.loc[peak_date,index]
            current_price = self.equity_prices_local.loc[self.end_date,index]
            downside = current_price / peak_price -1
            print(index,'peaked on', str_peak_date, "at {:.0f}. It is down {:.1%} since.".format(peak_price, downside))
            
            trades = self.equity_and_ETF_trades.loc[peak_date:self.end_date]
            trades = pd.DataFrame(trades.groupby(['Date'])['TransactionAmountUSD'].sum())
            amount = -trades['TransactionAmountUSD'].sum()
            print('A capital of {:,.0f} was deployed since peak.'.format(amount))
            
            cutoff_pct = 0
            while (amount>1 or amount <-1):
                cutoff_pct += 0.05
                cutoff = peak_price*(1-cutoff_pct)
                amount = 0.0
                for idx in trades.index:
                    if (self.equity_prices_local.loc[idx,index]<=cutoff):
                        amount -= trades.loc[idx,'TransactionAmountUSD']
                
                print('A capital of {:,.0f} was deployed at least {:.0%} below peak.'.format(amount, cutoff_pct))    
            
            print()
        



    
     
    
    def __analyzeEquityPortfolioMarketTiming__(self):
        #t = pd.DataFrame(self.equity_prices_local[self.benchmark])
        
        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])
        
        if self.equity_and_ETF_trades.empty:
            print('No trades happened. So not performing market timing analysis')
            return


        trades = self.equity_and_ETF_trades
        trades['LongAmount'] = trades.apply(lambda x: x['TransactionAmountUSD'] if x['Long_Short'] == 'Long' else 0.0, axis = 1)
        trades['ShortAmount'] = trades.apply(lambda x: x['TransactionAmountUSD'] if x['Long_Short'] == 'Short' else 0.0, axis = 1)
        trades = trades.groupby(['Date'])['TransactionAmountUSD','LongAmount','ShortAmount'].sum()
        trades.rename(columns = {'TransactionAmountUSD':'TotalAmount'}, inplace = True)
        trades[self.benchmark] = self.equity_prices_local[self.benchmark]
        
        if self.long_only:
            l = ['Total']
        else:
            l = ['Total', 'Long', 'Short']
        for long_short in l:
            
            col = long_short + 'Amount'
            
            trades[long_short+'BuyAmount'] = -trades[trades[col]<0][col]
            trades[long_short+'SellAmount'] = trades[trades[col]>0][col]
            trades.fillna(0,inplace = True)
            trades[long_short+'BuyQ'] = trades[long_short+'BuyAmount'] / trades[self.benchmark]
            trades[long_short+'SellQ'] = trades[long_short+'SellAmount'] / trades[self.benchmark]
        #end for loop
        
        self.equity_market_timing = trades
        self.statusAnalyzedMarketTiming = True

           
    def __calculatePerfByEquity__(self, frequency):

        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])



        if frequency == 'W':
            frequency = 'W-SAT'
        
        results = {}
        for long_short in self.long_short_list:
            monthlyPnL = self.equity_PnL[long_short].resample(frequency).sum()
            monthlySelection = self.selection_contribution_by_equity[long_short].resample(frequency).sum()
            monthlyPositioning = self.positioning_contribution_by_equity[long_short].resample(frequency).sum()
            daily_equity_return = self.equity_daily_return_USD.loc[:,monthlyPnL.columns]
            monthlyMWRR = pd.DataFrame(index = monthlyPnL.index.values, columns=monthlyPnL.columns.values).fillna(np.nan)
            monthlyTWRR = pd.DataFrame(index = monthlyPnL.index.values, columns=monthlyPnL.columns.values).fillna(np.nan)
            monthlyMoM = pd.DataFrame(index = monthlyPnL.index.values, columns=monthlyPnL.columns.values).fillna(np.nan)
            monthlyEquityReturn = pd.DataFrame(index = monthlyPnL.index.values, columns=monthlyPnL.columns.values).fillna(np.nan)
            
            #print(monthlyPnL)
            
            
            
            index = daily_equity_return.index
           
            start_date = pd.to_datetime('2000-1-1')
            end_date = pd.to_datetime('2000-1-1')
            
            for idx in monthlyPnL.index:
                if (frequency == 'M'):
                    partial_index = index[(index.year == idx.year) & (index.month == idx.month)]
                elif (frequency == 'Y'):
                    partial_index = index[index.year == idx.year]
                elif (frequency == 'W-SAT'):
                    partial_index = daily_equity_return.loc[idx - pd.DateOffset(6) : idx].index
                else:
                    raise ValueError('Code incomplete to process all sampling frequency')
                '''
                for a month say May, partial_index = 5/1 - 5/31, start_date:end_date = 4/30-5/31
                '''
                               
                
                start_date = end_date    
                end_date = partial_index[-1]
                
                #print(start_date, end_date)
                #print()
                #print(partial_index)                
                
                
                holding_window = self.equity_holding_window[long_short].loc[partial_index]
                holding_days = holding_window.apply(lambda x: sum(x))
                
                
                equity_value = self.equity_value_USD[long_short].loc[start_date:end_date]

                '''
                Cashflow happens at the end of the day, thus - 
                Investment happens at the beginning of the day, so needed to move to the end of the previous day
                Payment happens at the end of the day
                '''
                
                #equity_investment_cashflows = self.equity_investment_cashflows[long_short].loc[start_date:end_date].copy()
                #equity_investment_cashflows = equity_investment_cashflows.shift(-1)
                equity_investment_cashflows = self.equity_investment_cashflows_shift_minus_one[long_short].loc[start_date:end_date].copy()                
                equity_investment_cashflows.iloc[-1] = 0
                
                
                
                equity_payment_cashflows = self.equity_payment_cashflows[long_short].loc[start_date:end_date].copy()
                
                '''
                the first row, i.e., the last day of the previous month, is irrelevant here, so zero it out
                '''
                equity_payment_cashflows.iloc[0,:] = 0
                
                
                

                
                '''
                if (long_short=='Long'):
                    equity_investment_cashflows.iloc[0,:] -= equity_value.iloc[0,:]
                    equity_payment_cashflows.iloc[-1,:] += equity_value.iloc[-1,:]
                else:
                    # for shorts, inverse cash flow so that investment and payment are consistent in signs (+/-) with longs
                    equity_investment_cashflows.iloc[0,:] += equity_value.iloc[0,:]
                    equity_payment_cashflows.iloc[-1,:] -= equity_value.iloc[-1,:]    
                
                
                equity_cashflows = equity_investment_cashflows + equity_payment_cashflows
                '''
                    
                equity_cashflows = self.__prepare_cashflows__(long_short, equity_investment_cashflows, equity_payment_cashflows, equity_value)
                
                
                for v in equity_cashflows.columns:
                    if (holding_days[v]<1): continue

                    cashflows = list(equity_cashflows.loc[:, v])
                    if (sum([abs(x) for x in cashflows])>0.0000001):
                        #print(long_short, idx, v)
                        #print(cashflows)
                        #print(finance.xirr(cashflows))

                        irr, mom, investment = finance.xirr(cashflows, error_msg=(v, idx))
                                                        
                        monthlyMWRR.loc[idx, v] = (1+irr)**(holding_days[v])-1
                        monthlyMoM.loc[idx, v] = mom
                
                der = daily_equity_return.loc[partial_index]
                monthlyEquityReturn.loc[idx] = der.apply(lambda x: np.prod(x+1)-1)
                monthlyTWRR.loc[idx] = (der*holding_window).apply(lambda x: np.prod(x+1)-1) 
                
            '''      
            print(long_short)
            print()
            print(monthlyMWRR.head)
            print()
            print(monthlyEquityReturn)
            print()
            print(monthlyTWRR)
            print()
            '''
            
            
            '''
            The portfolio weight of short positions is calculated using the size of the long book
            '''
            equity_weight = self.equity_value_USD[long_short].div(self.PA_snapshots_pure_cap_return.loc[:,'Equity_Long'], axis = 0)
            equity_weight = equity_weight.astype('float32')
            try:
                equity_weight = equity_weight.resample(frequency).mean()
            except Exception as err:
                warnings.warn(str(err))
            
            
            results[long_short] = {'MWRR': monthlyMWRR, 'TWRR': monthlyTWRR, 'StockReturn': monthlyEquityReturn, 'PnL': monthlyPnL, 'Port_Weight': equity_weight, 'MoM':monthlyMoM, 'Selection':monthlySelection, 'Positioning':monthlyPositioning}
    
        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])
    
        return results
    
    
    def __prepare_cashflows__(self, long_short, equity_investment_cashflows, equity_payment_cashflows, equity_value):
        
        '''
        please note - no self is used here
        '''
        equity_cashflows = equity_investment_cashflows + equity_payment_cashflows
            
        if (long_short=='Long'):    
            equity_cashflows.iloc[0,:] -= equity_value.iloc[0,:]
            equity_cashflows.iloc[-1,:] += equity_value.iloc[-1,:]                    
        else:
            # for shorts, inverse cash flow so that investment and payment are consistent in signs (+/-) with longs
            equity_cashflows.iloc[0,:] += equity_value.iloc[0,:]
            equity_cashflows.iloc[-1,:] -= equity_value.iloc[-1,:]    
            
        return equity_cashflows
    
    '''
    populate self.equity_PnL_periodic_summary = {'Long_M', 'Long_Y', 'Short_M', 'Short_Y', 'Long_W', 'Short_W'}
    '''
    def __analyze_perf_by_equity__(self):

        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])


        if self.statusPerformanceAttributed == False:
            self.__attribute_performance__()
        
        self.equity_PnL_periodic_summary = {}
                
        for frequency in ['M','Y','W']:
        
            results = self.__calculatePerfByEquity__(frequency)
            for long_short in self.long_short_list:

                #results = {'MWRR': monthlyMWRR, 'TWRR': monthlyTWRR, 'StockReturn': monthlyEquityReturn, 'PnL': monthlyPnL, 'Port_Weight': equity_weight}
                tables = []
                table_names = []
                for k, v in results[long_short].items():
                    tables.append(v)
                    table_names.append(k)
                    
                output = helpers.excel_tables_to_relational_tables(tables, table_names, 'Period', 'Ticker', 'Period')
                    
                self.equity_PnL_periodic_summary[long_short+'_'+frequency] = self.__add_company_name__(output, ticker_is_index=False)
            #end for long_short
            
                
        self.statusAnalyzedPnLbyEquity = True
                
        if self.__debug_mode__: print('Exiting', inspect.stack()[0][3])
    
    def getHoldingByClassifier(self, classifier, date=None, pct=True):
        if date is None:
            h = self.current_holdings
            cap = self.latest_capital
        else:
            h = self.holdings_snapshot_of_date(date)
            cap = self.PA_snapshots['Balance_EOD'][date]

        a = self.sum_along_ticker_index_by_classifier(pd.DataFrame(h['Long']['Size']), classifier)
        b = self.sum_along_ticker_index_by_classifier(pd.DataFrame(h['Short']['Size']), classifier)
        t = pd.concat([a, b], axis=1, sort=True).fillna(0.0)
        t.index.name = classifier
        t.columns = ['Long', 'Short']
        if pct:
            t = t / cap
        t['Net'] = t['Long'] + t['Short']            
        return t.sort_values(by='Net', ascending=False)
        
        
    
    def getCurrentHoldingByClassifier(self, classifier, pct=True):
        if not (classifier in self.current_holding_by_classifiers):
            self.current_holding_by_classifiers[classifier] = self.getHoldingByClassifier(classifier, date=None, pct=pct)
        return self.current_holding_by_classifiers[classifier]
    
    def analyzePerfByClassfier(self, classifier, start_date = None, end_date = None, attribution = False, net_pnl = True):
        
        if self.statusPerformanceAttributed is False:
            self.__attribute_performance__()

        if not (classifier in self.equity_PnL_by_classifiers):
            self.equity_daily_return_USD_by_classifiers[classifier] = {}
            self.equity_PnL_by_classifiers[classifier] = {}
            self.positioning_contribution_by_classifiers[classifier] = {}
            self.selection_contribution_by_classifiers[classifier] = {}
            self.size_by_classifiers[classifier] = {}
            
            self.stock_info.loc[:,classifier].fillna('Unspecified', inplace = True)
            
            for long_short in ['Long', 'Short']:
                t = self.sum_along_ticker_columns_by_classifier(self.equity_daily_capital_base[long_short], classifier)
                tt = self.sum_along_ticker_columns_by_classifier(self.equity_PnL[long_short] , classifier)
                
                self.equity_PnL_by_classifiers[classifier][long_short] = tt
    
                if long_short == 'Short':
                    ttt = -tt/t
                else:
                    ttt = tt/t
    
                self.equity_daily_return_USD_by_classifiers[classifier][long_short] = ttt

                self.positioning_contribution_by_classifiers[classifier][long_short] = self.sum_along_ticker_columns_by_classifier(self.positioning_contribution_by_equity[long_short], classifier)
                self.selection_contribution_by_classifiers[classifier][long_short] = self.sum_along_ticker_columns_by_classifier(self.selection_contribution_by_equity[long_short], classifier)
                self.size_by_classifiers[classifier][long_short] = self.sum_along_ticker_columns_by_classifier(self.equity_value_USD[long_short], classifier)
        

        start_date, end_date = self.assign_start_and_end_date(start_date, end_date)
        br = self.equity_prices_local.loc[end_date, self.benchmark] / self.equity_prices_local.loc[start_date, self.benchmark] -1

        start_date = start_date + timedelta(1)
        
        r = {}
        #r1 = {}
        for long_short in ['Long','Short']:
            twrr = self.equity_daily_return_USD_by_classifiers[classifier][long_short].loc[start_date:end_date]
            pnl = self.equity_PnL_by_classifiers[classifier][long_short].loc[start_date:end_date]
            posi = self.positioning_contribution_by_classifiers[classifier][long_short].loc[start_date:end_date]
            select = self.selection_contribution_by_classifiers[classifier][long_short].loc[start_date:end_date]
            
            twrr = twrr.apply(helpers.growth_resampler).rename('TWRR', inplace = True)
            pnl = pnl.sum().rename('P&L', inplace = True)
            posi = posi.sum().rename('Positioning', inplace = True)
            expo = (posi / br).rename('Avg_Exposure', inplace = True)
            select = select.sum().rename('Selection', inplace = True)
            total = (posi+select).rename('Total', inplace = True)
            size = self.size_by_classifiers[classifier][long_short].loc[end_date].rename('Size', inplace=True)


            t = pd.concat([size, expo, twrr, pnl], axis=1)
            if attribution: t = pd.concat([t, posi, select, total], axis=1)
            t.index.name = classifier
            x = helpers.non_zero_dataFrame_columns_or_rows(t, 'row')
            r[long_short] = t.loc[x]

        if net_pnl:
            #r = pd.merge(r['Long'], r['Short'], how='outer', on = classifier, suffixes=(' - Long', ' - Short')).fillna(0.0)
            r = pd.concat(r.values(), axis = 1, keys = r.keys(), sort = True)
            r[('Long','Avg_Exposure')].fillna(0, inplace = True)
            r[('Short','Avg_Exposure')].fillna(0, inplace = True)
            r[('Long','P&L')].fillna(0, inplace = True)
            r[('Short','P&L')].fillna(0, inplace = True)
            r[('Long','Size')].fillna(0, inplace = True)
            r[('Short','Size')].fillna(0, inplace = True)

            r['Net', 'Size'] = r['Long']['Size'] + r['Short']['Size']
            r['Net', 'P&L'] = r['Long']['P&L'] + r['Short']['P&L']
            r = r.sort_values(by=[('Net', 'Size')], ascending=[0])
        else:
            r = r.sort_values(by=[('Long', 'Size')], ascending=[0])
        return r
    

    def __analyze_batting_average__(self):
        if self.__debug_mode__: print('Entering', inspect.stack()[0][3])
        
        bats = {}
        self.metrics['Batting avg'] = {}
        for long_short in ['Long','Short']:
            
            equity_cashflows = self.__prepare_cashflows__(long_short, self.equity_investment_cashflows_shift_minus_one[long_short], self.equity_payment_cashflows[long_short], self.equity_value_USD[long_short])
            
            a = self.equity_holding_and_trading_window[long_short]
            
            bats[long_short] =[]
            self.metrics['Batting avg'][long_short] = {}
            
            
            for c in a.columns:
                start = True
                end = False
                for idx in a.index:
                    if(start and a.loc[idx, c] == 1):
                        start = False
                        start_date = idx
                        end = True
                        last_idx = idx
                    elif a.loc[idx, c] ==1:
                        last_idx = idx
                    elif (end and a.loc[idx,c] ==0):
                        start = True
                        end = False
                        bats[long_short].append([c, start_date, last_idx])
                if end:
                    bats[long_short].append([c, start_date, last_idx])
            
            self.metrics['Batting avg'][long_short]['Count'] = len(bats[long_short])
            count_profitable = 0
            count_beats_bm = 0
            
            for idx, [ticker, start_date, end_date] in enumerate(bats[long_short]):
                profits = self.equity_PnL[long_short].loc[start_date, ticker] + self.equity_PnL_cumulative[long_short].loc[end_date, ticker] - self.equity_PnL_cumulative[long_short].loc[start_date, ticker]
                bats[long_short][idx].append(profits)
                
                start_date_minus_one = self.__previous_day__(start_date)
                
                cf = list(equity_cashflows.loc[start_date_minus_one :end_date, ticker])
                
                #print(ticker, cf)
                
                irr, mon, investment = finance.xirr(cf, error_msg=(ticker, start_date_minus_one , end_date))
                irr = (1+irr)**(len(cf)-1)-1
                bats[long_short][idx].append(investment)
                bats[long_short][idx].append(irr)
                bats[long_short][idx].append(mon)
                
                benchmark_return = self.equity_prices_USD.loc[end_date, self.benchmark]/self.equity_prices_USD.loc[start_date_minus_one, self.benchmark] - 1               
                bats[long_short][idx].append(benchmark_return)

                stock_return = self.equity_prices_USD.loc[end_date, ticker]/self.equity_prices_USD.loc[start_date_minus_one, ticker] - 1               
                bats[long_short][idx].append(stock_return)

                if profits > 1: count_profitable = count_profitable+1
                if long_short == 'Long':
                    if irr > benchmark_return: count_beats_bm = count_beats_bm+1
                else:
                    if irr < benchmark_return: count_beats_bm = count_beats_bm+1
                
            self.metrics['Batting avg'][long_short]['Count_Profitable'] = count_profitable
            self.metrics['Batting avg'][long_short]['Count_Beats_BM'] = count_beats_bm


        bats_long = pd.DataFrame(bats['Long'])
        bats_short = pd.DataFrame(bats['Short'])
        bats_long['Long_Short'] = 'Long'
        bats_short['Long_Short'] = 'Short'
        self.bats = pd.concat([bats_long, bats_short], axis = 0, sort=False)
        self.bats.rename({0:'Ticker', 1:'StartDate',2:'EndDate', 3:'Profits', 4:'Capital', 5:'IRR', 6:'MoM', 7:'BenchmarkR', 8:'StockR'}, axis = 'columns', inplace = True)
                
        
    

    def print_metrics(self, metric):
        
        if super().print_metrics(metric):
            return True
        elif metric == 'Batting avg':        
            for long_short in ['Long', 'Short']:
                print(long_short, 'book: made', self.metrics['Batting avg'][long_short]['Count'],'bats, where',self.metrics['Batting avg'][long_short]['Count_Profitable'],'are profitable,', self.metrics['Batting avg'][long_short]['Count_Beats_BM'],'beat', self.benchmark)
            return True
        else:
            return False        

    
    def add_classifier_multilevel_column(self, df, classifier):
        t = df.copy(deep = True)
        t.columns = pd.MultiIndex.from_tuples([(self.stock_info.loc[k,classifier], k) for k in t.columns], names=[classifier, 'Ticker'])
        return t
    
    def sum_along_ticker_columns_by_classifier(self, df, classifier):
        df = self.add_classifier_multilevel_column(df, classifier)
        return df.groupby(level = classifier, axis = 1).sum()
    
    def add_classifier_multilevel_index(self, df, classifier):
        t = df.copy(deep = True)
        t.index = pd.MultiIndex.from_tuples([(self.stock_info.loc[k,classifier], k) for k in t.index], names=[classifier, 'Ticker'])
        return t
    
    def sum_along_ticker_index_by_classifier(self, df, classifier):
        df = self.add_classifier_multilevel_index(df, classifier)
        return df.groupby(level = classifier, axis = 0).sum()    
    
            
    def __offset_days__(self, current_day, offset):
        current_day = helpers.string_to_date(current_day)
        i = self.date_dict[current_day] + offset
        return self.date_list[i]
        
    def __previous_day__(self, current_day):
        return self.__offset_days__(current_day, -1)
    
    def __next_day__(self, current_day):
        return self.__offset_days__(current_day, 1)
    
