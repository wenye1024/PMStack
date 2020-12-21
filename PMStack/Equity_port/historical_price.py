'''
Created on Sep 28, 2018

@author: yewen
'''
import pandas as pd
import numpy as np
import PMStack.Tools.helpers as helpers
import matplotlib.pyplot as plt
from datetime import timedelta

#from pandas.plotting import register_matplotlib_converters


def scopePriceTable(price_table, tickers = None, start_date = None, end_date = None):
    if start_date != None: price_table = price_table.loc[start_date:]
    if end_date != None:  price_table = price_table.loc[: (end_date+timedelta(1))]
    if tickers != None: price_table = price_table[tickers]
    return price_table

def generate_trade_annotation(trades, tickers, last = 3, equity_currency = None, forex = None):
    if last <= 0: return None
    
    xss = []
    textss = []

    cols = ['Quantity','TransactionAmount', 'Currency']
    pos_currency = cols.index('Currency')
    pos_TransAmt = cols.index('TransactionAmount')

    for ticker in tickers:
        a = trades[trades['Ticker']==ticker].copy()
        a = a[(a['TransactionType']=='BUY') | (a['TransactionType']=='SELL') | (a['TransactionType']=='COVER') | (a['TransactionType']=='LONG') | (a['TransactionType']=='SHORT')][cols]

        # switch to local currency
        if equity_currency is not None:
            curncy = equity_currency[ticker]
            if curncy != 'USD':

                for i in range(len(a.index)):
                    
                    if a.iloc[i, pos_currency] == 'USD':
                        a.iloc[i, pos_currency] = curncy
                        a.iloc[i, pos_TransAmt] /= forex.loc[a.index[i], curncy]


        # Combining trades within 3 days of one another
        ll = len(a.index)
        i = ll-1
        while i>0:
            j = i-1
            while j>=0:
                if (a.index[i] - a.index[j]).days <= 3:
                    a.iloc[i] += a.iloc[j]
                    a.iloc[j,:] =0
                    j = j-1
                else:
                    i = j
                    break
            if j<0: break

        a = a[a['Quantity']!=0]

        ll = len(a.index)
        if last > ll: last = ll

        xs = []
        texts = []
        for i in range( -last, 0):
            if a.iloc[i,]['Quantity'] > 0: trans_type = 'Bought'
            else: trans_type = 'Sold'
            s = '{:} {:,.0f} {:} @{:,.2f}'.format(trans_type, np.abs(a.iloc[i,]['Quantity']),ticker, -a.iloc[i,]['TransactionAmount']/a.iloc[i,]['Quantity'])
            xs.append(a.index[i])
            texts.append(s)
        xss.append(xs)
        textss.append(texts)
    return[xss, textss]



def combine_annotationss(ann1, ann2):
    if ann1 is None: return ann2
    if ann2 is None: return ann1
    
    xss = ann1[0]
    textss = ann1[1]
    for i, xs in enumerate(xss):
        ann2[0][i] += xs
        ann2[1][i] += textss[i]

    return ann2


def plotPriceNormalized(price_table, tickers, names = None, start_date = None, end_date = None, colors = None, show = False, plot_type = 'line', annotationss=None, trades = None, annotate_trades = 3):
    price_table = scopePriceTable(price_table, tickers, start_date, end_date).copy(deep = True)    
    price_table = price_table.apply(lambda x: x*100/x.iloc[0])
    
    plotPrice(price_table, tickers, names = names, colors = colors, show = show, plot_type = plot_type, annotationss = annotationss, trades = trades, annotate_trades = annotate_trades )
    

def plotPriceRelative(price_table, ticker_pairs, names = None, start_date = None, end_date = None, colors = None, show = False, plot_type = 'line', annotationss=None, trades = None, annotate_trades = 3):
    price_table = scopePriceTable(price_table, start_date = start_date, end_date = end_date)
    
    tickers = []
    old_tickers = []
    if names is None:
        for pair in ticker_pairs:
            tickers.append(pair[0] + ' / ' + pair[1])
            old_tickers.append(pair[0])
    else:
        for pair in ticker_pairs:
            tickers.append(names.loc[pair[0],'Name'] + ' / ' + names.loc[pair[1],'Name'])
            old_tickers.append(names.loc[pair[0],'Name'])

        
    
    table = pd.DataFrame(index = price_table.index.copy(), columns=tickers)
    
    for idx, pair in enumerate(ticker_pairs):
        table[tickers[idx]] = price_table[pair[0]] / price_table[pair[1]]
    
    
    if trades is not None:
        ann2 = generate_trade_annotation(trades, old_tickers, annotate_trades)
        annotationss = combine_annotationss(annotationss, ann2)

    
    plotPriceNormalized(table, tickers, colors = colors, show = show, plot_type = plot_type, annotationss = annotationss )
    

def plotAnnotate(xss, textss,yss, colors):
    xmin, xmax, ymin, ymax = plt.axis()
    offset = (ymax-ymin)/20
    for idx, xs in enumerate(xss):
        for idxx, x in enumerate(xs):
            plt.annotate(textss[idx][idxx], xy=(x, yss[idx][idxx]), xytext=(x, yss[idx][idxx]+offset), xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3", facecolor=colors[idx]))
            


def plotPrice(price_table, tickers, names = None, start_date = None, end_date = None, colors = None, show = False, plot_type = 'line' , annotationss=None, trades = None, annotate_trades = 3):
    ll = len(tickers)
    if ll>9:
        print('Only supports up to 9 tickers')
        return
    
    if colors == None:
        colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']
    
    price_table = scopePriceTable(price_table, start_date = start_date, end_date = end_date)
    
    
    x = price_table.index
    
    if plot_type != 'line':    
        high = -float('inf')
        low = - high
    
    for i, ticker in enumerate(tickers):
        
        y = price_table.loc[:, ticker]  
        
        
        if names is None:
            name = ticker
        else:
            name = names.loc[ticker,'Name']
          

        if plot_type == 'line':
            plt.plot(x, y, color = colors[i], label = name)
        else:
            low = min(low, min(y))
            high = max(high, max(y))

            if plot_type == 'area': plt.fill_between(x, y, color = colors[i], alpha = 0.2, label = name)
            elif plot_type == 'bar': plt.bar(x, y, color = colors[i], alpha = 0.2, label = name)

    if plot_type != 'line':
        plt.ylim([np.ceil(low-0.5*(high-low)), np.ceil(high+0.5*(high-low))])

    plt.legend(loc='upper left')
    
    
    def __plot_annotationss__(ann):
        xss = ann[0]
        textss = ann[1]
        yss = []
        for i, ticker in enumerate(tickers):
            ys = []
            for x in xss[i]:
                y = price_table.loc[x, ticker]
                ys.append(y)
            yss.append(ys)
        plotAnnotate(xss, textss, yss, colors)  
    
    
    if annotationss is not None:
        __plot_annotationss__(annotationss)

    if trades is not None:
        annotationss = generate_trade_annotation(trades, tickers, annotate_trades)
        __plot_annotationss__(annotationss)
    
    if show:
        plt.show()



def define_equal_weight_daily_rebalance_port(new_ticker, tickers, price_table, daily_return_table, names):
    daily_return_table[new_ticker] = daily_return_table[tickers].mean(axis = 1)
    daily_return_table[new_ticker].iloc[0] = 0.0
    price_table[new_ticker] = (daily_return_table[new_ticker]+1).cumprod()
    names.loc[new_ticker] = new_ticker


def define_relative_strength_series(new_ticker, ticker1, ticker2, price_table, daily_return_table, names):
    price_table[new_ticker] = price_table[ticker1] / price_table[ticker2]
    price_table[new_ticker] = price_table[new_ticker] / price_table[new_ticker].iloc[0]
    daily_return_table[new_ticker] = price_table[new_ticker].pct_change()
    daily_return_table[new_ticker].iloc[0] = 0.0
    names.loc[new_ticker] = new_ticker
    

def __define_cash_port__(price_table, daily_return_table, names, new_ticker = 'Cash_Port'):
    daily_return_table[new_ticker] = 0.0
    price_table[new_ticker] = 1.0
    names.loc[new_ticker] = new_ticker


def define_day_one_levered_series(new_ticker, ticker, leverage, price_table, daily_return_table, names):
    cash = 1 - leverage
    price_table[new_ticker] = price_table[ticker] / price_table[ticker].iloc[0] * leverage + cash
    daily_return_table[new_ticker] = price_table[new_ticker].pct_change()
    daily_return_table[new_ticker].iloc[0] = 0.0
    names.loc[new_ticker] = new_ticker


'''
range does not include the last row/column
'''
def get_price_history (workbook_name='Historical price.xlsx', sheet_name='Historical price', print_out = True, base_currency = 'USD'):        

    
    xl = pd.ExcelFile(workbook_name)
    sheet = xl.parse(sheet_name)
    sheet.reset_index(inplace = True)
    sheet.set_index('Index', inplace = True)
    sheet.drop(columns = ['index'], inplace = True)
    #print(sheet)
    
    col = sheet.loc['Name',:].isnull()
    
    currencies = col[col==True].index.tolist()
    #print(currencies)
    forex = sheet[currencies].dropna(axis = 'index', how='any')
    

    # remove rows with #Calc
    valid_dates = (forex!='#Calc').all(axis = 1)
    #print(self.forex)
    forex = (forex[valid_dates]).astype('float32')

    forex.index = pd.to_datetime(forex.index)
    forex.loc[:, 'USD'] = 1.0

    if base_currency != 'USD':
        forex = forex.div(forex[base_currency], axis=0)

    tickers = col[col==False].index.tolist()        
    sheet_equity = sheet[tickers]#.dropna(axis = 'index', how='any')

    company_name = pd.DataFrame(sheet_equity.loc['Name',:])
    company_name.columns = ['Name']
    company_name.index.name = 'Ticker'
    
    
    
    equity_currency = sheet_equity.loc['Currency',:]
    
    sheet_equity = sheet_equity.drop(index = ['Name', 'Currency'])
    
    # remove rows with #Calc
    valid_dates = (sheet_equity!='#Calc').all(axis = 1)

    equity_prices_local = (sheet_equity[valid_dates]).astype('float32')        
    equity_prices_local.index = pd.to_datetime(equity_prices_local.index)
    equity_prices_local.index.name = 'Date'

    
    equity_daily_return_local = equity_prices_local.pct_change(1) #.dropna(axis = 0)

    equity_prices_USD = equity_prices_local.copy()
    for i, v in equity_currency.iteritems():
        if (v!=base_currency):
            equity_prices_USD[i] = equity_prices_local[i] * forex[v]
    
    equity_daily_return_USD = equity_prices_USD.pct_change(1)


    if print_out:
        start_date = helpers.date_to_string(forex.index[0]) #pd.Timestamp(forex.index[0]).strftime('%Y-%m-%d') 
        end_date = helpers.date_to_string(forex.index[-1]) #pd.Timestamp(forex.index[-1]).strftime('%Y-%m-%d')
        print('Loaded price history from', start_date, 'to', end_date)
        print()
    
    
    __define_cash_port__(equity_prices_local, equity_daily_return_local, company_name)
    
    r = {}
    r['forex'] = forex
    r['company_name'] = company_name
    r['equity_currency'] = equity_currency
    r['equity_price_local'] = equity_prices_local
    r['equity_price_USD'] = equity_prices_USD
    r['equity_daily_return_local'] = equity_daily_return_local
    r['equity_daily_return_USD'] = equity_daily_return_USD
    r['end_date'] = pd.Timestamp(forex.index[-1])
    
    #return forex, company_name, equity_currency, equity_prices_local, equity_daily_return_local, equity_prices_USD, equity_daily_return_USD
    return r


if __name__ == '__main__':
    print('Running historical_price.py ...')
    mywb = 'P:\\GitHub\\PA_Portfolio\\PA_Portfolio\\Historical price.xlsx'
    file_price = {'workbook_name':mywb, 'sheet_name':'Historical price', 'base_currency':'USD'}
    r = get_price_history(**file_price)
    #r = r['equity_price_local'].iloc[-1]
    #print(r)
    pass