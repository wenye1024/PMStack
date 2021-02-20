'''
Created on Sep 9, 2017

@author: yewen
'''

import pandas as pd
import numpy as np
import Equity_port.historical_trades as trade_history
import Equity_port.historical_price as historical_price
import Tools.helpers as helpers
import warnings
from IPython.display import display


def update_unrealized_gains (sheet, equity_prices_local, current_date):
    if current_date is None:
        print('Parameter current_date cannot be None')
        return
    
    sheet1 = sheet[(sheet['OpenClose']=='O') & ((sheet['Balance']>0.0001) | (sheet['Balance']<-0.0001))].copy()

    sheet1['Valid'] = True
    for i in sheet1.index:
        sheet1.loc[i, 'Valid'] = sheet1.loc[i, 'Ticker'] in equity_prices_local.index
        if sheet1.loc[i, 'Valid'] == False:
            print('Ignoring a trade involving', sheet1.loc[i, 'Ticker'])
    sheet1 = sheet1[sheet1['Valid']]
    #sheet1 = sheet1[(sheet1['Balance']>0.1) | (sheet1['Balance']<-0.1)]
    #sheet1 = sheet1[(sheet1['SecurityType'] == 'Equity') | (sheet1['SecurityType'] == 'ETF')]

    #d = sheet1[['Ticker','Balance']]
    sheet1['Price'] = sheet1.apply(lambda x: equity_prices_local[x['Ticker']], axis = 1)
    sheet1['BalanceAmount'] = sheet1['Price'] * sheet1['Balance']
    
    
    sheet1['Unrealized_Gain'] = sheet1['TransactionAmount'] / sheet1['Quantity'] * sheet1['Balance'] + sheet1['BalanceAmount']
    

    sheet1['Days'] = sheet1.apply(lambda x: (current_date - x['Date']).days, axis =1)
    sheet1['Unrealized_LTG'] = sheet1.apply(lambda x: x['Unrealized_Gain'] if x['Days']>365 else 0.0, axis = 1)
    #sheet2 = sheet1[sheet1['Days'] > 365]
    #sheet2['Unrealized_LTG'] = sheet2['Unrealized_Gain']
    sheet1['Unrealized_STG'] = sheet1['Unrealized_Gain'] - sheet1['Unrealized_LTG']

    sheet['Unrealized_LTG'] = sheet1['Unrealized_LTG']
    sheet['Unrealized_STG'] = sheet1['Unrealized_STG']
    
    #sheet1.drop(columns = ['Unrealized_Gain', 'Price', 'BalanceAmount'], inplace=True)
    #print(sheet1['Unralized_LTG'])
    #sheet2['Unrealized_LTG'] = sheet2['Unrealized_Gain']
    #sheet1['Unrealized_LTG'] = sheet2['Unrealized']
    #print(sheet1.head())
    #sheet['Unrealized_LTG'] = sheet1['Unrealized_LTG']

def inverse_transaction_type(s):
    if s == 'BUY':
        return 'SELL'
    elif s == 'SELL':
        return 'BUY'
    else:
        return None

        


def update_realized_gains (sheet, print_trades = False, tax_year = None, writer = None):
    
    starting_rows = {}
    starting_rows['BUY'] = {}
    starting_rows['SELL'] = {}
    position = {}
    header = ['Asset','DateAcquired', 'DateDisposed', 'Proceeds', 'CostBasis','f','g', 'Profits','IntentionallyEmpty','Ticker', 'Quantity','PxAcquired','PxDisposed', 'Year','AcctAcquired', 'AcctDisposal']
    form8949_st = [header]
    form8949_lt = [header]
    
    
    for idx in sheet.index:
        
        ticker = sheet.loc[idx,'Ticker']
        acct_disposal = sheet.loc[idx,'Account']
        t_date = sheet.loc[idx, 'Date']
        dollar = sheet.loc[idx, 'TransactionAmount'].astype('float32')
        quantity = -sheet.loc[idx, 'Quantity'].astype('float32')
        price = dollar/quantity
            
        
        if(sheet.loc[idx, 'TransactionType'] in ['SELL', 'BUY', 'SELL-LOST']):
            t_type = sheet.loc[idx, 'TransactionType']
            if t_type == 'SELL-LOST':
                t_type = 'SELL'
                sell_lost = True
            else:
                sell_lost = False
            
        else:
            continue
        
         
        inverse_type = inverse_transaction_type(t_type)

       
        if ticker not in position:
            trade = 'O'
            position[ticker] = -quantity
        elif helpers.isZero(position[ticker]):
            trade = 'O'
            position[ticker] = -quantity
        elif position[ticker] * quantity > 0:

            prev_pos = position[ticker]
            position[ticker] -= quantity

            if position[ticker] * prev_pos < 0:
                trade = 'CO'
                quantity = prev_pos
                print('Encountered a trade that flips long/short:', t_date, t_type, -quantity, ticker)
            else:
                trade = 'C'
            
        else:
            trade = 'O'
            position[ticker] -= quantity
        
        sheet.loc[idx,'OpenClose'] = trade

        if trade in ['O','CO']:
            if (ticker not in starting_rows[t_type]):
                starting_rows[t_type][ticker] = 0
        
        if trade in ['C','CO']:
            if print_trades:
                print()
                if (t_type == 'BUY'):
                    to_print = 'COVER ' + ticker + ':'
                else:
                    to_print = 'SELL ' + ticker + ':'
                print(to_print, str(abs(quantity)) + ' shares on ' + pd.Timestamp(t_date).strftime('%Y-%m-%d') + ' at {:.2f}'.format(price))
            short_term_gain = 0
            long_term_gain = 0
            
            note = 'row, quantity: '

            # for debug
            #print(starting_rows)
            #print(ticker)
            #print(inverse_type)
            #print(idx)

            for row1 in range(starting_rows[inverse_type][ticker], idx):

                if sheet.loc[row1,'Ticker'] != ticker:
                    continue
                if sheet.loc[row1, 'TransactionType'] != inverse_type:
                    continue
                if sheet.loc[row1,'OpenClose'] != 'O':
                    continue
                if sheet.loc[row1,'Balance'] != 0:
                    o_date = sheet.loc[row1, 'Date']
                    o_price = - sheet.loc[row1, 'TransactionAmount'].astype('float32') / sheet.loc[row1, 'Quantity']
                    balance = sheet.loc[row1,'Balance']
                    diff_days = (t_date - o_date).days
                    acct_acquired = sheet.loc[row1,'Account']
                    
                    if quantity>0:
                        var_amount = min (quantity, balance)
                    else:
                        var_amount = max (quantity, balance)
                    balance -= var_amount
                    quantity -= var_amount
                    
                    note += (str(row1) + ' ' + str(var_amount) +'; ')
                    
                    proceeds = var_amount * price
                    cost_basis = var_amount * o_price
                    if sell_lost:
                        profits = 0
                    else:
                        profits = proceeds - cost_basis
                    
                    
                    asset = '{:.5f} {:}'.format(var_amount, ticker)

                    new_row = []

                    if tax_year is not None:
                        if  t_date.year != tax_year:
                            new_row = None

                    if new_row is not None:
                        new_row = [asset, o_date, t_date, proceeds, cost_basis,'','', profits,'',ticker, var_amount, o_price, price, t_date.year, acct_acquired, acct_disposal]

                        if print_trades: 
                            if inverse_type == 'BUY':
                                to_print = '    bought ' + str(var_amount) + ' shares '
                            else:
                                to_print = '    sold short ' + str(-var_amount) + ' shares '
                            to_print += ('at ' + ' at {:.2f}'.format(o_price) + ' on ' + pd.Timestamp(o_date).strftime('%Y-%m-%d')  + ', closed position after ' + str(diff_days) + ' days,')
                    
                    
                    if (diff_days <=365):
                        short_term_gain += profits
                        if new_row is not None:
                            form8949_st.append(new_row)
                            if print_trades: 
                                print(to_print, 'with a short term gain of {:.2f}'.format(var_amount * (price - o_price)))
                    else:
                        long_term_gain += profits
                        if new_row is not None:
                            form8949_lt.append(new_row)
                            if print_trades: 
                                print(to_print, ' with a long term gain of {:.2f}'.format(var_amount * (price - o_price)))
                    
                    sheet.loc[row1,'Balance'] = balance
                    
                    if helpers.isZero(quantity):
                        if helpers.isZero(balance):
                            starting_rows[inverse_type][ticker] = row1+1
                        else:
                            starting_rows[inverse_type][ticker] = row1
                        break
        
            sheet.loc[idx,'Realized_STG'] = short_term_gain
            sheet.loc[idx,'Realized_LTG'] = long_term_gain
            sheet.loc[idx,'Note'] = note
            
            if print_trades:
                if position[ticker] != 0:
                    print("    still holding {:.0f} shares".format(position[ticker]))
                else:
                    print("    fully exited the position")
    
    form8949_st = helpers.make_first_row_as_header(pd.DataFrame(form8949_st))
    form8949_lt= helpers.make_first_row_as_header(pd.DataFrame(form8949_lt))

    if writer is not None:
        sheet.to_excel(writer, 'Tax')
        form8949_st.to_excel(writer, 'F8949_ST')
        form8949_lt.to_excel(writer, 'F8949_LT')

    return form8949_st, form8949_lt



'''
if by_same_execution_price is False, then aggregation by date only
'''
def aggregate_f8949(f8949, by_same_execution_price):

    x = f8949.copy()

    date_only = lambda d: d.normalize()
    x['DateDisposed']=x['DateDisposed'].map(date_only)
    x['DateAcquired']=x['DateAcquired'].map(date_only)

    '''
    def multiply_by_100(f):
        return int(round(f*100.0,0))

    def divide_by_100(f):
        return float(f)/100.0
    '''
    if by_same_execution_price:    
        multiply_by_100 = lambda f: int(round(f*100.0,0))
        x['PxAcquired'] = x['PxAcquired'].map(multiply_by_100)
        x['PxDisposed'] = x['PxDisposed'].map(multiply_by_100)
    
        x = pd.pivot_table(x,index=['DateAcquired','DateDisposed', 'Ticker','Year', 'AcctAcquired', 'AcctDisposal', 'PxAcquired', 'PxDisposed'], values=['Proceeds', 'CostBasis', 'Profits', 'Quantity'], 
                       aggfunc=np.sum)
    else:
        x = pd.pivot_table(x,index=['DateAcquired','DateDisposed', 'Ticker','Year', 'AcctAcquired', 'AcctDisposal'], values=['Proceeds', 'CostBasis', 'Profits', 'Quantity'], 
                        aggfunc=np.sum)
        
    x = x.reset_index()
    x['Asset'] = ''
    for i in x.index:
        x.loc[i,'Asset'] = '{:.5f} {:}'.format(x.loc[i,'Quantity'], x.loc[i,'Ticker'])
    x['f'] = ''
    x['g'] = ''
    x['IntentionallyEmpty'] = ''

    if by_same_execution_price:
        divide_by_100 = lambda f: float(f)/100.0
        x['PxAcquired'] = x['PxAcquired'].map(divide_by_100)
        x['PxDisposed'] = x['PxDisposed'].map(divide_by_100)
    else:
        x['PxAcquired'] = x['CostBasis'] / x['Quantity']
        x['PxDisposed'] = x['Proceeds'] / x['Quantity']
    
    
    x= x[['Asset','DateAcquired','DateDisposed','Proceeds','CostBasis','f','g', 'Profits','IntentionallyEmpty', 'Ticker', 'Quantity', 'PxAcquired', 'PxDisposed', 'Year', 'AcctAcquired', 'AcctDisposal']]
    
    
    return x.set_index('Asset')






def summarize_realized_gains(sheet, tax_year, writer = None, tables_to_print = None, print_out = False):
    sheet['Year'] = (sheet['Date'].apply(lambda x: x.year)).astype(int)
    sheet1 = sheet[(sheet['OpenClose']=='C') & (sheet['Year'] == tax_year)]   
    
    table = pd.pivot_table(sheet1, values=['Realized_STG','Realized_LTG'], index=['TransactionType','Ticker','Currency'],aggfunc=np.sum)
    table.rename(index = {'SELL':'LONG', 'BUY':'SHORT'}, inplace = True)
    if writer is not None:
        table.to_excel(writer, 'Realized by ticker')
       
    lt_or_st = ['Realized_STG','Realized_LTG']
    
    for i in table.index.levels[0]: #LONG and SHORT book
        try:
            tt = table.loc[i]
            for j in lt_or_st:
                ttt = tt[j].sort_values(ascending = False)
                ttt = ttt[(ttt>1000) | (ttt<-1000)]
                if ttt.empty:
                    continue
                if print_out:                
                    print()
                    if (j == 'Realized_STG'):
                        print('Year', tax_year, i,'book short term gain or loss greater than 1,000:')
                    else:
                        print('Year', tax_year, i,'book long term gain or loss greater than 1,000:')                    
                    helpers.printFloatingPointTable(ttt)
                if tables_to_print is not None:
                    tables_to_print[j+'_'+i] = ttt
                    
        except KeyError:
            pass
            
    table = pd.pivot_table(table, values=['Realized_STG','Realized_LTG'], index=['TransactionType','Currency'],aggfunc=np.sum)
    if writer is not None:
        table.to_excel(writer, 'Realized Summary')

    
    
    return table


def summarize_unrealized_gains(sheet, writer = None, output_tables = None): 

    sheet1 = sheet[(sheet['OpenClose']=='O') & ((sheet['Balance']>0.1) | (sheet['Balance']<-0.1))]
    table1 = pd.pivot_table(sheet1, values=['Unrealized_STG','Unrealized_LTG'], index=['TransactionType','Ticker','Currency'],aggfunc=np.sum)
    table1.rename(index = {'SELL':'SHORT', 'BUY':'LONG'}, inplace = True)

    def unrealized_gain_balance_by_stock(str):
        sheet2 = sheet1[(sheet1[str]<-0.1) | (sheet1[str]>0.1)]
        table2 = pd.pivot_table(sheet2, values=[str,'Balance'], index=['TransactionType','Ticker','Currency'],aggfunc=np.sum)
        table2.rename(index = {'SELL':'SHORT', 'BUY':'LONG'}, inplace = True)
        table2.rename(columns = {'Balance': str+'_Share_#'}, inplace = True)
        return table2

    table2_lt = unrealized_gain_balance_by_stock('Unrealized_LTG')
    table2_st = unrealized_gain_balance_by_stock('Unrealized_STG')
    table2 = table2_lt.join(table2_st, how='outer').fillna(0)

    if writer is not None: 
        table2.to_excel(writer, 'Unrealized by ticker')
    
    if output_tables is not None:
        output_tables['Unrealized_Gains'] = table2
    
    table1 = pd.pivot_table(table1, values=['Unrealized_STG','Unrealized_LTG'], index=['TransactionType','Currency'],aggfunc=np.sum)
    return table1






def calculate_gains(trades, equity_prices_local, forex, tax_year, date_for_unrealized_gain = None, output_excel_name = None, account = None):
    
    if account is not None:
        sheet = trades[trades['Account'] == account].reset_index()
        sheet.drop(columns=['index'], inplace = True)
    else:
        sheet = trades.copy()
    
    date_for_unrealized_gain = pd.to_datetime(date_for_unrealized_gain)
    
    
    form8949_st, form8949_lt = update_realized_gains(sheet, print_trades=False, tax_year = tax_year)
    update_unrealized_gains(sheet, equity_prices_local, date_for_unrealized_gain) 
    
    if output_excel_name is not None:
        writer = pd.ExcelWriter(output_excel_name)
        sheet.to_excel(writer, 'Tax')
        form8949_st.to_excel(writer, 'F8949_ST')
        form8949_lt.to_excel(writer, 'F8949_LT')
    else:
        writer = None
    warnings.warn('Gains from option and forex transactions are not included')
    
    output_tables = {}
    output_tables['F8949_ST'] = form8949_st
    output_tables['F8949_LT'] = form8949_lt
    
    table = summarize_realized_gains(sheet, tax_year, writer, output_tables)
    table1 = summarize_unrealized_gains(sheet, writer, output_tables)
    table = table.join(table1, how='outer')
    
    

    long_or_short = ['LONG','SHORT']
    
    for i in long_or_short:
        t = table.loc[i]
        for j in t.index:
            t.loc[j,:] *= forex[j]
    table = pd.pivot_table(table, values=['Realized_STG','Realized_LTG', 'Unrealized_STG','Unrealized_LTG'], index=['TransactionType'],aggfunc=np.sum) 
    
    table['Realized'] = table['Realized_LTG'] + table['Realized_STG']
    table['Unrealized'] = table['Unrealized_LTG'] + table['Unrealized_STG']
    table.loc['Total'] = table.loc['LONG'] + table.loc['SHORT']
    
    if output_excel_name is not None:
        table.to_excel(writer, 'Summary')
        writer.save()

 
    helpers.printFloatingPointTable(table)
    output_tables['Summary'] = table
    return output_tables


def load_trades_file(filename, sheetname):
    sheet = trade_history.get_trade_history(filename, sheetname)
    
    x = sheet['TransactionType']
    
    #Sell-Lost is for crypto, representing that the coins are lost. So it's treated mathmatically as selling for zero.
    x = (x == 'BUY') | (x == 'SELL') | (x == 'LONG') | (x == 'SHORT') | (x == 'COVER') | (x == 'SELL-LOST') 
    
    sheet = sheet.loc[x == True].reset_index()

    # ignored forex trade. However, the algo theoretically supports forex - treating say "USD.JPY" as an equity.
    sheet = sheet[sheet['SecurityType'] != 'Forex'].reset_index()
    sheet = sheet[['TransactionType', 'Ticker','Date','TransactionAmount','Currency','Quantity','Account']]
    sheet['Balance'] = sheet['Quantity']
    return sheet


def load_price_file(filename):
    r = historical_price.get_price_history(workbook = filename, print_out = False)
    equity_prices_local = r['equity_price_local'].iloc[-1]
    forex = r['forex'].iloc[-1]
    return equity_prices_local, forex




if __name__ == '__main__':
    #sheet = load_trades_file('P:\\Example portfolio tracking.xlsx', 'US trades')
    #equity_prices_local, forex = load_price_file('P:\\Historical price.xlsx')
    #output_excel_name = 'P:\\tax.xlsx'
    #tables_to_print = calculate_gains(sheet, equity_prices_local, forex, 2020, '2020-04-20', output_excel_name = output_excel_name) 
    
    pass

    