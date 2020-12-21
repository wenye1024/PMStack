'''
Created on Sep 25, 2019

@author: ye.wen
'''

import pandas as pd
import numpy as np
from Tax import equity_tax
from Crypto_port import crypto_price_history
from IPython.core.display import display


def add_historical_price_column(sheet, crypto_price_USD, coin_list = None):
    sheet['HistPx'] = 0.0
    for idx in sheet.index:
        d = sheet.loc[idx, 'Date'].normalize()
        ticker = sheet.loc[idx, 'Ticker']
        if (coin_list is None) or ((coin_list is not None) and (ticker in coin_list)):
            px = crypto_price_USD[ticker].loc[d, 'Close']
            sheet.loc[idx,'HistPx'] = px
        


def process_crypto_pair_trades(sheet, crypto_price_USD):
    sheet_copy = sheet.copy()
    crypto_pair_trades = sheet_copy[sheet_copy['Currency']!='USD'].copy()
    for idx in crypto_pair_trades.index:
        d = crypto_pair_trades.loc[idx, 'Date'].normalize()
        ticker = crypto_pair_trades.loc[idx, 'Currency']
        amt = crypto_pair_trades.loc[idx, 'TransactionAmount']
        px = crypto_price_USD[ticker].loc[d, 'Close']

        sheet_copy.loc[idx, 'TransactionAmount'] = amt * px
        sheet_copy.loc[idx, 'Currency'] = 'USD'

        crypto_pair_trades.loc[idx, 'Quantity'] = amt
        crypto_pair_trades.loc[idx, 'Balance'] = amt
        crypto_pair_trades.loc[idx, 'Ticker'] = ticker
        crypto_pair_trades.loc[idx, 'TransactionAmount'] = - amt * px
        crypto_pair_trades.loc[idx, 'Currency'] = 'USD'
        crypto_pair_trades.loc[idx, 'TransactionType'] = equity_tax.inverse_transaction_type(crypto_pair_trades.loc[idx, 'TransactionType'])
        crypto_pair_trades.loc[idx, 'HistPx'] = px

    df = pd.concat([sheet_copy, crypto_pair_trades])
    df.index.rename('TrxOldIndex', inplace = True)
    return df.sort_index(axis=0).reset_index()

if __name__ == '__main__':
    
    writer = pd.ExcelWriter('C:\\crypto_tax.xlsx')

    
    coin_list = ['BTC', 'ETH', 'BCH', 'XLM', 'XMR']
    crypto_price_USD = crypto_price_history.load_historical_price_files(coin_list)
    sheet = equity_tax.load_trades_file('C:\\All crypto transaction history - v3.xlsx', 'All')
    add_historical_price_column(sheet, crypto_price_USD, coin_list)
    
    sheet.to_excel(writer, 'transactions')
    
    
    new_sheet = process_crypto_pair_trades(sheet, crypto_price_USD)
    form8949_st, form8949_lt = equity_tax.update_realized_gains(new_sheet, writer = writer)


    '''
    The function aggregate_f8949 ignores transactions where account is N/A
    '''
    equity_tax.aggregate_f8949(form8949_st, False).to_excel(writer, 'F8949_ST_AGG')
    equity_tax.aggregate_f8949(form8949_lt, False).to_excel(writer, 'F8949_LT_AGG')
    
    
    tables_to_print = []
    table = equity_tax.summarize_realized_gains(new_sheet, start_year = 2010, writer = writer, tables_to_print = tables_to_print)
    display(table)
    writer.save()
    
    
    pass

    