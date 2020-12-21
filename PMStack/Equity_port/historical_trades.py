'''
Created on Oct 10, 2018

@author: yewen
'''
import pandas as pd


def get_trade_history (workbook_name, sheet_name, accounts = None):
    
    xl = pd.ExcelFile(workbook_name)
    sheet = xl.parse(sheet_name)
    
    
    
    x = sheet['TransactionType']
    x = x.dropna(axis = 'index', how='any')
    trades = sheet.loc[x.index].reset_index()

    if accounts is not None:
        trades = trades[trades['Account'].map(lambda x: x in accounts)]


    def tx_type_convert(x):
        x = str(x).upper()
        if x == 'SHORT':
            x = 'SELL'
        elif x in ['LONG', 'COVER']:
            x = 'BUY'
        return x


    trades['TransactionType'] = trades['TransactionType'].map(tx_type_convert)
    trades['Ticker'] = trades['Ticker'].map(lambda x: str(x).upper())
    trades['ToIgnore'] = trades['ToIgnore'].map(lambda x: str(x).upper())
    trades['Currency'] = trades['Currency'].map(lambda x: str(x).upper())
    values = {'Fee':0, 'Quantity':0}
    trades = trades.fillna(value = values)
    trades['Fee'] = trades['Fee'].astype('float32')
    #trades['Quantity'] = trades['Quantity'].astype('int32')
    trades['Quantity'] = trades['Quantity'].astype('float32')
    
    #print(trades)
    #print('hhhhhhhhhhhhhhhhhhhhhhhhhhh')
    
    x = trades['ToIgnore']
    x = (x != 'Y')
    trades = trades.loc[x == True].reset_index()
    trades.drop(columns = ['level_0','index'], inplace = True)
    
    return trades
            
        
# end of get_trade_history        



if __name__ == '__main__':
    pass
    #mywb = 'C:\\Users\\yewen\\OneDrive\\Documents\\Sheets\\PA portfolio tracking v5.5.xlsx'
    
    #print(get_trade_history(mywb, 'US trades'))