'''
Created on Oct 10, 2018

@author: yewen
'''
import pandas as pd


def get_stock_info (workbook_name, sheet_name):
    
    xl = pd.ExcelFile(workbook_name)
    sheet = xl.parse(sheet_name)
    
    
    
    #x = sheet['TransactionType']
    sheet.dropna(subset=['Ticker'], inplace = True)
    
    sheet['Ticker'] = sheet['Ticker'].map(lambda x: str(x).upper())
    sheet.set_index('Ticker', inplace = True)
    
    
    return sheet
            
        
# end of get_trade_history        



if __name__ == '__main__':
    mywb = 'C:\\Users\\ye.wen\\Downloads\\Paper portfolio tracking v1.1.xlsx'
    
    print(get_stock_info(mywb, 'Latest price'))