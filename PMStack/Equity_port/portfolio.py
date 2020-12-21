'''
Created on Sep 9, 2017
@author: yewen



'''


from PMStack.Equity_port import portfolio_misc
import warnings
from IPython.core.display import display




    
class Portfolio(portfolio_misc.Portfolio_Misc):
    '''
    A wrapper to rename the class to 'Portfolio'
    '''
    def __init__(self, file_price, file_trades, start_date, end_date=None, capital=None, benchmark = 'SPX', benchmark2=[], base_currency = 'USD', ignore_ticker_list = [], file_stock_info = None, output_dir = '' ):
        super().__init__(file_price, file_trades, start_date, end_date, capital, benchmark, benchmark2, base_currency, ignore_ticker_list, file_stock_info, output_dir)







if __name__ == '__main__':
    
    pass
    
    


    
    



    
    