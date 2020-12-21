'''
Created on Apr 21, 2018

@author: yewen
'''
from scipy import optimize
from datetime import datetime
from scipy import stats
import numpy as np
import warnings


def optionDecompose(option):
    #option = 'ttwo 19jan18 42.0 c'
    s = option.split()
    ticker = s[0]
    expiration = datetime.strptime(s[1], '%d%b%y')  
    strike = float(s[2])
    
    if (s[3] in ['c','C']):
        t = 'Call'
    elif (s[3] in ['p', 'P']):
        t = 'Put'
    
    return ticker, expiration, strike, t

def optionIntrinsicValue(strike, optionType, currentPrice):
    if (optionType in ['CALL', 'Call', 'call', 'c', 'C']):
        return max(currentPrice - strike, 0)
    elif (optionType in ['PUT', 'Put', 'put', 'p', 'C']):
        return max(strike-currentPrice, 0)
    else:
        return 'optionTypeError'


def xnpv(rate,cashflows):
    #return sum([cf/(1+rate)**((t-t0).days/365.0) for (t,cf) in chron_order])
    return sum([cf/(1+rate)**idx for (idx, cf) in enumerate(cashflows)])
    
'''
if MoM == True: also return the money-on-money return
'''
def xirr(cashflows, error_msg = None):
    '''
    get a sound guess for IRR
    '''
    investment = 0
    payment = 0
    weighted_total_investment = 0
    weighted_total_payment = 0
    
    #print(cashflows)
    
    for (idx, cf) in enumerate(cashflows):
        if (cf>0):
            payment += cf
            weighted_total_payment += cf*idx
        elif (cf<0):
            investment -= cf
            weighted_total_investment -= cf*idx
            
    #print('investment and its period:', investment)
    #print('payment and its period:', payment)
    
    if (investment == 0 or payment == 0):
        if error_msg!=None: print('Error occurred in calculating IRR:', error_msg)
        return 0.0, 0.0, 0.0 #irr, mom, investment


    investment_index = weighted_total_investment / investment
    payment_index = weighted_total_payment / payment

    mom = payment / investment - 1.0
    
    guess = (payment/investment)**(1/(payment_index-investment_index)) - 1
    
    '''
    print('investment and its period:', investment, investment_index)
    print('payment and its period:', payment, payment_index)
    print('guess =', guess)
    '''

    try:
        irr = optimize.newton(lambda r: xnpv(r,cashflows),guess)
    except RuntimeError as re:
        warnings.warn(str(re))
        if error_msg!=None: print('Error occurred in calculating IRR:', error_msg)
        irr = 0.0
        
    return irr, mom, investment



def beta(stock_return, benchmark_return):
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(benchmark_return, stock_return)
    #print('slope =', slope, 'intercept =', intercept, 'r_value =', r_value, 'p_value =', p_value, 'std_err =', std_err)
    return slope
        
    '''
    # The other way
    X = np.stack((X,Y),axis=0)
    cov = np.cov(X)
    print(cov[0][1]/cov[0][0])
    '''

def sharp(stock_return, benchmark_return = 0):
    excess_return = stock_return - benchmark_return
    return np.mean(excess_return) / np.std(excess_return)
    
    


if __name__ == '__main__':
    cashflows = [-100, 120, 0, 0, 0, 0, 0, 0, 0, 0, -100, 120]
    print (xirr(cashflows, True))

    cashflows = [-100, 120, -100, 120]
    print (xirr(cashflows, True))
    
    

    
    