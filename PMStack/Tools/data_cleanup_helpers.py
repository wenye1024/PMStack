'''
Created on Aug 28, 2019

@author: ye.wen
'''
import pandas as pd
import math


def make_first_column_as_index(sheet):
    sheet.set_index([sheet.columns[0]], inplace = True)
    return sheet

def float_to_string(f):
    if type(f) is str:
        return f
    elif math.isnan(f):
        return ''
    else:
        return str(f)    

def convert_number_if_applicable(s):
    ssss = s
    ss = ssss
    neg = False
    pct = False
    if ssss == '—':
        return 0.0
    if ssss == '-':
        return 0.0
    if ssss[-1]==')':
        if (ssss[0]=='('):
            ssss = ssss[1:-1]
            neg = True
        elif (ssss[-3:] in ['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)']) and (len(ssss)>3):
            ssss = ssss[:-3]
    elif (ssss[0]=='(') and ((ssss[-2:]==')%') or (ssss[-2:]=='%)')):
        ssss = ssss[1:-2]
        neg = True
        pct = True
    try:
        f = float(ssss.replace(',',''))
        if neg: f = -f
        if pct: f = f/100.0
        return f
    except ValueError:
        return ss


def make_first_row_as_header(sheet):
    new_header = sheet.iloc[0]
    s = sheet[1:]
    s.columns = new_header
    return s

def div_1000(x):
    if (type(x) is float) or (type(x) is int):
        return x/1000.0
    else:
        return x

def fix_column_offsets(df, starting_column = 1, print_out = True):
    
    df = df.copy(deep = True)
    cols_to_drop = []
    
    cols_minus_one = len(df.columns) -1
    rows = len(df.index)
    col = starting_column
    while col < cols_minus_one:
        
        complementary = True
        for r in range(0, rows):
            if not(pd.isna(df.iloc[r,col]) or pd.isna(df.iloc[r,col+1])):
                complementary = False
                break
        
        if complementary:
            for r in range(0, rows):
                if pd.isna(df.iloc[r,col]):
                    df.iloc[r,col] = df.iloc[r,col+1]
            cols_to_drop.append(df.columns[col+1])
            col = col+2
        else:
            col = col+1
    
    if print_out: print('Columns dropped:', cols_to_drop)    
    return df.drop(cols_to_drop, axis=1)

def fix_column_offsets_specified_columns(df, columns, print_out = True):
    
    df = df.copy(deep = True)
    cols_to_drop = []
    
    rows = len(df.index)

    for col in columns:
    
        complementary = True
        for r in range(0, rows):
            if not(pd.isna(df.iloc[r,col]) or pd.isna(df.iloc[r,col+1])):
                complementary = False
                break
        
        if complementary:
            for r in range(0, rows):
                if pd.isna(df.iloc[r,col]):
                    df.iloc[r,col] = df.iloc[r,col+1]
            cols_to_drop.append(df.columns[col+1])
    
    if print_out: print('Columns dropped:', cols_to_drop)    
    return df.drop(cols_to_drop, axis=1)



if __name__ == '__main__':
    pass