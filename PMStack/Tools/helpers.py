'''
Created on Dec 8, 2018

@author: yewen
'''
import pandas as pd
import numpy as np
from IPython.display import display


def make_first_row_as_header(sheet):
    new_header = sheet.iloc[0]
    s = sheet[1:]
    s.columns = new_header
    return s

def printPercentagePointTable(table, columns = None, head = 0, tail = 0, only_print_formatted_columns = True, reset_index=None, print_zero_as=None, print_nan_as=None, to_display = True, decimal = 1):
    format = '{:,.' + str(decimal) + '%}'
    printFormattedTable(table, format, columns=columns, head=head, tail=tail, only_print_formatted_columns = only_print_formatted_columns, reset_index=reset_index, print_zero_as=print_zero_as, print_nan_as=print_nan_as, to_display = to_display)
    
def printFloatingPointTable(table, columns = None, head = 0, tail = 0, only_print_formatted_columns = True, reset_index=None, print_zero_as=None, print_nan_as=None, to_display = True, decimal = 1):
    format = '{:,.' + str(decimal) + 'f}'
    printFormattedTable(table, format, columns=columns, head=head, tail=tail, only_print_formatted_columns = only_print_formatted_columns, reset_index=reset_index, print_zero_as=print_zero_as, print_nan_as=print_nan_as, to_display = to_display)

def printIntegerTable(table, columns = None, head = 0, tail = 0, only_print_formatted_columns = True, reset_index=None, print_zero_as=None, print_nan_as=None, to_display = True):
    printFormattedTable(table, "{:,.0f}", columns=columns, head=head, tail=tail, only_print_formatted_columns = only_print_formatted_columns, reset_index=reset_index, print_zero_as=print_zero_as, print_nan_as=print_nan_as, to_display = to_display)


def printFormattedTable(table, format = None, columns = None, format2 = None, columns2 = None, format3 = None, columns3 = None, format4 = None, columns4 = None, format5 = None, columns5 = None, format6 = None, columns6 = None, format7 = None, columns7 = None, format8 = None, columns8 = None, head = 0, tail = 0, only_print_formatted_columns = True, reset_index=None, print_zero_as=None, print_nan_as=None, to_display=True):


    def format_cell(x, format_string):
        if (print_zero_as is not None) and ((type(x) is np.float64) or (type(x) is int) or (type(x) is float)):
            if isZero(x):
                return print_zero_as
        
        if (print_nan_as is not None) and ((type(x) is np.float64) or (type(x) is int) or (type(x) is float)):
            if np.isnan(x):
                return print_nan_as

        return format_string.format(x)

    t = pd.DataFrame(table).copy()

    if (format is not None) and (columns is None):
        columns = list(t.columns.values)
        
    if only_print_formatted_columns: columns_to_print = []

    if columns is not None:    
        for c in columns:
            t[c] = t[c].map(lambda x: format_cell(x, format))
        if only_print_formatted_columns: columns_to_print.extend(columns)
        
    if columns2 != None:
        for c in columns2:
            #t[c] = t.apply(lambda x: format2.format(x[c]), axis=1)
            t[c] = t[c].map(lambda x: format_cell(x, format2))
        if only_print_formatted_columns: columns_to_print.extend(columns2)
        
    if columns3 != None:
        for c in columns3:
            #t[c] = t.apply(lambda x: format3.format(x[c]), axis=1)
            t[c] = t[c].map(lambda x: format_cell(x, format3))
        if only_print_formatted_columns: columns_to_print.extend(columns3)

    if columns4 != None:
        for c in columns4:
            #t[c] = t.apply(lambda x: format4.format(x[c]), axis=1)
            t[c] = t[c].map(lambda x: format_cell(x, format4))
        if only_print_formatted_columns: columns_to_print.extend(columns4)

    if columns5 != None:
        for c in columns5:
            #t[c] = t.apply(lambda x: format5.format(x[c]), axis=1)
            t[c] = t[c].map(lambda x: format_cell(x, format5))
        if only_print_formatted_columns: columns_to_print.extend(columns5)


    if columns6 != None:
        for c in columns6:
            t[c] = t[c].map(lambda x: format_cell(x, format6))
        if only_print_formatted_columns: columns_to_print.extend(columns6)


    if columns7 != None:
        for c in columns7:
            t[c] = t[c].map(lambda x: format_cell(x, format7))
        if only_print_formatted_columns: columns_to_print.extend(columns7)

    if columns8 != None:
        for c in columns8:
            t[c] = t[c].map(lambda x: format_cell(x, format8))
        if only_print_formatted_columns: columns_to_print.extend(columns8)



    if reset_index is not None:
        if reset_index == '':
            t = t.reset_index()
            t.index +=1
        else:
            t.set_index([reset_index], inplace=True)

    if only_print_formatted_columns:
        t = t[columns_to_print]



    if (head+tail)>0:
        t = pd.concat([t.head(head), t.tail(tail)], axis = 0)
    
    if to_display:
        display(t)
    else:
        return t


def non_zero_dataFrame_columns_or_rows(df, column_or_row):
    if (column_or_row == 'column'):
        x = ((df>0.0001)|(df<-0.0001)).any(axis = 0)
        return df.columns[x]
    else:
        x = ((df>0.0001)|(df<-0.0001)).any(axis = 1)
        return df.index[x]
    

def excel_tables_to_relational_tables(tables, table_names, index_name, column_name, new_index_name):
        index = []
        for p in tables[0].index:
            for t in tables[0].columns:
                index.append((p, t))
        output = pd.DataFrame(index = index, columns=[index_name, column_name])
                   
        for (p,t) in output.index:
            for k, v in enumerate(tables):
                output.loc[[(p,t)],table_names[k]] = v.loc[p,t]
            output.loc[[(p,t)],index_name] = p
            output.loc[[(p,t)],column_name] = t
                
        output.reset_index(drop = True, inplace = True)
        output.set_index(new_index_name, inplace = True)
        return output

def date_to_string(date):
    return pd.Timestamp(date).strftime('%Y-%m-%d') 

def string_to_date(s):
    if isinstance(s, pd._libs.tslibs.timestamps.Timestamp):
        return s
    else:
        return pd.to_datetime(s)

def isZero(r):
    if (r<0.000001) and (r>-0.000001):
        return True
    else:
        return False

def isNotZero(r):
    if (r<0.000001) and (r>-0.000001):
        return False
    else:
        return True




def growth_resampler (df):
    return np.prod(df + 1)-1


if __name__ == '__main__':
    a = pd.DataFrame([0.5, 0.2, 0.1])
    print(growth_resampler(a))
    
    b = pd.DataFrame([[0.04, 0.02, False], [0.03,False,True], [True, 0.001, 0.0002]])
    print(b)
    
    def temp(l):
        d = True
        r =""
        for element in l:
            if element == False:
                d = False
                break
            
            r = r + " ha"
        if d: return r
        else: return d
    
    b['r'] = b.apply(temp, axis=1)
    print(b)