'''
The main functions are
    extract_tables
    export_to_excel
'''

import pandas as pd
import numpy as np
import math
import urllib.request  as urllib2 
import openpyxl as pl
from openpyxl.styles import colors
import copy
from bs4 import BeautifulSoup
import Tools.data_cleanup_helpers as data_cleanup_helpers



def match_period(string):
    if str in ['Q3 FY2019', 'Q3 FY2018', 'Q2 FY2019']:
        return string
    else:
        return None

def preprocess_text(string):
    string = string.replace('\xa0', ' ')
    string = string.replace('\n', ' ')
    string = string.replace('•','')
    #string = string.strip()
    string = ' '.join(string.split())
    if string == '': string = np.nan
    return string

    
def process_row(row):
    new_row = []
    l = len(row)
    i = 0
    
    while i<l:
        t = row.iloc[i]
        
        #print('----------', t)
        
        if type(t) is str:
            
            if (t[-3:] in ['(1)','(2)','(3)','(4)','(5)','(6)','(7)','(8)','(9)']) and (len(t)>3):
                t = t[0:-3]
            
            
            if t == '%':
                a = new_row.pop()
                if type(a) is str:
                    new_row.append(a)
                    new_row.append('%')
                else:
                    new_row.append(a/100)
                    new_row.append(float('nan'))
            elif t in [')', ')%', '%)']:
                num_of_pops = 0
                tt = t
                while True:
                    a  = new_row.pop() # here, usually a is a string of '(' + number
                    num_of_pops += 1
                    tt = data_cleanup_helpers.float_to_string(a) + tt
                    to_break = False
                    if type(a) is str:
                        if a[0] == '(':
                            to_break = True
                    
                    if to_break: break
                new_row.append(data_cleanup_helpers.convert_number_if_applicable(tt))
                for j in range(0, num_of_pops):
                    new_row.append(float('nan'))
            else:
                new_row.append(data_cleanup_helpers.convert_number_if_applicable(t))
        elif type(t) is float:
            #if not(math.isnan(t)):
            new_row.append(t)

        i += 1
        
    
    # deal with $ dollar sign
    is_header_row = False #need to implement the logic
    if not(is_header_row):    
        ll = len(new_row)
        for i in range(0, ll):
            if (new_row[i] == '$') and (i+1 < ll):
                if (type(new_row[i+1]) is float) or (type(new_row[i+1]) is int):
                    #new_row[i] = new_row[i+1]
                    #new_row[i+1] = float('nan')
                    new_row[i] = float('nan')  # works for anaplan
            
                
    return new_row


def process_table(table):
    l = len(table.index)
    rows = []
    num_of_cols = 0
    for i in range(0, l):
        row = process_row(table.iloc[i,:])
        if len(row) > num_of_cols:
            num_of_cols = len(row)
        rows.append(row)
        
    #rowss.append(rows)
    return pd.DataFrame(rows)


def extract_tables(page):    
    soup = BeautifulSoup(page, "html.parser")
    tables = soup.find_all('table')
    dfs = []
    
    for table in tables:
        
        rows = table.find_all('tr')
        
        n_cols = 100 # big enough for tables that's possible. This can be a potential bug
            
        # Create list to store rowspan values 
        skip_index = [0 for i in range(0, n_cols)]
        
        
        current_row = 0
        
        df = pd.DataFrame(columns=range(0,n_cols), index = range(0,len(rows))) 
        
        for row in rows:
                        
            current_col = 0
            columns = row.find_all(['td','th'])
            
            col_dim = []
            row_dim = []
            col_dim_counter = -1
            row_dim_counter = -1
            current_col = -1
            this_skip_index = copy.deepcopy(skip_index) 
            
            
            for col in columns:
                # Determine cell dimensions
                colspan = col.get("colspan")
                if colspan is None:
                    col_dim.append(1)
                else:
                    col_dim.append(int(colspan))
                col_dim_counter += 1

                rowspan = col.get("rowspan")
                if rowspan is None:
                    row_dim.append(1)
                else:
                    row_dim.append(int(rowspan))
                row_dim_counter += 1

                # Adjust column counter
                if current_col == -1:
                    current_col = 0  
                else:
                    current_col = current_col + col_dim[col_dim_counter - 1]

                while skip_index[current_col] > 0:
                    current_col += 1


                t = col.get_text()
                df.iloc[current_row, current_col] = preprocess_text(t)

                # Record column skipping index
                if row_dim[row_dim_counter] > 1:
                    this_skip_index[current_col] = row_dim[row_dim_counter]

            current_row +=1

            # Adjust column skipping index for the next row
            skip_index = [i - 1 if i > 0 else i for i in this_skip_index]
            
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        if len(df.index)>1: 
            df = process_table(df).dropna(axis=1, how='all')
            dfs.append(df)
    
    return dfs

'''
Obsolete implementation that doesn't take into account colspan and rowspan
'''
def extract_tables_old(page):    
    soup = BeautifulSoup(page, "html.parser")
    tables = soup.find_all('table')
    dfs = []
    
    for table in tables:
    
        row_marker = 0
        rows = table.find_all('tr')
        df = pd.DataFrame(columns=range(0,50), index = range(0,len(rows)+1)) 
        for row in rows:
            column_marker = 0
            columns = row.find_all('td')
            for column in columns:
                #print(column.get_text())
                #print((row_marker, column_marker))
                t = column.get_text()
                df.iloc[row_marker, column_marker] = preprocess_text(t)
                
                column_marker += 1
            row_marker +=1
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, how='all')
        if len(df.index)>1: dfs.append(process_table(df))
    
    return dfs

def export_to_excel(tables, filename='output.xlsx'):
    writer = pd.ExcelWriter(filename)
    i = 0
    for df in tables:
        df.to_excel(writer,sheet_name = str(i), index=False, header = False)
        i += 1
    writer.save() 


if __name__ == '__main__':
    #addr = 'https://www.globenewswire.com/news-release/2019/08/27/1907424/0/en/Ooma-Reports-Second-Quarter-Fiscal-2020-Financial-Results.html'
    #page = urllib2.urlopen(addr)
    #local_page = open("3690_1.html", encoding='utf8')

    #tables = extract_tables(page)
    #tables = extract_tables(local_page)

    pass
