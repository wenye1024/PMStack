'''
Created on Feb 17, 2019

@author: yewen
'''

import pandas as pd



class ExcelWriter:

    def __init__(self, output_dir = ''):
        self.output_dir = output_dir
        self.files = {}


    def print(self, filename, dfs, sheetnames):
        if (isinstance(dfs, list) & isinstance(sheetnames, list)) == False:
            raise TypeError('ExcelWriter.print() takes list parameter instead of scalar')

        if len(dfs) != len(sheetnames) :
            raise ValueError('The number of DataFrames is not equal to the number of sheet names')

        
        if (filename in self.files.keys()) == False:
            writer = pd.ExcelWriter(self.output_dir + filename + '.xlsx')
            self.files[filename] = writer
        else:
            writer = self.files[filename]
        
        for idx, df in enumerate(dfs):
            df.to_excel(writer,sheetnames[idx])
        
    
    def save(self):
        for key, writer in self.files.items():
            writer.save()
        self.files = {}
            
    
if __name__ == '__main__':
    pass
