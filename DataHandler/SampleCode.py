# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:04:56 2021

@author: huanc
"""

import pandas as pd
import DataHandler as DH

dsg_csv = 'SampleData.csv'
Processed_dsg_csv = 'SampleData2.csv'

DH.FixCsvErro(dsg_csv)
DH.ValueFilter_shoc(dsg_csv,Processed_dsg_csv,'LV',200)
DH.FlagFilter_diff(Processed_dsg_csv,Processed_dsg_csv,'Status')

discg_df  = pd.read_csv(Processed_dsg_csv,index_col=False)
Time_list = DH.DateToSecond(discg_df["Rdate"])
DH.ReDoPlot(Time_list,discg_df["LV"])