# -*- coding: utf-8 -*-
"""
Version: PB-008
Created on Tue Mar  3 12:56:56 2020
@author: Simba_Lin
@Class: ReDoPlot.show(self, x1, y1)
@Function: ValueFlagPlot(x,xName_str,y1,y1Name_str,y2,y2Name_str)
@Function: ValueFlagXRangePlot(x,xName_str,y1,y1Name_str,y2,y2Name_str,x_left,x_right,y1_limi,y2_limi)
@Function: FindOutput(inp_list, out_list, inp_target)
@Function: FixCsvErro('Csv Direction')
@Function: Get720CsvList()
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from dateutil.parser import parse

############################################################
#'''detect(Target Data,Whether "00XX" is sure(00 is sure))'''
def dect_str(inpu,dect_4str):
    dect_bo = (((str(inpu).zfill(4)[0:4])[0] == dect_4str[0]) or (dect_4str[0] =='X')) and \
           (((str(inpu).zfill(4)[0:4])[1] == dect_4str[1]) or (dect_4str[1] =='X')) and \
           (((str(inpu).zfill(4)[0:4])[2] == dect_4str[2]) or (dect_4str[2] =='X')) and \
           (((str(inpu).zfill(4)[0:4])[3] == dect_4str[3]) or (dect_4str[3] =='X'))
    return (dect_bo)
############################################################
#'''plot setting'''
def ValueFlagPlot(x1,xName_str,y1,y1Name_str,x2,y2,y2Name_str): 
    plt.figure(figsize=(8,4))
    plt.plot(x1,y1,color='royalblue',label=y1Name_str,linewidth=3.0)
    plt.xlabel(xName_str,size=20,)
    plt.ylabel(y1Name_str,size=20)
    plt.minorticks_on()
    plt.tick_params(which='minor',direction='in')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower right',fontsize=14)
    plt.twinx()
    plt.plot(x2,y2,color='firebrick',marker='.',label=y2Name_str,linewidth=0)
    plt.ylabel(y2Name_str,size=20)
    plt.title(y1Name_str+'&'+y2Name_str,size=26,family='Times New Roman')
    plt.minorticks_on()
    plt.tick_params(which='minor',direction='in')
    plt.xticks(np.linspace(min(x2), max(x2), 6))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='best',fontsize=14)
    plt.show()

def ValueFlagXRangePlot(x1,xName_str,y1,y1Name_str,x2,y2,y2Name_str,x_limi,y1_limi,y2_limi): 
    plt.figure(figsize=(6,4))
    plt.plot(x1,y1,color='royalblue',label=y1Name_str,linewidth=3.0)
    plt.xlabel(xName_str,size=20,)
    plt.ylabel(y1Name_str,size=20)
    plt.minorticks_on()
    plt.tick_params(which='minor',direction='in')
    plt.xticks(fontsize=16)
    plt.yticks(np.linspace(y1_limi[0],y1_limi[1], 6))
    plt.yticks(fontsize=16)
    plt.ylim(y1_limi)
    plt.legend(loc='lower right',fontsize=14)
    plt.twinx()
    plt.plot(x2,y2,color='firebrick',marker='.',label=y2Name_str,linewidth=0)
    plt.ylabel(y2Name_str,size=20)
    plt.title(y1Name_str+'&'+y2Name_str,size=26,family='Times New Roman')
    plt.minorticks_on()
    plt.tick_params(which='minor',direction='in')
    plt.xticks(np.linspace(x_limi[0], x_limi[1], 6))
    plt.xticks(fontsize=16)
    plt.yticks(np.linspace(y2_limi[0],y2_limi[1], 6))
    plt.yticks(fontsize=16)
    plt.ylim(y2_limi)
    plt.xlim(x_limi[0], x_limi[1])
    plt.legend(loc='best',fontsize=14)
    plt.show()

############################################################
#'''Fix Csv Format erro (replace',,' to ',')'''
def FixCsvErro(csv_dir):
    file = open(csv_dir,'r')
    file_list = file.readlines()
    for i in range(0,len(file_list)):
        file_list[i] = file_list[i].replace(',,',',')
    file.close()
    with open(csv_dir,'w')as file_w:
        file_w.writelines(file_list)
############################################################
#Value filter
def ValueFilter_shoc(input_csv_name,output_name,value_colum,suspend_fact):
    
    FixCsvErro(input_csv_name)
    input_df = pd.read_csv(input_csv_name,index_col = False)
    print('len(input_df): ',len(input_df))
    
    loop_bo = True
    i = 0
    drop_counter = 0
    while(loop_bo):
        if (i == 0):
            if (abs(float(input_df[value_colum][0]) - float(input_df[value_colum][1])) \
            > suspend_fact):
                print('***********First Row or Second Row Erro*********************')
                print('$$$$$$$$$$$First row error will cause All Error$$$$$$$$$$$$$$')
            else:
                print('First Row and Second Row OK') 
        if (abs(float(input_df[value_colum][i]) - float(input_df[value_colum][i+1])) \
            > suspend_fact):
            print(value_colum,': ',input_df[value_colum][i+1],'; index: ',i+drop_counter)
            input_df = input_df.drop([i+1])
            drop_counter = drop_counter + 1
            i = i + 1
        i = i + 1
        if (i == len(input_df)-1):
            loop_bo = False
    input_df.to_csv(output_name,index=0)
    
############################################################
#Flag filter
#Drop data by different with last one and next one
def FlagFilter_diff(input_csv_name,output_name,flag_colum):
    
    FixCsvErro(input_csv_name)
    input_df = pd.read_csv(input_csv_name,index_col = False)
    print('len(input_df): ',len(input_df))
    
    loop_bo = True
    i = 0
    drop_counter = 0
    while(loop_bo):
        if (i == 0):
            i = i + 1
            continue
        if ((input_df[flag_colum][i-1] != input_df[flag_colum][i]) and \
            (input_df[flag_colum][i] != input_df[flag_colum][i+1])):
            print(flag_colum,': ',input_df[flag_colum][i],'; index: ',i+drop_counter)
            input_df = input_df.drop([i])
            drop_counter = drop_counter + 1
            i = i + 1
        i = i + 1
        if (i == len(input_df)-1):
            loop_bo = False
    input_df.to_csv(output_name,index=0)
    

############################################################
#Get 720L data direction list
def Get720CsvList():
    FileName_list = os.listdir()
    csv_720L_name_list = []
    for n in range(0,len(FileName_list)):#read each file except (.py)&(-ACC)
        if((FileName_list[n][0]>='0') & (FileName_list[n][0]<='9')):
            csv_720L_name_list.append(FileName_list[n])
    else:
        print(FileName_list[n],':first name is not number, would not load')
    return(csv_720L_name_list)
    
############################################################
#'''Plot Function'''
def ReDoPlot(x1, y1, in_y1Name_str = "y1",\
               in_xName_str = "x",\
               in_linewidth = 3.0,\
               in_fontsize = 16,\
               in_figsize = (6,4),\
               in_color = 'royalblue'):        
    plt.figure(figsize=in_figsize)
    plt.plot(x1,y1,color=in_color,label=in_y1Name_str,linewidth=in_linewidth)
    plt.xlabel(in_xName_str,size=in_fontsize,)
    plt.ylabel(in_y1Name_str,size=in_fontsize)
    plt.minorticks_on()
    plt.xticks(fontsize=in_fontsize)
    plt.yticks(fontsize=in_fontsize)
    plt.legend(loc='lower right',fontsize=in_fontsize)
    plt.title(in_y1Name_str,size=in_fontsize,family='Times New Roman')
    plt.minorticks_on()
    plt.tick_params(which='minor',direction='in')
    plt.xticks(fontsize=in_fontsize)
    plt.yticks(fontsize=in_fontsize)
    plt.legend(loc='best',fontsize=in_fontsize)
    plt.show()
    
############################################################
#'''Return Nearest Output'''
def FindOutput(inp_list, out_list, inp_target):
    i = 0
    while(i < len(inp_list)):
        if((inp_target <= inp_list[i+1]) == (inp_target >= inp_list[i])):
            return(out_list[i])
        i = i + 1
    return()

############################################################
#'''Return Time List'''
def DateToSecond(inp_list):
    out_list = []
    for i in range(len(inp_list)):
        if(i==0):
            time0 = parse(inp_list[i])
            out_list.append(0)
        else:
            time = parse(inp_list[i]) - time0
            out_list.append(time.total_seconds())
    return(out_list)
    

############################################################
#'''example main'''
'''
RangeFilColu_str = 'Relative State Of Charge'
RangeUpper_num   = 101
RangeLower_num   =  -1

FlagFilColu_str  = 'TempModeControl'
FlagSure_str   = '00XX'    # "00XX" 00 is sure

ResultData_tuple = DataHandling(RangeFilColu_str, RangeUpper_num, RangeLower_num, \
                 FlagFilColu_str, FlagSure_str, 'filter')
Time_list = ResultData_tuple[0]
RangeFilColu_list = ResultData_tuple[1]
Flag1FilColu_list = ResultData_tuple[2]
'''
'''Plot''''''
ValueFlagPlot(np.array(Time_list)/60,'time[60s]', RangeFilColu_list,'RSOC',np.array(Time_list)/60, Flag1FilColu_list,'flag' )
ValueFlagXRangePlot(np.array(Time_list),'time[1s]', RangeFilColu_list,'RSOC', np.array(Time_list)/60, Flag1FilColu_list,'flag',\
                    [12580, 12700], [48,53],[-0.02,0.02])
'''
