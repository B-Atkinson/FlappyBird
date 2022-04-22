#!/usr/bin/env python3

## Geoff: function to plot CDFs given an input dictionary containing multiple CDF raw data
##
import matplotlib.pyplot as plt
import pathlib
import os
import numpy as np
import pickle


def plot_cdf(data, x_ticks, x_text, legend_loc, x_label, y_label, title, log_x=False):
    for i in range(0, len(data)):
        if log_x == True:
            plt.semilogx(data[i]['X'], data[i]['Y'], label=data[i]['key'], basex=2)
        else:
            plt.plot(data[i]['X'], data[i]['Y'], label=data[i]['key'])

    plt.legend(loc=legend_loc, prop={'size':11})
    plt.xlabel(x_label)
    if not x_ticks == None:
        if log_x == True:
            plt.xticks(list(x_ticks), x_text)
        else:
            plt.xticks(list(x_ticks), rotation='vertical')

    plt.ylabel(y_label)
    if not title == None:
        plt.title(title)
        
    plt.show()

if 1:  # built-in testing code
    MAX=7
    x1 = [1, 2, 3, 4, 5, 6, 7]
    y1 = [0.2, 0.4, 0.8, 0.8, 0.9, 1.0, 1.0]
    
    x2 = [2, 3, 4, 5, 6, 8, 16]
    y2 = [0.6, 0.7, 0.8, 0.8, 1.0, 1.0, 1.0]

    cdf = [{}, {}]
    cdf[0]['key']='T1'
    cdf[0]['X']=x1
    cdf[0]['Y']=y1
    
    cdf[1]['key']='T2'
    cdf[1]['X']=x2
    cdf[1]['Y']=y2
    
    # another way of defining cdf
    #cdf = [{'key':'T1', 'X': x1, 'Y': y1}, {'key':'T2', 'X': x2, 'Y': y2}]
    
    legend_loc='lower right'
    xlabel = '$X$ = Message Type Code'
    #xticks = range(1, MAX + 1)
    xticks = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 6, 7]
    xtext = ('1/32', '1/16', '1/8', '1/4', '1/2', '1', '2', '4', '6', '7')
    ylabel = 'Prob[ $x\leq X$ ]'
    xlog = True
    title = "Testing"
    
    plot_cdf(cdf, xticks, xtext, legend_loc, xlabel, ylabel, title, xlog)


