# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import re
import  datetime
__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r')

                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                # print(self.names)
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot_prun(self, names=None):
        plt.figure()
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name])) * 10
            plt.plot(x, np.asarray(numbers[name]))

        plt.legend([self.title  + name  for name in names], fontsize=15)
        plt.grid(True)
        plt.ylabel('Accuracy (%)', fontsize=20)
        plt.xlabel('Pruning rate (%)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()

    def plot(self, names=None):
        plt.figure()
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
                    
if __name__ == '__main__':

    # file = open('/data-x/g12/zhangjie/3dIP/paper_fig/prun/B-S.txt',"r")
    # file2 = open('/data-x/g12/zhangjie/3dIP/paper_fig/prun/O-S.txt',"r")

    file = open('/data-x/g12/zhangjie/DeepIPR/paper_fig/prun/B-R-10.txt',"r")
    file2 = open('/data-x/g12/zhangjie/DeepIPR/paper_fig/prun/O-R-10.txt',"r")

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    name = file.readline()
    names = name.rstrip().split('\t')  #标题分开
    numbers = {}
    for _, name in enumerate(names):
        numbers[name] = []

    name2 = file2.readline()
    names2 = name2.rstrip().split('\t')  #标题分开
    numbers2 = {}
    for _, name2 in enumerate(names2):
        numbers2[name2] = []

    for line in file:
        numbers_clm = line.rstrip().split('\t')

        for clm in range(0, len(numbers_clm)):
            numbers[names[clm]].append(numbers_clm[clm])

    for line2 in file2:
        numbers_clm2 = line2.rstrip().split('\t')

        for clm2 in range(0, len(numbers_clm2)):
            numbers2[names2[clm2]].append(numbers_clm2[clm2])

#########plot####################

    plt.figure()
    names_legend = []
    names_legend2 = []
    names_legend_all = []
    colors = ['dodgerblue',  'g', 'darkorange', 'r']

    for i, name in enumerate(names):
        if i in [1,3]:
            # x = np.arange(len(numbers[name]))
            x = np.arange(len(numbers[name])) * 10
            num_float = []
            for num in numbers[name]:
                num_float.append(float(num))

            print(i)
            print(name)
            # name = name.replace("Valid", "Test")
            # name = name.replace("Model for Releasing", "Baseline Deployment")
            # name = name.replace("Model for Verification", "Baseline Verification")
            # name = name.replace("Trigger", "Baseline Trigger")
            # name = name.replace("Signature", "Baseline Signature")
            #
            name = name.replace("Valid", "Test")
            name = name.replace("Model for Releasing", "Deployment -bs")
            name = name.replace("Model for Verification", "Verification -bs")
            name = name.replace("Trigger", "Trigger -bs")
            name = name.replace("Signature", "Signature -bs")
            print(name)
            names_legend.append(name)
            names_legend_all.append(name)
            plt.plot(x, num_float, linestyle="--", color = colors[i] )

    for i, name2 in enumerate(names2):
        if i in [1, 3]:

            # x = np.arange(len(numbers2[name2]))
            x = np.arange(len(numbers2[name2])) * 10
            num_float2 = []
            for num2 in numbers2[name2]:
                num_float2.append(float(num2))

            print(i)
            print(name2)
            # name2 = name2.replace("Valid", "Test")
            # name2 = name2.replace("Model for Releasing", "Ours Deployment")
            # name2 = name2.replace("Model for Verification", "Ours Verification")
            # name2 = name2.replace("Trigger", "Ours Trigger")
            # name2 = name2.replace("Signature", "Ours Signature")

            name2 = name2.replace("Valid", "Test")
            name2 = name2.replace("Model for Releasing", "Deployment -our")
            name2 = name2.replace("Model for Verification", "Verification -our")
            name2 = name2.replace("Trigger", "Trigger -our")
            name2 = name2.replace("Signature", "Signature -our")
            print(name2)
            names_legend2.append(name2)
            names_legend_all.append(name2)
            plt.plot(x, num_float2, color = colors[i])
    # plt.legend([  name  for name in names_legend], fontsize=15, loc='center right')
    plt.legend([  name2  for name2 in names_legend_all],fontsize=15, loc='lower left' )
    plt.grid(True)
    plt.ylabel('Accuracy (%)',fontsize=20)
    # plt.xlabel('Epochs',fontsize=20)
    plt.xlabel('Pruning rate (%)' ,fontsize=20)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    # dst_pth = "/data-x/g12/zhangjie/3dIP/paper_fig/prun/"
    dst_pth = "/data-x/g12/zhangjie/DeepIPR/paper_fig/prun/"
    savefig(dst_pth+'R-10-'+ timestr +'.eps')