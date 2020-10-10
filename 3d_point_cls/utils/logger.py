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

    file = open('/data-x/g12/zhangjie/DeepIPR/paper_fig/M-all/PBN_SE.txt',"r")

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    name = file.readline()
    names = name.rstrip().split('\t')  #标题分开
    numbers = {}
    for _, name in enumerate(names):
        numbers[name] = []


    for line in file:
        numbers_clm = line.rstrip().split('\t')

        for clm in range(0, len(numbers_clm)):
            numbers[names[clm]].append(numbers_clm[clm])


#########plot####################

    plt.figure()
    names_legend = []
    names_legend2 = []
    for i, name in enumerate(names):
        if i not in [2, 5]:
        # if i not in [4]:
            x = np.arange(len(numbers[name]))
            num_float = []
            for num in numbers[name]:
                num_float.append(float(num)  * 100 )

            print(i)
            print(name)
            name = name.replace("Valid", "Test")
            name = name.replace("Model for Releasing", "Deployment")
            name = name.replace("Model for Verification", "Verification")
            name = name.replace(".", "")
            print(name)
            names_legend.append(name)
            plt.plot(x, num_float)



    plt.legend([  name  for name in names_legend], fontsize=20, loc='center right')
    plt.grid(True)
    plt.ylabel('Accuracy (%)',fontsize=20)
    # plt.xlabel('Pruning rate (%)')
    plt.xlabel('Epoch',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    dst_pth = "/data-x/g12/zhangjie/DeepIPR/paper_fig/M-all/"
    savefig(dst_pth+'PBN_SE'+ timestr +'.eps')