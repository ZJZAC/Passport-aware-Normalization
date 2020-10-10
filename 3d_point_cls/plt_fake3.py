# coding:utf-8

"""
Author: roguesir
Date: 2017/8/30
GitHub: https://roguesir.github.com
Blog: http://blog.csdn.net/roguesir
"""

import numpy as np
import matplotlib.pyplot as plt

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)

x = np.arange(0, 11) * 10
plt.hlines(63.95, 0, 100, colors = "r", linestyles = "dashed", label='valid signature (ours)')
plt.hlines(59.80, 0, 100,  linestyles = "dashed", label='valid signature (baseline)')

y1 = [5.88,	5.44,	5.04,	4.77,	5.16,	5.43,	4.79,	5.40,	4.92,	5.08,	4.75]

y2 = [59.78, 54.07, 	48.06,	42.36,	8.67,	8.45,	6.74,	8.43,	8.87,	9.56,	8.70 ]

plt.plot(x, y1 , label = 'fake signature (ours)')
plt.plot(x, y2 , label = 'fake signature (baseline)')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Dissimilarity between valid and fake signature (%)',  fontsize =15)
plt.ylabel('Accuracy (%)',  fontsize =15)

plt.legend( loc='center right',  fontsize =13)
plt.grid(True)
plt.tight_layout()

savefig('try.eps')
