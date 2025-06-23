from collections import defaultdict, OrderedDict
from csv import reader
from itertools import chain, combinations
from optparse import OptionParser
from utils import *

def fpgrowthFromFile(fname, minSupRatio, minConf):
    itemSetList, frequency = getFromFile(fname)
    minSup = len(itemSetList) * minSupRatio
    #I fix it to 2 ,joseph
    minSup=2
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
    if(fpTree == None):
        print('No frequent item set')
    else:
        freqItems = []
        mineTree(headerTable, minSup, set(), freqItems)
        rules = associationRule(freqItems, itemSetList, minConf)
        return freqItems, rules

inputFile='dataset/basket.csv'

freqItemSet, rules = fpgrowthFromFile(inputFile,minSupRatio=0.23,minConf=0.5)
print('*************')
print(freqItemSet)
print('------------')
#print(rules)

