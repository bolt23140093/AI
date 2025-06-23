# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 10:37:33 2022

@author: joseph@艾思程式教育

# this sample demostrates the usage of the python's 
frozenset and defaultdict.

"""
from collections import defaultdict
from itertools import chain, combinations
itemSetsList=[{'A','C','D'},{'B','C','E'},
              {'A','B','C','E'},{'B','E'}]

A={'A','B','C','D','E'}

itemSet=set(frozenset(a) for a in A)
#why do we use  frozenset ?
#because we want to use 'set' as a index to dict.
#but only immutable datatype can be the index to dict
#so ,we use frozenset instead of dict.
# {{'A'}:XXX ,{'B','C'}:XXX ,{'A','C'}:XXX,{'C'',B'.'D'}:xxx}
print('itemSet:',itemSet)  
min_suppot=0.5

def freq_item(itemSet,itemSetsList):
    
    freqItemSet = set()
    count=defaultdict(int)
    for item in itemSet:
        for itemset in itemSetsList:
            if item.issubset(itemset):
                count[item]+=1
    
    for item, value in count.items():
        #print(item,value,value/len(itemSetsList))
        if (value/len(itemSetsList)) >= min_suppot:
            freqItemSet.add(item)
    
    return freqItemSet
    
def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def self_join_set(A,length):
 
    S=set()      
    for i in A :
        for j in A:
            #print(i,j,'-->',i.union(j))
            if len(i.union(j)) == length:
                S.add(i.union(j))
    return S

def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            # if the subset is not in previous K-frequent get, then remove the set
            if(frozenset(subset) not in prevFreqSet):
                tempCandidateSet.remove(item)
                break
    return tempCandidateSet

def prune(LargeItemSet,CandateSet,size):
    
    new_CandateSet=CandateSet.copy()
    
    for item in CandateSet:

        subsets = combinations(item,size)
        for subset in subsets:
        
            # if the subset is not in previous K-frequent get, then remove the set
            if(frozenset(subset) not in LargeItemSet):
                new_CandateSet.remove(item)
                print('prune:',item)
                break
                    
    return new_CandateSet

L1=freq_item(itemSet,itemSetsList)   
print(L1)  

length=2
C2=self_join_set(L1,length)
print(C2)  


print('---------')


# length=2
# C2=prune(L1,C2,length-1)        
# L2=freq_item(C2,itemSetsList)
# print(L2) 

# length=3
# C3=self_join_set(L2,length)
# C3=prune(L2,C3,length-1)  
# L3=freq_item(C3,itemSetsList)
# print(L3)    

# length=4
# C3=self_join_set(L3,length)
# C3=prune(L2,C3,length-1)  
# L4=freq_item(C3,itemSetsList)
# print(L4)    

powerset(itemSet)