# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:23:54 2021
@author: joseph

 Do combination in recursive way (DFS)
 
 input: ['A','B','C']
 output:
    ['A']
    ['A', 'B']
    ['A', 'B', 'C']
    ['A', 'C']
    ['B']
    ['B', 'C']
    ['C']
     
"""

def combination(item,itemset):
    
    print(item)
    for i in itemset:
        if i > item[-1]:
            newitem=item+[i]
            combination(newitem,itemset)
      
            
itemset=['A','B','C','D']  
#itemset=['B','C','A']  
for i in itemset:
    combination([i],itemset)


