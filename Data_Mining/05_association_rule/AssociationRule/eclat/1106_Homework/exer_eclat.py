# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:39:51 2023

@author: joseph@艾鍗學院
"""




def to_vertical_format(dataset, min_support):
    """Convert dataset to vertical format """
    vertical = {}
    for tid, transaction in enumerate(dataset):
        for item in transaction:
            if item not in vertical:
                vertical[item] = set()
            vertical[item].add(tid+1)
            
    # Only consider items that meet the minimum support
    vertical = {item: tid for item, tid in vertical.items() if len(tid) >= min_support}
    return vertical

if __name__ == "__main__":
    
    # Example
    dataset = [['Bread', 'Milk'],
               ['Bread', 'Diaper', 'Beer', 'Eggs'],
               ['Milk', 'Diaper', 'Beer', 'Coke'],
               ['Bread', 'Milk', 'Diaper', 'Beer'],
               ['Bread', 'Milk', 'Diaper', 'Coke']]
   
    min_support=3
    itemsets=to_vertical_format(dataset,min_support)
    print(itemsets)
    

