# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 17:20:46 2023

@author: joseph@艾鍗學院
"""


def vertical_format(dataset,min_support=2):
    
    vertical_data = {}
    for tid, transaction in enumerate(dataset):
        for item in transaction:
            if item not in vertical_data:
                vertical_data[item] = set()
            vertical_data[item].add(tid+1)
            
    # remove single items based on min_support
    vertical_data = {k: v for k, v in vertical_data.items() if len(v) >= min_support}

    return vertical_data

def eclat_recursive(itemset, tid_list, vertical_data, min_support, frequent_itemsets):
   
    frequent_itemsets[tuple(itemset)] = len(tid_list)
    
    for item, tid in vertical_data.items():
        if item > itemset[-1]:  # To ensure lexicographical order and avoid duplicates
            new_itemset = itemset + [item]
            new_tid_list = tid_list & tid
            
            if len(new_tid_list) >= min_support:
                eclat_recursive(new_itemset, new_tid_list, vertical_data, min_support, frequent_itemsets)


if __name__ == "__main__":
    
       # Example
    dataset = [['Bread', 'Milk'],
               ['Bread', 'Diaper', 'Beer', 'Eggs'],
               ['Milk', 'Diaper', 'Beer', 'Coke'],
               ['Bread', 'Milk', 'Diaper', 'Beer'],
               ['Bread', 'Milk', 'Diaper', 'Coke']]
    
    min_support=3
    vertical_data = vertical_format(dataset,min_support)
    frequent_itemsets = {}

    for item, tid_list in vertical_data.items():
        eclat_recursive([item], tid_list, vertical_data, min_support, frequent_itemsets)
        
    result=frequent_itemsets
    
    for items, count in result.items():
        print(', '.join(items), ":", count)
