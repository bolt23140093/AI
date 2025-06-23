# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:11:10 2022

@author: joseph@艾思程式教育
"""
from collections import defaultdict
from itertools import chain
from itertools import chain, combinations


# print('-----defaultdict--------')
# ddict=defaultdict(int) #default value is 0
# print(ddict['A'])   # return defult value when Key doesn't exist
# ddict['B']+=1
# print(ddict)


# print('-----frozenset--------')

'''
Why do we use frozenset ?

Example 1:

count[{'B','C'}]+=1  # TypeError Error : "unhashable type: 'set'"
in this case, we can use frozenset to make 'set' as a key
count[frozenset({'B','C'})]=1  # OK

Example 2:
  set within set 
 {{'A'},{'B'},{'C'}} --> TypeError: unhashable type: 'set'    

 we can use frozenset to achieve 
 {frozenset({'A'}),frozenset({'B'}),frozenset({'C'})} 
'''

count=defaultdict(int)
count[frozenset({'D'})]=0
print(count)

print('-----frozenset as a index--------')

#itemsets={frozenset({'B','C'}),frozenset({'D'})}
itemsets=[{'A','B','C'},{'D','E'}]
itemsetsList=[{'A','B','C'},{'A','B'},
              {'A','B','C','D'},{'D','E'}]
for item in itemsets:
    for itemset in itemsetsList:
        #print('{} in {}:{}'.format(item,itemset,item.issubset(itemset)))
        if item.issubset(itemset):
            count[frozenset(item)]+=1 #use set as a key index
         
for k,v in count.items():
    print(k,":",v)


print('-----frozenset with a set--------')

itemsets=[{'A','B','C'},{'A','B','E'}]
newitemsets=[]
for itemset in itemsets:
    # t is a set of set 
    t=[set(item) for item in itemset ]
    newitemsets.append(t)

print('newitemsets:',newitemsets)
print('----------')
joinset=set()
for i in newitemsets[0]: # [{'A'},{'B'},{'C'}]
    for j in newitemsets[1]:# [{'D'},{'E'}]
        if len(i.union(j)) == 2: # do join set with length=2
            print(i.union(j))
            joinset.add(frozenset(i.union(j)))
print('joinset:',joinset)


# for itemset in itemsets:
#     for i in range(1,len(itemset)):
#         subsets = combinations(itemset,i)
#         print(list(subsets))


# print('-----combinations--------')
# itemset=['A','B','C']
# for i in range(1,len(itemset)+1):
#     subsets = combinations(itemset,i)
#     print(list(subsets))

print('-----chain--------')
# chain
# a list of odd numbers
odd = [1, 3, 5, 7, 9]
# a list of even numbers
even = [2, 4, 6, 8, 10]
# # chaining odd and even numbers
# numbers = list(chain(odd, even))
# print(numbers)

#chain.from_iterable only one arguments
str3 = list(chain.from_iterable([odd, even]))
print(str3)

