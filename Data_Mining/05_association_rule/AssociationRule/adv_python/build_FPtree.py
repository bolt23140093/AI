# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:27:40 2022

@author: joseph@艾思程式教育

Create a Tree and a header table.
The table will use  linked list to link the 

"""

from collections import defaultdict

class Node:

    def __init__(self, itemName, frequency):
        self.itemName = itemName
        self.count = frequency
        #self.parent = parentNode
        self.children = {}  # {key:value=> 'itemName':Node}
        self.next = None  #node link

    def increment(self, frequency):
        self.count += frequency

    def display(self, ind=0): # do DFS(Depth First Search)
        print('   ' * ind, '{}:{}'.format(self.itemName,self.count))
        for child in list(self.children.values()):
            child.display(ind+1)
            
            
itemSetList=[['A', 'B'],
             ['B', 'D'],
             ['B', 'C'],
             ['A', 'B', 'D'],
             ['A', 'C'],
             ['B', 'C'],
             ['A', 'C'],
             ['A', 'B', 'C', 'E'],
             ['A', 'B', 'C']]  


headerTable = defaultdict(int)
# Counting frequency and create header table
for idx, itemSet in enumerate(itemSetList):
    for item in itemSet:
        headerTable[item] += 1 
        
print(headerTable)

# HeaderTable column [Item: [frequency, headNode]]
for item in headerTable:
    headerTable[item] = [headerTable[item], None]

print(headerTable)

def updateHeaderTable(item, targetNode, headerTable):
    if(headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        # Traverse to the last node then link it to the target
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode

def updateTree(item, treeNode, headerTable):
    if item in treeNode.children:
        # If the item already exists, increment the count
        treeNode.children[item].increment(1)
    else:
        # Create a new branch
        newItemNode = Node(item, 1)
        treeNode.children[item] = newItemNode
        # Link the new branch to header table
        updateHeaderTable(item, newItemNode, headerTable)

    return treeNode.children[item]


    
# Init Null head node
fpTree = Node('Null', 0)  #root node
# Update FP tree for each cleaned and sorted itemSet
for idx, itemSet in enumerate(itemSetList):
    itemSet = [item for item in itemSet if item in headerTable]
    itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
    # Traverse from root to leaf, update tree with given item
    currentNode = fpTree
    for item in itemSet:
        currentNode = updateTree(item, currentNode, headerTable)

print(fpTree.display())
#print(headerTable)

node = headerTable['C'][1] 
while node != None:
    print("{}:{}-->".format(node.itemName,node.count),end='  ')
    node=node.next
print()