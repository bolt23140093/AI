# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:20:40 2022
@author: joseph@艾思程式教育


"""
#use adjancent matrix to represent graph
graph2=[[1,1,0,0,0,0],
       [1,1,0,0,0,0],
       [0,0,1,0,1,0],
       [0,0,0,1,1,1],
       [0,0,1,1,1,1],
       [0,0,0,1,1,1]    
    ]


graph={'A':['B'],
      'B':['A'],
      'C':['E'],
      'D':['E','F'],
      'E':['C','D','F'],
      'F':['D','E']
     }


components = []  # To store connected components
visited = set()  # To keep track of visited vertices

def dfs(vertex, component):
    visited.add(vertex)
    component.add(vertex)
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            dfs(neighbor, component)


def connected_components(graph):

    for vertex in graph:
        if vertex not in visited:
            component = set()
            dfs(vertex, component)
            components.append(component)

    return components



print(connected_components(graph))
        