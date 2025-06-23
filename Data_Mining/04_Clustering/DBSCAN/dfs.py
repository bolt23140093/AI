# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:20:40 2022
@author: joseph@艾思程式教育


"""

graph={'A':['B'],
      'B':['A'],
      'C':['E'],
      'D':['E','F'],
      'E':['C','D','F'],
      'F':['D','E']
     }
#visited={'A':0,'B':0,'C':0,'D':0,'E':0,'F':0}

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


connected_components(graph)
        