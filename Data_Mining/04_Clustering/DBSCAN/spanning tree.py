# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:00:58 2023

@author: joseph@艾鍗學院

We'll use a sample graph represented as an adjacency matrix 
and implement Kruskal's algorithm to find the MST.

0 ---10---- 1
| \        |
|   \      |
6     5    15
|       \ |
2 ---4----3



0 ---10---- 1
 \        
   \      
     5   
       \  
2 --4---3



"""

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def add_edge(self, u, v, w):
        self.edges.append((u, v, w))

def kruskal_mst(graph):
    graph.edges = sorted(graph.edges, key=lambda edge: edge[2])
    parent = [i for i in range(graph.V)]
    mst = []
    edges_added = 0

    def find_set(vertex):
        if parent[vertex] == vertex:
            return vertex
        parent[vertex] = find_set(parent[vertex])
        return parent[vertex]

    while edges_added < graph.V - 1:
        u, v, w = graph.edges.pop(0)
        u_set = find_set(u)
        v_set = find_set(v)

        if u_set != v_set:
            mst.append((u, v, w))
            parent[u_set] = v_set
            edges_added += 1

    return mst

# Example usage:
g = Graph(4)
g.add_edge(0, 1, 10)
g.add_edge(0, 2, 6)
g.add_edge(0, 3, 5)
g.add_edge(1, 3, 15)
g.add_edge(2, 3, 4)

mst = kruskal_mst(g)
print("Minimum Spanning Tree:")
for u, v, w in mst:
    print(f"Edge: {u} - {v}, Weight: {w}")
