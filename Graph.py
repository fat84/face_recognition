from Node import Node
from collections import Counter

class Graph:

    def __init__(self):
        self.nodes = {}
        self.ID_to_labels = {}

    def addNode(self, node):
        self.nodes[node.ID] = node
        self.ID_to_labels[node.ID] = node.label

    def addEdge(self, edge_weight, node1_ID, node2_ID):
        self.nodes[node1_ID].neighbors.append((node2_ID, edge_weight))
        self.nodes[node2_ID].neighbors.append((node1_ID, edge_weight))

    def updateGraph(self):
        for ID in self.nodes:
            self.nodes[ID].update(self.ID_to_labels)

    def getLabelNum(self):
        counter = Counter()
        for node in self.nodes.values():
            counter[node.label] += 1
        return len(counter)