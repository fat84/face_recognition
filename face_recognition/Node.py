from collections import Counter
class Node:
    def __init__(self, label, ID):
        self.label = label # This is the one that changes
        self.ID = ID # Remains constant
        self.neighbors = [] # Contains a tuple of (ID, weight)

    def addEdge(self, node_ID, weight):
        self.neighbors.append((node_ID, weight))


    def update(self, ID_to_label):
        
        counter = Counter()

        for node_ID, weight in self.neighbors:
            counter[ID_to_label[node_ID]] += weight
        
        self.label = counter.most_common(1)[0][0]
        ID_to_label[self.ID] = self.label
        