import sys


class Node:
    def __init__(self, index):
        self.index = index
        self.neighbors = []
        self.dist = sys.maxsize
        self.shortest_route = []


class ShortestPathSolver:
    def __init__(self, distances):
        self.distances = distances
        self.nodes = {}
        self.unvisited_nodes = []
        for i in range(0, len(distances)):
            self.nodes[i] = Node(i)
            self.unvisited_nodes.append(self.nodes[i])

        self.unvisited_nodes.remove(self.nodes[0])

        for i in range(0, len(self.distances)):
            n = self.nodes[i]
            for j in range(i, len(self.distances)):
                if distances[i, j] < sys.maxsize:
                    n.neighbors.append(self.nodes[j])
                    if i == 0:
                        self.nodes[j].dist = distances[i, j]
                        self.nodes[j].shortest_route.append((i, j))

    def solve(self):
        while self.unvisited_nodes:
            self.unvisited_nodes.sort(key=lambda x: x.dist)
            nearest_node = self.unvisited_nodes[0]
            self.unvisited_nodes.remove(nearest_node)

            for n in nearest_node.neighbors:
                if n != nearest_node:
                    alt = nearest_node.dist + self.distances[nearest_node.index, n.index]
                else:
                    alt = nearest_node.dist
                if alt < n.dist:
                    n.dist = alt
                    n.shortest_route = nearest_node.shortest_route + [(nearest_node.index, n.index)]

        return self.nodes[len(self.distances) - 1].shortest_route
