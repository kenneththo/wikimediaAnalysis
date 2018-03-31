import networkx as nx
import pylab as plt
import collections


class eigenVectorCentralityTest(object):
    def __init__(self):
        self.graph = nx.Graph()



    def createEdges(self, dict_edge):
        edgelist = []
        for node in dict_edge.keys():
            for child in dict_edge[node]:
                edgelist.append((node, child))
        return edgelist

    def createGraph(self, EdgeList):
        for tuple in EdgeList:
            self.graph.add_edge(tuple[0], tuple[1])
        #print("# of nodes=", len(self.graph.nodes()))
        #print("# of edges=", len(self.graph.edges()))

    def ComputeTopKEigenVectorCentrality(self, graph, topK=10):
        eig_dict = nx.eigenvector_centrality(graph)
        res = collections.Counter(eig_dict).most_common(topK)
        print("Top ", topK , "most central articles:")
        for k, v in res:
            print(k,v)
        return res


    def DepthLimitedDFS(self, node, depth, nodeSet):
        if depth == 0:
            return nodeSet
        if depth > 0:
            for child in self.graph[node]:
                nodeSet.update([child])
                nodeSet = self.DepthLimitedDFS(child, depth - 1, nodeSet)
        return nodeSet


    def getSubGraph(self, nodeName, depth):
        nodes = set()
        nodeList = list(self.DepthLimitedDFS(nodeName, depth, nodes) )
        subgraph = nx.subgraph(self.graph, nodeList)
        return subgraph


    def plot(self):
        nx.draw(self.graph)
        plt.show()
