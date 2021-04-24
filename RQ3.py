import os
import networkx as nx
from node2vec import Node2Vec
import evaluation
import karateclub
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from typing import List


edge_list_file = open(os.getcwd() + "\\data\\q3_edge_list.txt")
edge_list = edge_list_file.readlines()
node_list_file = open(os.getcwd() + "\\data\\q3_node_list.txt")
node_list = node_list_file.readlines()
graphs = []
for graph in edge_list:
    G = nx.DiGraph()
    for edge in graph:
        node_1 = edge.strip().split(" ")[0]
        node_2 = edge.strip().split(" ")[1]
        G.add_edge(node_1, node_2)
    graphs.append(G)
dic_ = {}
for i in range(len(graphs)):
    node2vec_ = Node2Vec(graphs[i], dimensions=64, walk_length=30, num_walks=200, workers=4)
    output_list = node2vec_.fit()
    keys_ = output_list.wv.index_to_key
    values_ = output_list.wv.vectors
    for j in range(len(keys_)):
        dic_[keys_[j] + "_" + str(i)] = [values_[i]]
print("Node2vec Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(node_list, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))


#DeepWalk requires different versions of packages to run. To achieve it, you could:
# git clone https://github.com/phanein/deepwalk.git
# cd deepwalk
# pip install -r requirements.txt
# python setup.py install
edge_list_file = open(os.getcwd() + "\\data\\q3_edge_list.txt")
edge_list = edge_list_file.readlines()
for i in range(len(edge_list)):
    output_file = open(os.getcwd() + "\\data\\graphs\\q3_edge_list" + str(i) + ".edgelist", "a")
    for edge in edge_list[i]:
        print(edge.strip(), file=output_file)
#then run deepwalk --input data/graphs/q3_edge_list_i.edgelist --output q3_edge_list_i.embeddings
dic = {}
for i in range(len(edge_list)):
    embed_file = open("q3_edge_list" + str(i) + ".embeddings")
    lines = embed_file.readlines()
    for j in range(len(lines)):
        if j != 0:
            key_ = lines[j].strip().split(" ")[0]
            values = lines[j].strip().split(" ")[1:]
            dic_[key_ + "_" + str(i)] = [values]
print("DeepWalk Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(node_list, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))


class our_graph2vec(karateclub.Graph2Vec):
    def fit(self, graphs: List[nx.classes.graph.Graph]):
        self._set_seed()
        graphs = self._check_graphs(graphs)
        documents = [WeisfeilerLehmanHashing(graph, self.wl_iterations, self.attributed, self.erase_base_features) for
                     graph in graphs]
        documents = [TaggedDocument(words=doc.get_graph_features(), tags=[str(i)]) for i, doc in enumerate(documents)]

        model = Doc2Vec(documents,
                        vector_size=self.dimensions,
                        window=0,
                        min_count=self.min_count,
                        dm=0,
                        sample=self.down_sampling,
                        workers=self.workers,
                        epochs=self.epochs,
                        alpha=self.learning_rate,
                        seed=self.seed)

        self._embedding = model

    def get_embedding(self):
        return self._embedding


graph2vec = our_graph2vec()
graph2vec.fit(graphs)
embedding = graph2vec.get_embedding()
keys_ = embedding.wv.index_to_key
values_ = embedding.wv.vectors
dic_ = {}
for i in range(len(keys_)):
    dic_[keys_[i]] = [values_[i]]
print("Graph2vec Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(node_list, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))

