from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, auc


def intro_sim(sentences, dic, size=100):
    output = 0
    counter = 0
    for sentence in sentences:
        embedding = []
        for token in sentence:
            if token in dic.keys():
                embedding.append(dic[token][0])
            else:
                embedding.append(np.zeros(size))
        embedding = np.array(embedding)
        average = np.mean(embedding, axis=0)
        total = 0
        for embed in embedding:
            total = total + cosine_similarity(average.reshape(1, -1), embed.reshape(1, -1))
        output = output + total/len(embedding)
        counter = counter + 1
    output = output/counter
    return output[0][0]


def anisotropy(dic):
    output = 0
    counter = 0
    for key in dic.keys():
        for new_key in dic.keys():
            if key != new_key:
                for i in range(len(dic[key])):
                    for j in range(len(dic[new_key])):
                        output = output + cosine_similarity(dic[key][i].reshape(1, -1), dic[new_key][j].reshape(1, -1))
                        counter = counter + 1
    output = output/counter
    return output[0][0]


def mev(sum_dic):
    output = 0
    for key in sum_dic.keys():
        mev_ = 0
        for i in range(len(sum_dic[key])):
             mev_ = mev_ + np.dot(sum_dic[key][i], np.transpose(sum_dic[key][i]))
        mev_ = np.dot(sum_dic[key][0], np.transpose(sum_dic[key][0]))/mev_
        output = output + mev_
    return output/len(sum_dic.keys())


def self_sim(sum_dic):
    output = 0
    for key in sum_dic.keys():
        if len(sum_dic[key]) == 1:
            output = output + 1
        else:
            selfsim = 0
            for i in range(len(sum_dic[key])):
                for j in range(len(sum_dic[key])-i-i):
                    selfsim = selfsim + cosine_similarity(np.array(sum_dic[key][i]), sum_dic[key][j+i+1])
            selfsim = selfsim/(len(sum_dic[key])*len(sum_dic[key]) - len(sum_dic[key]))
            output = output + selfsim
    return output/len(sum_dic.keys())


def evaluate(truth, predict):
    print("Recall: ", recall_score(truth, predict))
    print("Precision: ", precision_score(truth, predict))
    print("F1 Score: ", f1_score(truth, predict))
    print("AUC: ", auc(truth, predict))
