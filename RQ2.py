import evaluation
import numpy as np
import os
from tree_lstm import TreeLSTM, BatchedTree
import pickle
import torch.optim as optim
import torch
import torch.utils.data as Data


train_input_file = open(os.getcwd() + "//data//tree_training.pkl")
train_label = np.load(os.getcwd() + "//data//tree_training_label.npy")
test_input_file = open(os.getcwd() + "//data//tree_testing.pkl")
test_label = np.load(os.getcwd() + "//data//tree_testing_label.npy")
train_data = pickle.load(train_input_file)
test_data = pickle.load(test_input_file)
train_data = BatchedTree(train_data)
test_data = BatchedTree(test_data)
batch_size = 8
model = TreeLSTM(x_size=100, h_size=25, dropout=0.3, cell_type='n_ary', n_ary=4)

train_set = Data.TensorDataset(torch.tensor(train_data), torch.tensor(train_label))
train_loader = Data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = Data.TensorDataset(torch.tensor(test_data), torch.tensor(test_label))
test_loader = Data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
output = []
with torch.no_grad():
    for data in test_loader:
        tree, labels = data
        outputs = model(tree)
        embeddings = outputs.get_hidden_state()
        final_embedding = embeddings[-1]
        output.append(final_embedding)
dic_ = {}
data_keys = np.load(os.getcwd() + "//data//tree_label.npy")
data_groups = np.load(os.getcwd() + "//data//tree_group.npy")
for i in range(len(output)):
    if hash(data_keys[i]) not in dic_.keys():
        new_list = [output[i]]
        dic_[hash(data_keys[i])] = new_list
    else:
        dic_[hash(data_keys[i])].append(output[i])
print("Tree-LSTM Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(data_groups, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))


#ASTNN please fix the code as follow and then run the code:
# Fixing code:
# pipline.py line 127 changed to: ppl = Pipeline('4:0:1', 'data/')
# model.py line 150 add code: rep_output = tf.reshape(gru_out, (1, -1)) // rep_output = tf.squeeze(rep_output)
# model.py line 153 changed to: return y, rep_output
# train.py line 127 add code: representation = []
# train.py line 76, 104, 137 changed to: output, _ = model(train_inputs)//  output, _ = model(val_inputs) //output, rep_output = model(test_inputs)// representation.append(rep_output)
# train.py line 145 add code: np.save(os.path.abspath(os.path.join(os.getcwd(), "..")) + "//data//astnn_out.npy", np.array(representation))
dic_ = {}
data_input = np.load(os.getcwd() + "//data//astnn_out.npy")
data_keys = np.load(os.getcwd() + "//data//tree_label.npy")
data_groups = np.load(os.getcwd() + "//data//tree_group.npy")
for i in range(len(data_input)):
    if hash(data_keys[i]) not in dic_.keys():
        new_list = [data_input[i]]
        dic_[hash(data_keys[i])] = new_list
    else:
        dic_[hash(data_keys[i])].append(data_input[i])
print("ASTNN Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(data_groups, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))


#TBCNN requires different versions of packages. Please fix the original code as follow and get the results as dic_ to do the following evaluation:
# Fixing code:
# network.py line 17 add code: rep_output = tf.reshape(pooling, (1, -1)) // rep_output = tf.squeeze(rep_output)
# network.py line 28 changed to: return nodes, children, hidden, rep_output
# train.py line 26 changed to: nodes_node, children_node, hidden_node, rep_node = network.init_net(
# test.py line 23 changed to: nodes_node, children_node, hidden_node, rep_node = network.init_net(
# test.py line 44 add code: representations = []
# test.py line 55 add code: output_ = sess.run([rep_node],
#                                feed_dict={
#                                nodes_node: nodes,
#                                children_node: children,
#                                }
#                            )
#                            representations.append(output_)
# test.py line 59 add code: np.save(os.path.abspath(os.path.join(os.getcwd(), "..")) + "//data//tbcnn_out.npy", np.array(representation))
# representations is the list that stores the representation vector.
# Evaluation:
dic_ = {}
data_input = np.load(os.getcwd() + "//data//tbcnn_out.npy")
data_keys = np.load(os.getcwd() + "//data//tree_label.npy")
data_groups = np.load(os.getcwd() + "//data//tree_group.npy")
for i in range(len(data_input)):
    if hash(data_keys[i]) not in dic_.keys():
        new_list = [data_input[i]]
        dic_[hash(data_keys[i])] = new_list
    else:
        dic_[hash(data_keys[i])].append(data_input[i])
print("TBCNN Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(data_groups, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))


#TreeCaps requires different versions of packages. Please fix the original code as follow and get the results as dic_ to do the following evaluation:
#Fixing code:
# network.py line 105 add code: Caps_output = tf.reshape(codeCaps, (1, -1)) // Caps_output = tf.squeeze(Caps_output)
# network.py line 110 changed to: return nodes, children, out, Caps_output
# main.py line 57, 150 change to: nodes_node, children_node, codecaps_node, representation_node = network.init_net_treecaps(num_feats,len(labels))
# main.py line 172 add code: representations = []
# main.py line 183 add code: output_ = sess.run([representation_node],
#                                feed_dict={
#                                nodes_node: nodes,
#                                children_node: children,
#                                }
#                            )
#                            representations.append(output_)
# representations is the list that stores the representation vector.
# main.py line 186 add code: np.save(os.path.abspath(os.path.join(os.getcwd(), "..")) + "//data//treecaps_out.npy", np.array(representation))
# Evaluation:
dic_ = {}
data_input = np.load(os.getcwd() + "//data//treecaps_out.npy")
data_keys = np.load(os.getcwd() + "//data//tree_label.npy")
data_groups = np.load(os.getcwd() + "//data//tree_group.npy")
for i in range(len(data_input)):
    if hash(data_keys[i]) not in dic_.keys():
        new_list = [data_input[i]]
        dic_[hash(data_keys[i])] = new_list
    else:
        dic_[hash(data_keys[i])].append(data_input[i])
print("TreeCaps Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(data_groups, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))

