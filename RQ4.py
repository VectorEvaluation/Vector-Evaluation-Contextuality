import numpy as np
import torch
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer
import gensim
import os
from sklearn import svm
import evaluation


def word2vec(text_sentence, vec_size=100, win=5, min_count=1, workers=4):
    word_list = gensim.models.Word2Vec(text_sentence, vector_size=vec_size, window=win, min_count=min_count, workers=workers)
    return word_list


def bert(training, testing_1, testing_2, fine_tune):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    testing_data_1 = np.load(testing_1)
    testing_data_2 = np.load(testing_2)
    testing_data = np.concatenate((testing_data_1, testing_data_2))
    training_data = np.load(training)
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, )
    if fine_tune:
        train_data = []
        eval_data = []
        for i in range(len(training_data)):
            if i % 2 == 0 and i < len(training_data)*0.8:
                train_data.append(training_data[i])
            else:
                if i % 2 == 0:
                    eval_data.append(training_data[i])
        inputs = tokenizer(train_data, padding="max_length", truncation=True)
        training_args = TrainingArguments(output_dir=os.getcwd() + "\\data\\", do_eval=False)
        trainer = Trainer(model=model, args=training_args, train_dataset=inputs, eval_dataset=eval_data)
        trainer.train()
    output = []
    model.eval()
    for i in range(len(testing_data)):
        if i%2 == 0:
            sentence = "[CLS] " + testing_data[i] + " [SEP]"
            tokenized_sentence = tokenizer.tokenize(sentence)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
            segments_ids = [1] * len(tokenized_sentence)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            with torch.no_grad():
                outputs = model(tokens_tensor, segments_tensors)
                hidden_states = outputs[2]
                token_embeddings = torch.stack(hidden_states, dim=0)
                token_embeddings = torch.squeeze(token_embeddings, dim=1)
                token_embeddings = token_embeddings.permute(1, 0, 2)
                token_vecs_cat = []
                for token in token_embeddings:
                    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
                    token_vecs_cat.append(cat_vec)
            for i, token_str in enumerate(tokenized_sentence):
                output.append(np.array(token_vecs_cat[i]))
    return output[:len(testing_data_1)], output[len(testing_data_1):]


root = os.getcwd()
train_path = root + "\\data\\q4_train.npy"
test_path = root + "\\data\\q4_test.npy"
embed_path = root + "\\data\\q4_embed.npy"
training_data_ = np.load(train_path)
new_ = []
training_label = []
for i in range(len(training_data_)):
    if i % 2 == 0:
        new_.append(training_data_[i])
    else:
        training_label.append(training_data_[i])
training_data_ = new_
testing_data_ = np.load(test_path)
new_ = []
testing_label = []
for i in range(len(testing_data_)):
    if i % 2 == 0:
        new_.append(testing_data_[i])
    else:
        testing_label.append(training_data_[i])
testing_data_ = new_
bert_embedding_train, bert_embedding_testing = bert(embed_path, train_path, test_path, True)
lines = np.load(embed_path)
statement = []
for i in range(len(lines)):
    if i % 2 == 0:
        statement.append(lines[i])
word2vec_embedding = word2vec(statement)
keys_ = word2vec_embedding.wv.index_to_key
values_ = word2vec_embedding.wv.vectors
dic_ = {}
for i in range(len(keys_)):
    dic_[keys_[i]] = [values_[i]]
replacing_label_train = root + "\\data\\q4_replacing_train.npy"
replacing_label_train = np.load(replacing_label_train)
for i in range(len(replacing_label_train)):
    for j in range(len(replacing_label_train[i])):
        if replacing_label_train[i][j] == 1:
            bert_embedding_train[i][j] = dic_[training_data_[i][j]]
replacing_label_test = root + "\\data\\q4_replacing_train.npy"
replacing_label_test = np.load(replacing_label_test)
for i in range(len(replacing_label_test)):
    for j in range(len(replacing_label_test[i])):
        if replacing_label_test[i][j] == 1:
            replacing_label_test[i][j] = dic_[testing_data_[i][j]]
classifier = svm.SVC()
classifier.fit(bert_embedding_train, training_label)
output = classifier.predict(testing_data_)
evaluation.evaluate(testing_label, output)
