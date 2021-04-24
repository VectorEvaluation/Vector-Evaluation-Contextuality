import fasttext
import gensim
import glove
import numpy as np
from nltk.tokenize import word_tokenize
import evaluation
import os
import torch
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer


def word2vec(text_sentence, vec_size=100, win=5, min_count=1, workers=4):
    word_list = gensim.models.Word2Vec(text_sentence, vector_size=vec_size, window=win, min_count=min_count, workers=workers)
    return word_list


def ftext(text_file, model='skipgram'):
    word_list = fasttext.train_unsupervised(text_file, model=model)
    return word_list


def glove2vec(text_sentence, win=10, noc=1, lr=0.05, epochs=10, nothr=1, verbose=True):
    corpus_model = glove.Corpus()
    corpus_model.fit(text_sentence, window=win)
    word_list = glove.Glove(no_components=noc, learning_rate=lr)
    word_list.fit(corpus_model.matrix, epochs=epochs, no_threads=nothr, verbose=verbose)
    word_list.add_dictionary(corpus_model.dictionary)
    return word_list


def bert(training, testing, fine_tune):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    testing_data = np.load(testing)
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
    dic = {}
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
                if token_str not in dic.keys():
                    new_list = [token_vecs_cat[i]]
                    dic[token_str] = new_list
                else:
                    dic[token_str].append(token_vecs_cat[i])
    return dic


def getting_data(address_link):
    lines = np.load(address_link)
    output = []
    label = []
    for i in range(len(lines)):
        if i % 2 == 0:
            line_out = word_tokenize(lines[i])
            output.append(line_out)
        else:
            label.append(lines[i])
    return output, label


root = os.getcwd()
training_data, training_label = getting_data(root + "\\data\\training.npy")
testing_data, testing_label = getting_data(root + "\\data\\testing.npy")
word2vec_out = word2vec(testing_data)
keys_ = word2vec_out.wv.index_to_key
values_ = word2vec_out.wv.vectors
dic_ = {}
for i in range(len(keys_)):
    dic_[keys_[i]] = [values_[i]]
print("Word2vec Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(testing_data, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))

glove_out = glove2vec(testing_data)
dic_ = {}
keys_ = glove_out.dictionary
for key_ in keys_:
   dic_[key_] = [glove_out.word_vectors[key_]]
print("GloVe Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(testing_data, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))

dic_ = {}
fast_text_out = ftext(root + "\\data\\testing.npy")
keys_ = fast_text_out.words
for key_ in keys_:
    dic_[key_] = [fast_text_out[key_]]
print("FastText Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(testing_data, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))

#Becasue the EMLO requires using the docker, we cannot directly run it in one file. To evaluate the performance, please refer to the github repository to generate the dictionary for all tokens as dic_ and run the evaluation below:
#print("Elmo Results:")
#print("Self Similarity: ", evaluation.self_sim(dic_))
#print("MEV: ", evaluation.mev(dic_))
#print("Intro Similarity: ", evaluation.intro_sim(testing_data, dic_))
#print("Anisotropy: ", evaluation.anisotropy(dic_))


dic_ = bert(root + "\\data\\training.npy", root + "\\data\\testing.npy", False)
print("Bert Results:")
print("Self Similarity: ", evaluation.self_sim(dic_))
print("MEV: ", evaluation.mev(dic_))
print("Intro Similarity: ", evaluation.intro_sim(testing_data, dic_))
print("Anisotropy: ", evaluation.anisotropy(dic_))
