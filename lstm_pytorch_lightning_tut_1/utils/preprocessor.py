import sys
from distutils.log import info
import string
import re
import pickle
import os

import torch
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords 
from collections import Counter

from json.tool import main
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

class preprocessor():
    def __init__(self,
                 batch_size,
                 x_train_dir = "./datasets/imdb/preprocessed/x_train.pt",
                 y_train_dir = "./datasets/imdb/preprocessed/y_train.pt",
                 x_val_dir = "./datasets/imdb/preprocessed/x_val.pt",
                 y_val_dir = "./datasets/imdb/preprocessed/y_val.pt",
                 info_dir = "./datasets/imdb/preprocessed/info.pkl") -> None:
                 
        nltk.download('stopwords')
        self.device = torch.device("cuda:0")
        self.batch_size = batch_size

        self.x_train_dir = x_train_dir
        self.y_train_dir = y_train_dir
        self.x_val_dir = x_val_dir
        self.y_val_dir = y_val_dir

        self.info_dir = info_dir

    def clean_string(self, s) -> string:
        s = re.sub(r"[^\w\s]", '', s)
        s = re.sub(r"\s+", '', s)
        s = re.sub(r"\d", '', s)
        return s

    def sentence_padding(self, sent, seq_len):
        features = np.zeros((len(sent), seq_len),dtype=int)
        for ii, review in enumerate(sent):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features

    def tokenizer(self, x_train, y_train, x_val, y_val):
        word_list = []

        stop_words = set(stopwords.words('english')) 
        for sent in tqdm(x_train, desc="Creating Corpus"):
            for word in sent.lower().split():
                word = self.clean_string(word)
                if word not in stop_words and word != '':
                    word_list.append(word)
    
        corpus = Counter(word_list)
        # sorting on the basis of most common words
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]
        # creating a dict
        onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

        # Tokenizer
        final_list_train,final_list_test = [],[]
        for sent in tqdm(x_train, desc="Tokenizing train"):
                final_list_train.append([onehot_dict[self.clean_string(word)] for word in sent.lower().split() 
                                         if self.clean_string(word) in onehot_dict.keys()])
        for sent in tqdm(x_val, desc="Tokenizing validation"):
                final_list_test.append([onehot_dict[self.clean_string(word)] for word in sent.lower().split() 
                                        if self.clean_string(word) in onehot_dict.keys()])

        encoded_train = [1 if label =='positive' else 0 for label in y_train]  
        encoded_test = [1 if label =='positive' else 0 for label in y_val] 
        return np.array(final_list_train), np.array(encoded_train), np.array(final_list_test), np.array(encoded_test),onehot_dict

    def load_dataset(self):

        if os.path.exists(self.x_train_dir) and os.path.exists(self.y_train_dir) and os.path.exists(self.x_val_dir) and os.path.exists(self.y_val_dir) :
            print("Load Preprocessed Dataset")
            
            x_train = torch.load(self.x_train_dir)
            y_train = torch.load(self.y_train_dir)

            x_val = torch.load(self.x_val_dir)
            y_val = torch.load(self.y_val_dir)

            with open(self.info_dir, 'rb') as handler:
                info_data = pickle.load(handler)

            vocab_size = info_data["vocab_size"]
            print("Vocab size = ", vocab_size)

        else:
            data_imdb = pd.read_csv("datasets/imdb/IMDB Dataset.csv")
            data_imdb["label"] = data_imdb["sentiment"].apply(lambda x: 1 if x == "positive" else 0 )
            print(data_imdb.head())
            print(data_imdb.shape)

            X,y = data_imdb['review'].values, data_imdb['sentiment'].values
            x_train, x_val, y_train, y_val = train_test_split(X,y,stratify=y)
            print(f'shape of train data is {x_train.shape}')
            print(f'shape of test data is {x_val.shape}')

            x_train, y_train, x_val, y_val, vocab = self.tokenizer(x_train, y_train, x_val, y_val)
            vocab_size = len(vocab)

            x_train = self.sentence_padding(x_train, 500)
            x_val = self.sentence_padding(x_val, 500)

            x_train = torch.from_numpy(x_train)
            y_train = torch.from_numpy(y_train)

            x_val = torch.from_numpy(x_val)
            y_val = torch.from_numpy(y_val)

            torch.save(x_train, self.x_train_dir)
            torch.save(y_train, self.y_train_dir)

            torch.save(x_val, self.x_val_dir)
            torch.save(y_val, self.y_val_dir)

            with open(self.info_dir, "wb") as handler:
                pickle.dump({"vocab_size": vocab_size}, handler, protocol=pickle.HIGHEST_PROTOCOL)
            
        x_train = x_train.to(self.device)
        y_train = y_train.to(self.device)

        x_val = x_val.to(self.device)
        y_val = y_val.to(self.device)

        train_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_data, shuffle=True, batch_size = self.batch_size)
        val_loader = DataLoader(val_data, shuffle=True, batch_size = self.batch_size)

        return train_loader, val_loader, vocab_size
