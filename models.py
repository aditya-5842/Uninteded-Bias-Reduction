import os, time, re

import numpy as np
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# https://stackoverflow.com/a/14463362/12005970
import warnings
warnings.filterwarnings("ignore")

# import DL libraries
import tensorflow as tf
import keras
from keras.layers import Input, Conv1D, Dense, Activation, LSTM, Reshape, add
from keras.layers import SpatialDropout1D, concatenate, CuDNNGRU,CuDNNLSTM, Bidirectional
from keras.layers import BatchNormalization, Dropout, concatenate, Embedding
from keras.layers import Flatten, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

############################################################################################################################
############################################################################################################################
def log_reg():
    '''
    this will return the logisitic regression function.
    '''
    clf = SGDClassifier(loss = "log", penalty="l2", alpha = 0.1,
                        class_weight = {0: 115467, 1: 1328432},
                         random_state = 42)
    return clf

############################################################################################################################
############################################################################################################################
def gbdt_model():
    '''
    This will return the GBDT model.
    '''
    clf = LGBMClassifier(boosting_type='gbdt', class_weight={0: 115467, 1: 1328432},
                        colsample_bytree=1.0, importance_type='split', learning_rate=0.1,
                        max_depth=50, min_child_samples=20, min_child_weight=0.001,
                        min_split_gain=0.0, n_estimators=1000, n_jobs=-1, num_leaves=31,
                        objective=None, random_state=42, reg_alpha=0.0, reg_lambda=0.0,
                        silent=True, subsample=1.0, subsample_for_bin=200000,
                        subsample_freq=0)
    return clf

############################################################################################################################
############################################################################################################################
def DL_model_1(uni_bow_train):
    '''
    This will create the model 1.
    '''
    bow_input = Input(shape = (uni_bow_train.shape[1], ), name = "imput_layer")
    # dense layer
    X = Dense(units = 64, kernel_initializer = 'he_normal', name = "dense_1")(bow_input)
    X = Activation("relu", name = "dense1_layer_activation")(X)
    X = Dropout(0.4, name = "dropout_1")(X)
    # dense layer
    X = Dense(units = 32, kernel_initializer = 'he_normal', name = "dense_2")(X)
    X = BatchNormalization(name = "BN_2")(X)
    X = Activation("relu", name = "dense2_layer_activation")(X)
    X = Dropout(0.3, name = "dropout_2")(X)
    # dense layer
    X = Dense(units = 16, kernel_initializer = 'he_normal', name = "dense_3")(X)
    X = BatchNormalization(name = "BN_3")(X)
    X = Activation("relu", name = "dense3_layer_activation")(X)
    X = Dropout(0.3, name = "dropout_3")(X)
    # dense layer
    X = Dense(units = 8, kernel_initializer = 'he_normal', name = "dense_4")(X)
    X = BatchNormalization(name = "BN_4")(X)
    X = Activation("relu", name = "dense4_layer_activation")(X)
    # output layer
    out = Dense(1, activation='sigmoid', name = "output_layer")(X)
    model = Model(inputs = bow_input, outputs = out, name = "Model_1")
    return model

############################################################################################################################
############################################################################################################################
def DL_model_2(max_lenght, vocab_size, embedding_matrix1):
    '''
    This wil create the model 2.
    '''
    #clearing the graph
    keras.backend.clear_session()
    seq_input  = Input(shape = (max_lenght, ), name = "input_layer")
    embed_text = Embedding(input_dim = vocab_size, output_dim = 200,
                                weights=[embedding_matrix1], input_length = max_lenght,
                                trainable = False, name = 'text_embedding')(seq_input)
    x   = Conv1D(128, 2, activation='relu', padding='same', name = "conv_1d_1")(embed_text)
    x   = MaxPooling1D(5, padding='same', name = "maxpool_1")(x)
    x   = Conv1D(128, 3, activation='relu', padding='same', name = "conv_1d_2")(x)
    x   = MaxPooling1D(5, padding='same', name = "maxpool_2")(x)
    x   = Conv1D(128, 4, activation='relu', padding='same', name = "conv_1d_3")(x)
    x   = MaxPooling1D(40, padding='same', name = "maxpool_3")(x)
    x   = Flatten(name = "flatten_1")(x)
    x   = Dropout(0.3, name = "dropout_1")(x)
    x   = Dense(128, activation='relu', name = "dense_layer_1")(x)
    out = Dense(1, activation='sigmoid', name = "output_layer")(x)
    model = Model(inputs = seq_input, outputs = out, name = "Model_2")
    return model

############################################################################################################################
############################################################################################################################
def DL_model_3(max_lenght, vocab_size, embedding_matrix1):
    '''
    This will creat the DL mdoel-3.
    '''
    #clearing the graph
    keras.backend.clear_session()
    # input layer
    seq_input  = Input(shape = (max_lenght, ), name = "input_layer")
    # embedding layer
    embed_text = Embedding(input_dim = vocab_size, output_dim = 200,
                                weights=[embedding_matrix1], input_length = max_lenght,
                                trainable = False, name = 'text_embedding')(seq_input)
    # lstm layer
    # https://github.com/keras-team/keras/issues/5222#issuecomment-276563109
    X = LSTM(units = 16, dropout = 0.3, recurrent_dropout = 0.3, return_sequences=True,
             name = "lstm_layer_1")(embed_text)
    # lstm layer
    X = LSTM(units = 8, dropout = 0.2, recurrent_dropout = 0.2, return_sequences=True,
                name = "lstm_layer_2")(X)
    # flatten layer
    X = Flatten(name = "flatten_1")(X)
    # Dense layer
    X = Dense(units = 32, kernel_initializer = 'he_normal', name = "dense_1")(X)
    X = BatchNormalization(name = "BN_1")(X)
    X = Activation("relu", name = "dense1_layer_activation")(X)
    X = Dropout(0.3, name = "dropout_1")(X)
    # dense layer
    X = Dense(units = 16, kernel_initializer = 'he_normal', name = "dense_2")(X)
    X = BatchNormalization(name = "BN_2")(X)
    X = Activation("relu", name = "dense2_layer_activation")(X)
    X = Dropout(0.3, name = "dropout_2")(X)
    #dense layer
    X = Dense(units = 8, kernel_initializer = 'he_normal', name = "dense_3")(X)
    X = BatchNormalization(name = "BN_3")(X)
    X = Activation("relu", name = "dense3_layer_activation")(X)
    # output layer
    out = Dense(units = 1, activation = "sigmoid", name = "output_layer")(X)
    # model
    model = Model(inputs = seq_input, outputs = out, name = "model_3")
    return model

############################################################################################################################
############################################################################################################################
def DL_model_4(max_lenght, vocab_size, embedding_matrix1):
    '''
    This function will create the DL model-4.
    It uses CuDNNGRU which run using Nvidea GPU only.
    '''
    #clearing the graph
    keras.backend.clear_session()
    input_s = Input(shape=(max_lenght,))
    grn_s = Embedding(input_dim = vocab_size, output_dim = 200,
                      weights=[embedding_matrix1],input_length = max_lenght,
                      trainable=False)(input_s)
    grn_s = Bidirectional(CuDNNGRU(128, return_sequences=True), name = "bidirectional_1")(grn_s)
    grn_s = Bidirectional(CuDNNGRU(128, return_sequences=True), name = "bidirectional_2")(grn_s)
    hidden = concatenate([GlobalMaxPooling1D()(grn_s),GlobalAveragePooling1D()(grn_s),], name = "concatenate_1")
    hidden = add([hidden, Dense(512, activation='relu')(hidden)])
    hidden = add([hidden, Dense(512, activation='relu')(hidden)])
    out = Dense(1, activation='sigmoid', name = "output_layer")(hidden)
    model = Model(inputs=input_s, outputs = out, name = "Model_4")
    return model
