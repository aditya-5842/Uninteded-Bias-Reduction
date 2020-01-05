import os, time, re

import numpy as np
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from tqdm import tqdm
import pickle
from sklearn import metrics
from sklearn import model_selection
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from prettytable import PrettyTable

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

from models import *
from functions import *
##################################################################################################
##################################################################################################
print("loading the dataset...")
#keep only those columns which are required
identity_columns = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian',
                    'jewish','muslim', 'black', 'white', 'psychiatric_or_mental_illness']
cols = ["id","comment_text"] + identity_columns + ["target"]
df = load_dataset(cols)
print("Now shape of the data: ", df.shape)
####################################################################################################
####################################################################################################
def convert_to_bool(data, cols):
    """
    function to convert values into boolean
    """
    for col in cols:
        data[col] = np.where(data[col] >= 0.5, True, False)
    return data

print("converting into boolean...")
df = df.astype({"comment_text":"str"})
df = convert_to_bool(df, identity_columns)
# converting  target feature to 0 and 1
df["target"] = df["target"].apply(lambda x: 1 if x >= 0.5 else 0)
######################################################################################################
######################################################################################################
print("creating the text based features...")
tic = time.time()
df = feature(df)
print("processing text data...")
df.loc[:,"comment_text"] = df.apply(text_process, axis = 1)
print("Time take to process the text data: {:.2f} seconds".format(time.time()-tic))
######################################################################################################
######################################################################################################
print("Splitting the data...")
X = df[[col for col in df.columns]]
Y = df[["target"]]
X_train, X_test, y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8,
                                                                    stratify = Y, random_state = 42)
X_test, X_cv, y_test, y_cv = model_selection.train_test_split(X_test, Y_test,train_size = 0.5,
                                                              stratify = Y_test, random_state = 42)
######################################################################################################
######################################################################################################
print("getting hand-crafted features...")
hand_crafted_train = X_train[X_train.columns[-14:]].values
hand_crafted_cv    = X_cv[X_train.columns[-14:]].values
hand_crafted_test  = X_test[X_train.columns[-14:]].values

print("creating StandardScaler...")
if os.path.isfile("Models/std.pkl"):
    print("loadin the StandardScaler form the disk...")
    with open("Models/std.pkl","rb") as f:
        std = pickle.load(f)
else:
    print("StandardScaler is not in disk, so let's create that...")
    std = StandardScaler()
    std.fit(hand_crafted_train)
    print("saving the StandardScaler in disk...")
    with open("Models/std.pkl", "wb") as f:
        pickle.dump(std, f)
hand_crafted_train = std.transform(hand_crafted_train)
hand_crafted_cv    = std.transform(hand_crafted_cv)
hand_crafted_test  = std.transform(hand_crafted_test)

print("creating bi-gram BoW vectors...")
if os.path.isfile("Models/vectorizer.pkl"):
    print("loadin the vectorizer form the disk...")
    with open("Models/vectorizer.pkl","rb") as f:
        vectorizer = pickle.load(f)
else:
    print("vec is not in disk, so let's create that...")
    vectorizer = CountVectorizer(ngram_range=(1,1), min_df = 1, max_features = 10000)
    vectorizer.fit(X_train["comment_text"].values)
    print("saving the vectorizer in disk...")
    with open("Models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
uni_bow_train = vectorizer.transform(X_train["comment_text"].values)
uni_bow_cv    = vectorizer.transform(X_cv["comment_text"].values)
uni_bow_test  = vectorizer.transform(X_test["comment_text"].values)

print("loading the W2V model...")
w2v_loaded_dict = load_w2v()

print("stacking the vectorized text data and hand-crafted features...")
train_bow_uni = hstack((uni_bow_train, hand_crafted_train)).tocsr()
cv_bow_uni = hstack((uni_bow_cv, hand_crafted_cv)).tocsr()
test_bow_uni = hstack((uni_bow_test, hand_crafted_test)).tocsr()

######################################################################################################
######################################################################################################
print("creating tokens...")
if os.path.isfile("Models/tokens.pkl"):
    print("loadin the vectorizer form the disk...")
    with open("Models/tokens.pkl","rb") as f:
        tokens = pickle.load(f)
else:
    print("vec is not in disk, so let's create that...")
    tokens = Tokenizer()
    tokens.fit_on_texts(X_train["comment_text"].values)
    print("saving the tokenizer in disk...")
    with open("Models/tokens.pkl", "wb") as f:
        pickle.dump(tokens, f)

max_lenght = 400
# padding the encoded data to make each datapoint of same dimension
encoded_text_train = text_to_seq(X_train["comment_text"].values, tokens, max_lenght)
encoded_text_cv    = text_to_seq(X_cv["comment_text"].values, tokens, max_lenght)
encoded_text_test  = text_to_seq(X_test["comment_text"].values, tokens, max_lenght)

# gettting the length of unique words in train data, and adding (+1) becasue of zeros padding and words are encoded from 1 to n
vocab_size = len(tokens.word_index) + 1
embedding_matrix1 = create_embedding_matrix(vocab_size, tokens, w2v_loaded_dict)
######################################################################################################
########################## now we've data to train our models. #######################################
######################################################################################################

n_epochs = 40
batch_size = 1024
print("defining callbacks...")
es = EarlyStopping(monitor = 'val_loss', mode = 'min', min_delta = 0.001, patience = 10, verbose = 1)

if not os.path.isfile("Models/log_reg.pkl"):
    print("Trained logisitic regression Model is not in disk,So Training logisitic regression...")
    model1 = log_reg()
    print(model1)
    model1.fit(train_bow_uni,y_train)
    print("saving the model in disk...")
    with open("Models/log_reg.pkl", "wb") as f:
        pickle.dump(model1, f)
else:
    print("trained logisitic regression model is disk, so loading the model from disk...")
    with open("Models/log_reg.pkl", "rb") as f:
        model1 = pickle.load(f)
######################################################################################################
######################################################################################################
if not os.path.isfile("Models/GBDT.pkl"):
    print("Trained GBDT Model is not in disk,So Training GBDT...")
    model2 = gbdt_model()
    print(model2)
    train_bow_uni = train_bow_uni.astype("float64")
    model2.fit(train_bow_uni,y_train)
    print("saving the model in disk...")
    with open("Models/GBDT.pkl", "wb") as f:
        pickle.dump(model2, f)
else:
    print("trained GBDT model is disk, so loading the model from disk...")
    with open("Models/GBDT.pkl", "rb") as f:
        model2 = pickle.load(f)
######################################################################################################
######################################################################################################
if not os.path.isfile("Models/DL_model_1.h5"):
    print("Trained DL Model-1 is not in disk,So Training it...")
    model3 = DL_model_1(uni_bow_train)
    print(model3.summary())
    mc = ModelCheckpoint("Models/DL_model_1.h5", monitor = "val_acc", mode = 'max', save_best_only = True, verbose = 1)
    model3.compile(loss='binary_crossentropy', optimizer = "rmsprop", metrics=['acc'])
    model3.fit(uni_bow_train, y_train, validation_data = (uni_bow_cv, y_cv),
              epochs = n_epochs, batch_size = 32, callbacks = [es, mc])
else:
    print("trained logisitic regression model is disk, so loading the model from disk...")
    model3 = load_model("Models/DL_model_1.h5")
######################################################################################################
######################################################################################################
if not os.path.isfile("Models/DL_model_2.h5"):
    print("Trained DL Model-2 is not in disk,So Training it...")
    model4 = DL_model_2(max_lenght, vocab_size, embedding_matrix1)
    print(model4.summary())
    mc = ModelCheckpoint("Models/DL_model_2.h5", monitor = "val_acc", mode = 'max', save_best_only = True, verbose = 1)
    model4.compile(loss='binary_crossentropy', optimizer = "rmsprop", metrics=['acc'])
    model4.fit(encoded_text_train, y_train, validation_data = (encoded_text_cv, y_cv),
               epochs = n_epochs, batch_size = batch_size, callbacks = [es, mc])
else:
    print("trained logisitic regression model is disk, so loading the model from disk...")
    model4 = load_model("Models/DL_model_2.h5")
######################################################################################################
######################################################################################################
if not os.path.isfile("Models/DL_model_3.h5"):
    print("Trained DL Model-3 is not in disk,So Training it...")
    model5 = DL_model_3(max_lenght, vocab_size, embedding_matrix1)
    print(model5.summary())
    mc = ModelCheckpoint("Models/DL_model_3.h5", monitor = "val_acc", mode = 'max', save_best_only = True, verbose = 1)
    model5.compile(loss='binary_crossentropy', optimizer = "rmsprop", metrics=['acc'])
    model5.fit(encoded_text_train, y_train, validation_data = (encoded_text_cv, y_cv),
               epochs = n_epochs, batch_size = batch_size, callbacks = [es, mc])
else:
    print("trained logisitic regression model is disk, so loading the model from disk...")
    model4 = load_model("Models/DL_model_3.h5")
######################################################################################################
######################################################################################################
if not os.path.isfile("Models/DL_model_2.h5"):
    print("Trained DL Model-4 is not in disk,So Training it...")
    model6 = DL_model_4(max_lenght, vocab_size, embedding_matrix1)
    print(model6.summary())
    mc = ModelCheckpoint("Models/DL_model_3.h5", monitor = "val_acc", mode = 'max', save_best_only = True, verbose = 1)
    model6.compile(loss='binary_crossentropy', optimizer = "rmsprop", metrics=['acc'])
    model6.fit(encoded_text_train, y_train, validation_data = (encoded_text_cv, y_cv),
               epochs = n_epochs, batch_size = batch_size, callbacks = [es, mc])
else:
    print("trained logisitic regression model is disk, so loading the model from disk...")
    model6 = load_model("Models/DL_model_4.h5")
######################################################################################################
######################################################################################################
print("*"*20,"All models are trained/loaded.","*"*20)

# mdoel_3 prediction
print("\npredicting for DL model-1...\n")
y_pred_tr_1 = model_3.predict(uni_bow_train, batch_size = 8192, verbose=1)
y_pred_cv_1 = model_3.predict(uni_bow_cv, batch_size = 8192, verbose=1)
y_pred_te_1 = model_3.predict(uni_bow_test, batch_size = 8192, verbose=1)

# mdoel_2 prediction
print("\npredicting for DL model-2...\n")
y_pred_tr_2 = model_4.predict(encoded_text_train, batch_size = 8192, verbose=1)
y_pred_cv_2 = model_4.predict(encoded_text_cv, batch_size = 8192, verbose=1)
y_pred_te_2 = model_4.predict(encoded_text_test, batch_size = 8192, verbose=1)

# mdoel_3 prediction
print("\npredicting for  DL model-3...\n")
y_pred_tr_3 = model_5.predict(encoded_text_train, batch_size = 8192, verbose=1)
y_pred_cv_3 = model_5.predict(encoded_text_cv, batch_size = 8192, verbose=1)
y_pred_te_3 = model_5.predict(encoded_text_test, batch_size = 8192, verbose=1)

# # mdoel_4 prediction
print("\npredicting for  DL model-4...\n")
y_pred_tr_4 = model_6.predict(encoded_text_train, batch_size = 1024, verbose=1)
y_pred_cv_4 = model_6.predict(encoded_text_cv, batch_size = 1024, verbose=1)
y_pred_te_4 = model_6.predict(encoded_text_test, batch_size = 1024, verbose=1)

# model_6 prediction
print("\npredicting for logisitic regression...\n")
y_pred_tr_6 = model_1.predict(train_bow_uni)
y_pred_cv_6 = model_1.predict(cv_bow_uni)
y_pred_te_6 = model_1.predict(test_bow_uni)

train_bow_uni = train_bow_uni.astype("float64")
cv_bow_uni    = cv_bow_uni.astype("float64")
test_bow_uni  = test_bow_uni.astype("float64")

# model_5 prediction
print("\npredicting for GBDT...\n")
y_pred_tr_5 = model_2.predict(train_bow_uni)
y_pred_cv_5 = model_2.predict(cv_bow_uni)
y_pred_te_5 = model_2.predict(test_bow_uni)

tr = [y_pred_tr_1, y_pred_tr_2, y_pred_tr_3, y_pred_tr_4, y_pred_tr_5, y_pred_tr_6]
cv = [y_pred_cv_1, y_pred_cv_2, y_pred_cv_3, y_pred_cv_4, y_pred_cv_5, y_pred_cv_6]
te = [y_pred_te_1, y_pred_te_2, y_pred_te_3, y_pred_te_4, y_pred_te_5, y_pred_te_6]

subgroups    = identity_columns
actual_label = "target"
pred_label   = "pred_target"
######################################################################################################
######################################################################################################
def report_stacked_mode():
    alphas = [2*0.81, 2.5*0.88, 4*0.89, 5.5*0.90, 1.2*0.78, 1.1*0.77]

    # define arrays of zeros to store above predicted values
    a = np.zeros((X_train.shape[0], 6))
    b = np.zeros((X_cv.shape[0], 6))
    c = np.zeros((X_test.shape[0], 6))

    # storing with wieghtage
    for i in range(6):
        a[:,i] = alphas[i] * tr[i].flatten()/sum(alphas)
        b[:,i] = alphas[i] * cv[i].flatten()/sum(alphas)
        c[:,i] = alphas[i] * te[i].flatten()/sum(alphas)

    # final prediction
    X_train["pred_target"] = np.sum(a, axis = 1)
    X_cv["pred_target"]    = np.sum(b, axis = 1)
    X_test["pred_target"]  = np.sum(c, axis = 1)

    final_train_auc, _ = return_final_metric(X_train, subgroups, actual_label, pred_label, verbose = False)
    final_cv_auc, _ = return_final_metric(X_cv, subgroups, actual_label, pred_label, verbose = False)
    final_test_auc, _ = return_final_metric(X_test, subgroups, actual_label, pred_label, verbose = False)

    print("Final metric for:\nTrain: {:.5f}\nCV: {:.5f}\nTest: {:.5f}".format(final_train_auc,
                                                                              final_cv_auc,
                                                                              final_test_auc))

    print("\n\nPloting Confusion matrix for whole data....\n")
    plot_confusion_matrix(X_train, X_cv, X_test)

    print("\n\nPlotting confusion matrix indentity-wise ...\n")
    plot_confusion_for_each_identity(X_train, X_cv, X_test, subgroups)

    return final_train_auc, final_cv_auc, final_test_auc
######################################################################################################
######################################################################################################
report_stacked_mode()
