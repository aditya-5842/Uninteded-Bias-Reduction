import os, time, re

import numpy as np
from numpy import asarray
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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

#####################################################################################################################
#####################################################################################################################
def load_dataset(cols):
    df = pd.read_csv("Data/Train/train1.csv")
    print("Shape of dataset is: ", df.shape)
    df = df[cols]
    return df
#####################################################################################################################
#####################################################################################################################
# list of punctuations
punc = [".", "?", "!", ",", ";", ":", "-", "--", "(", ")", "[", "]", "{", "}", "'", '"', "..."]
# symbols list
symbols = ["@", "#", "$", "%", "^", "&", "*", "~"]

def feature(data):
    """
    this created the designed features
    """
    print("Creating hand crafted features...")
    start = time.time()
    data_df = data.copy()
    # 1.
    print(" For 'word_count' feature...")
    data_df['word_count'] = data_df['comment_text'].apply(lambda x : len(x.split()))

    # 2.
    print(" For 'char_count' feature...")
    data_df['char_count'] = data_df['comment_text'].apply(lambda x : len(x.replace(" ","")))

    # 3.
    print(" For 'word_density' feature...")
    data_df['word_density'] = data_df['word_count'] / (data_df['char_count'] + 1)

    # 4.
    print(" For 'total_length' feature...")
    data_df['total_length'] = data_df['comment_text'].apply(len)

    # 5.
    print(" For 'capitals' feature...")
    data_df['capitals'] = data_df['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))

    # 6.
    print(" For 'caps_vs_length' feature...")
    data_df['caps_vs_length'] = data_df.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)

    # 7.
    print(" For 'punc_count' feature...")
    data_df['punc_count'] = data_df['comment_text'].apply(lambda x : len([a for a in x if a in punc]))

    # 8.
    print(" For 'num_exclamation_marks' feature...")
    data_df['num_exclamation_marks'] =data_df['comment_text'].apply(lambda x: x.count('!'))

    # 9.
    print(" For 'exlamation_vs_punc_count' feature...")
    data_df['exlamation_vs_punc_count'] = data_df['num_exclamation_marks']/data_df['punc_count']

    # 10.
    print(" For 'num_question_marks' feature...")
    data_df['num_question_marks'] = data_df['comment_text'].apply(lambda x: x.count('?'))

    # 11.
    print(" For 'question_vs_punc_count' feature...")
    data_df['question_vs_punc_count'] = data_df['num_question_marks']/data_df['punc_count']

    # 12.
    print(" For 'num_symbols' feature...")
    data_df['num_symbols'] = data_df['comment_text'].apply(lambda x: sum(x.count(w) for w in '*&$%'))

    # 13.
    print(" For 'num_unique_words' feature...")
    data_df['num_unique_words'] = data_df['comment_text'].apply(lambda x: len(set(w for w in x.split())))

    # 14.
    print(" For 'words_vs_unique' feature...")
    data_df['words_vs_unique'] = data_df['num_unique_words'] / data_df['word_count']

    data_df.fillna(0, inplace = True)
    print("\nALL Done!\nTime take for this is {:.4f} seconds".format(time.time() - start))
    return data_df

#####################################################################################################################
#####################################################################################################################
#removing "not" stop word from stop_words
stop_words = set(stopwords.words("english"))
stop_words = stop_words - {"not"}

def text_process(row):
    try:
        text = row["comment_text"]
        text = str(text).lower()
        porter = PorterStemmer()

        #expansion
        text = text.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
        .replace("n't", " not").replace("what's", "what is").replace("it's", "itis")\
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar")\
        .replace("€", " euro ").replace("'ll", " will")

        text = re.sub(r"<.*?>","", text) # removes the htmltags: https://stackoverflow.com/a/12982689

        #special character removal
        text = re.sub('[^a-zA-Z0-9\n]', ' ', text)
        #extra space removal
        text = re.sub('\s+',' ', text)

        # stopword removal
        text_to_words = []
        for word in text.split():
            if word not in stop_words:
                text_to_words.append(word)
            else:
                continue
        text = " ".join(text_to_words)

        # stemming the words
        text = porter.stem(text)
        return text
    except:
        print("There is no value in comment_text, so returnin 'nan'")
        return np.nan

#####################################################################################################################
#####################################################################################################################
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
def load_w2v():
    '''
    this will load the W2V model into dicstionary.
    '''
    file_name = "Data/glove.6B.200d.txt"
    with open(file_name, 'r') as f:
        w2v_loaded_dict = {}
        for line in f:
            values = line.split()
            word = values[0]
            vector = [float(i) for i in values[1:]]
            w2v_loaded_dict[word] = vector
    return w2v_loaded_dict

#####################################################################################################################
#####################################################################################################################
def text_to_seq(texts, keras_tokenizer, max_len):
    """this function  return sequence of text after padding/truncating"""
    x = pad_sequences(keras_tokenizer.texts_to_sequences(texts),
                      maxlen = max_len, padding = 'post',truncating = 'post')
    return x

#####################################################################################################################
#####################################################################################################################
def create_embedding_matrix(vocab_size, tokens, w2v_loaded_dict):
    '''
    This will create the embedding matrix.
    '''
    embedding_matrix1 = np.zeros((vocab_size, 200), dtype = 'float32')
    for word, index in tokens.word_index.items():
        embedding_vector = w2v_loaded_dict.get(word)
        if embedding_vector is not None:
            embedding_matrix1[index] = embedding_vector
    return embedding_matrix1


#####################################################################################################################
#####################################################################################################################
from sklearn.metrics import roc_auc_score
########################################################################################
#######################     function to calculate the AUC        #######################
########################################################################################
def cal_auc(y_true, y_pred):
    "returns the auc value"
    return roc_auc_score(y_true, y_pred)

########################################################################################
#######################  function to calculate the Subgroup AUC  #######################
########################################################################################
def cal_subgroup_auc(data, subgroup, actual_label, pred_label):
    subgroup_examples = data[data[subgroup]]
    return cal_auc(subgroup_examples[actual_label], subgroup_examples[pred_label])

########################################################################################
#######################   function to calculate the BPSN AUC     #######################
########################################################################################
def cal_bpsn_auc(data, subgroup, actual_label, pred_label):
    """This will calculate the BPSN auc"""
    # subset where subgroup is True and target label is 0
    subgroup_negative_examples = data[data[subgroup] & ~data[actual_label]]
    # subset where subgroup is False and target label is 1
    background_positive_examples = data[~data[subgroup] & data[actual_label]]
    # combine above tow sets
    bpsn_examples = subgroup_negative_examples.append(background_positive_examples)
    return cal_auc(bpsn_examples[actual_label], bpsn_examples[pred_label])


########################################################################################
#######################   function to calculate the BNSP AUC     #######################
########################################################################################
def cal_bnsp_auc(data, subgroup, actual_label, pred_label):
    """This will calculate the BNSP auc"""
    # subset where subgroup is True and target label is 1
    subgroup_positive_examples = data[data[subgroup] & data[actual_label]]
    # subset where subgroup is False and target label is 0
    background_negative_examples = data[~data[subgroup] & ~data[actual_label]]
    # combine above tow sets
    bnsp_examples = subgroup_positive_examples.append(background_negative_examples)
    return cal_auc(bnsp_examples[actual_label], bnsp_examples[pred_label])

########################################################################################
#######################    function to calculate Bias metric     #######################
########################################################################################
def cal_bias_metric(data, subgroups, actual_label, pred_label):
    """Computes per-subgroup metrics for all subgroups and one model
    and returns the dataframe which will have all three Bias metrices
    and number of exmaples for each subgroup"""
    records = []
    for subgroup in subgroups:
        record = {"subgroup": subgroup, "subgroup_size": len(data[data[subgroup]])}
        record["subgroup_auc"] = cal_subgroup_auc(data, subgroup, actual_label, pred_label)
        record["bpsn_auc"]     = cal_bpsn_auc(data, subgroup, actual_label, pred_label)
        record["bnsp_auc"]     = cal_bnsp_auc(data, subgroup, actual_label, pred_label)
        records.append(record)
    submetric_df = pd.DataFrame(records).sort_values("subgroup_auc", ascending = True)
    return submetric_df

########################################################################################
#######################   function to calculate Overall metric   #######################
########################################################################################
def cal_overall_auc(data, actual_label, pred_label):
    return roc_auc_score(data[actual_label], data[pred_label])

########################################################################################
#######################    function to calculate final metric    #######################
########################################################################################
def power_mean(series, p):
    total_sum = np.sum(np.power(series, p))
    return np.power(total_sum/len(series), 1/p)

def final_metric(submetric_df, overall_auc, p = -5, w = 0.25):
    generalized_subgroup_auc = power_mean(submetric_df["subgroup_auc"], p)
    generalized_bpsn_auc = power_mean(submetric_df["bpsn_auc"], p)
    generalized_bnsp_auc = power_mean(submetric_df["bnsp_auc"], p)

    overall_metric = w*overall_auc + w*(generalized_subgroup_auc
                                        + generalized_bpsn_auc
                                        + generalized_bnsp_auc)
    return overall_metric

########################################################################################
#######################   function all above function into one   #######################
########################################################################################
def return_final_metric(data, subgroups,actual_label, pred_label, verbose = False):
    """Data is dataframe which include whole data
    and it also has the predicted target column"""
    submetric_df = cal_bias_metric(data, subgroups, actual_label, pred_label)
    if verbose:
        print("printing the submetric table for each identity or subgroup")
        print(submetric_df)
    overall_auc =  cal_overall_auc(data, actual_label, pred_label)
    overall_metric = final_metric(submetric_df, overall_auc, p = -5, w = 0.25)
    return overall_metric, submetric_df

from sklearn.metrics import confusion_matrix

########################################################################################
#######################    function to plot Confusion matrix     #######################
########################################################################################
def plot_confusion_matrix(train, cv, test):
    tr_pred = np.where(train["pred_target"] >= 0.5, 1, 0)
    cv_pred = np.where(cv["pred_target"] >= 0.5, 1, 0)
    te_pred = np.where(test["pred_target"] >= 0.5, 1, 0)

    tr_con_mat = confusion_matrix(train["target"], tr_pred)
    cv_con_mat = confusion_matrix(cv["target"], cv_pred)
    te_con_mat = confusion_matrix(test["target"], te_pred)

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(19,4))
    sns.heatmap(tr_con_mat, annot=True, fmt="d", annot_kws={"size":15}, ax = ax1)
    ax1.set_title("For Train data", fontsize = 15)
    ax1.set_xlabel("Pridicted target", fontsize = 12)
    ax1.set_ylabel("Actual target", fontsize = 12)

    sns.heatmap(cv_con_mat, annot=True, fmt="d", annot_kws={"size":15}, ax = ax2)
    ax2.set_title("For CV data", fontsize = 15)
    ax2.set_xlabel("Pridicted target", fontsize = 12)
    ax2.set_ylabel("Actual target", fontsize = 12)

    sns.heatmap(te_con_mat, annot=True, fmt="d", annot_kws={"size":15}, ax = ax3)
    ax3.set_title("For Test data", fontsize = 15)
    ax3.set_xlabel("Pridicted target", fontsize = 12)
    ax3.set_ylabel("Actual target", fontsize = 12)

    plt.show()

########################################################################################
##############    function to plot Confusion matrix for each identity   ################
########################################################################################
def plot_confusion_for_each_identity(train, cv, test, subgroups):
    for subgroup in subgroups:
        print("{}{} for '{}' identity {}".format(" "*40, "*"*15, subgroup, "*"*15))
        TR, CV, TE = train[train[subgroup]], cv[cv[subgroup]], test[test[subgroup]]
        plot_confusion_matrix(TR, CV, TE)
        print("\n\n")
