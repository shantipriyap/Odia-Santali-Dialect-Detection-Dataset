######################################################################################################################
# This code train the Supervised Autoencoder for the language detection task.
# The language detection task include detect Odia language(ori) and Santali language written in Odia script (sat) 
# from the given dataset. Please follow the README for more details.
# 
# Author: Shantipriya Parida (IDIAP)
#
#
#######################################################################################################################

from skopt.utils import use_named_args
from supervisedAE import SupervisedAE
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from zipfile import ZipFile
import requests
import numpy as np
import shutil
import os
import json
from io import open
import pickle

##Load dataset
#...............

DATA_DIR = os.path.join(os.getcwd(),"data")
OUTPUT_DIR = os.path.join(os.getcwd(),"output")
MODEL_DIR = os.path.join(os.getcwd(),"model")

if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

train_file = os.path.join(DATA_DIR,"train.txt")
dev_file = os.path.join(DATA_DIR,"dev.txt")
test_file = os.path.join(DATA_DIR,"test.txt")

metric_file_dev = os.path.join(OUTPUT_DIR,"metrics_santali_dev.txt")
metric_file_test = os.path.join(OUTPUT_DIR,"metrics_santali_test.txt")
result_file_dev = os.path.join(OUTPUT_DIR,"result_santali_dev.csv")
result_file_test = os.path.join(OUTPUT_DIR,"result_santali_test.csv")


langdict = {
        "ori":"0",
        "sat":"1"
        }

train_data = open(train_file,encoding="utf-8").read()
dev_data = open(dev_file,encoding="utf-8").read()
test_data = open(test_file,encoding="utf-8").read()

#Split the train and test into lines
train_data = train_data.splitlines()
dev_data = dev_data.splitlines()
test_data = test_data.splitlines()

#Create arrays to store the integer version of the data
train_sentences = []
train_classes = []

dev_sentences = []
dev_classes = []

test_sentences = []
test_classes = []

for _, line in enumerate(train_data):
    sen, lang_class = line.split("\t")
    val = langdict[lang_class]
    train_sentences.append(sen)
    train_classes.append(int(val))
for _, line in enumerate(dev_data):
    sen, lang_class = line.split("\t")
    val = langdict[lang_class]
    dev_sentences.append(sen)
    dev_classes.append(int(val))

for _, line in enumerate(test_data):
    sen, lang_class = line.split("\t")
    val = langdict[lang_class]
    test_sentences.append(sen)
    test_classes.append(int(val))

cf = CountVectorizer(analyzer='char_wb', ngram_range=(1,3), min_df=0.001, max_df=0.3)
#cf = CountVectorizer(analyzer='char_wb', ngram_range=(1,5), min_df=0.001, max_df=0.3)
train_counts = cf.fit_transform(train_sentences)
dev_counts = cf.transform(dev_sentences)
test_counts = cf.transform(test_sentences)

tfidf_transformer = TfidfTransformer()
train_tf = tfidf_transformer.fit_transform(train_counts)
dev_tf = tfidf_transformer.transform(dev_counts)
test_tf = tfidf_transformer.transform(test_counts)

X_train = train_tf.toarray()
X_dev = dev_tf.toarray()
X_test = test_tf.toarray()
y_train = np.asarray(train_classes)
y_dev = np.asarray(dev_classes)
y_test = np.asarray(test_classes)


#path to save model file
path = os.path.join(MODEL_DIR,"model_santali.pt")
print("model path:",path)

#Input parameter for Autoencoders
input_dim = X_train.shape[1]
emb_dim = 300
num_targets = 2
num_chunks = 1
supervision = 'clf'
convg_thres = 1e-5
#max_epochs = 500
max_epochs = 30
#is_gpu = True
is_gpu = False
gpu_ids = 0

model = SupervisedAE(input_dim=input_dim, emb_dim=emb_dim, num_targets=num_targets, \
                    num_chunks=num_chunks, supervision=supervision, convg_thres=convg_thres, \
                    max_epochs=max_epochs, is_gpu=is_gpu, gpu_ids=gpu_ids)

#Hyperparameter search space
search_space = [Integer(1, 5, name='num_layers'),
                Real(10**-5, 10**-2, "log-uniform", name='learning_rate'),
                Real(10**-6, 10**-3, "log-uniform", name='weight_decay'),
                Categorical(['relu', 'sigma'], name='activation')]

@use_named_args(search_space)
def optimize(**params):
    print("Training with hyper-parameters: ", params)
    model.set_params(params)
    model.fit(X_train, y_train)
    return model.loss

# define the space of hyperparameters to search
result = gp_minimize(optimize, search_space, n_calls=10)
# summarizing finding:
print('best score: {}'.format(result.fun))
print('best params:')
print('num_layers: {}'.format(result.x[0]))
print('learning_rate: {}'.format(result.x[1]))
print('weight_decay: {}'.format(result.x[2]))
print('activation: {}'.format(result.x[3]))

num_layers = result.x[0]
learning_rate = result.x[1]
weight_decay = result.x[2]
activation = result.x[3]
model = SupervisedAE(input_dim=input_dim, emb_dim=emb_dim, num_targets=num_targets, \
                    num_layers=num_layers, learning_rate=learning_rate, weight_decay=weight_decay,
                    num_chunks=num_chunks, supervision=supervision, convg_thres=convg_thres, \
                    activation=activation, max_epochs=max_epochs, is_gpu=is_gpu, gpu_ids=gpu_ids)

model.fit(X_train, y_train)
print('saving model here:',path)
model.save(path)

########################################################
# Predict Dev set
########################################################
predicted,_ = model.predict(X_dev)
prob, predicted = torch.max(predicted, 1)

cf = metrics.confusion_matrix(y_dev, predicted.numpy())
print("confusion matrix for dev set:")
print(cf)

##Change according to the appropriate metrics function
print("Writing the metrics into dev set file....")
f = open(metric_file_dev, 'w+')

#########################################################
# Print devset result
#########################################################
f.write("Devset Matrics")
f.write("\n")
f.write(metrics.classification_report(y_dev, predicted.cpu().numpy()))
f.write(str(cf))
f.close()

#########################################################
# Print devset output
#########################################################
df = pd.DataFrame(data={'dev set':dev_sentences, 'pred':predicted.numpy(), 'prob':prob.numpy()})
df.to_csv(result_file_dev, index=False)

print("Writing the metrics into test set file....")
f = open(metric_file_test, 'w+')

########################################################
# Predict Test set
########################################################

predicted1,_ = model.predict(X_test)
prob1, predicted1 = torch.max(predicted1, 1)

cf1 = metrics.confusion_matrix(y_test, predicted1.numpy())
print("confusion matrix for test set:")
print(cf1)

#########################################################
# Print testset result
#########################################################
f.write("Testset Matrics")
f.write("\n")
f.write(metrics.classification_report(y_test, predicted1.cpu().numpy()))
f.write(str(cf1))
f.close()

print("Writing output/result into file....")
#########################################################
# Print testset output
#########################################################
df1 = pd.DataFrame(data={'test set':test_sentences, 'pred':predicted1.numpy(), 'prob':prob1.numpy()})
df1.to_csv(result_file_test, index=False)

print("done....")
