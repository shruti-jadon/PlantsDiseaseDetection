#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:36:41 2018

@author: shrutijadon
"""

#cleaning file
import numpy as np
import os
import shutil
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications import xception
from keras.applications import inception_v3
from keras.applications.vgg16 import preprocess_input
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

INPUT_SIZE=224
POOLING='avg'

class DataCleaning:
    def __init__(self):
        print ("Initiated")
        
    def read_img(self,address, size):
        img = image.load_img(address, target_size=size)
        img = image.img_to_array(img)
        return img
    
    def moving_files(self):
        directory_lists=os.listdir("./data/")
        global_path="./data/"+str(directory_lists[1])+"/"
        print directory_lists
        for l in directory_lists[2:]:
            categories=os.listdir("./data/"+str(l)+"/")
            path_current="./data/"+str(l)+"/"
            print categories
            if ('.DS_Store' in categories):
                categories.remove('.DS_Store')
            for c in categories:
                files=os.listdir(path_current+c+"/")
                for f in files:
                    shutil.move(path_current+c+"/"+f, global_path+c+"/")

    
    def extract_features(self,path):
        directory_lists=os.listdir(path)
        X=[]
        Y=[]
        count=0
        if ('.DS_Store' in directory_lists):
                directory_lists.remove('.DS_Store')
        for d in directory_lists:
            nest=os.listdir(path+"/"+d)
            if ('.DS_Store' in nest):
                nest.remove('.DS_Store')
            for f in nest:
                img = image.load_img(path+"/"+d+"/"+f, target_size=(224, 224))
                img_data = image.img_to_array(img)
                img_data = preprocess_input(img_data)
                X.append(img_data)
                Y.append(count)
            count+=1
        print count
        X=np.array(X)
        y=np.array(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        return X_train, X_test, y_train, y_test 
    
    def forward_pass_Resnt(self):
        X_train, X_test, y_train, y_test = self.extract_features("./Data/")
        
        vgg_bottleneck = VGG16(weights='imagenet', include_top=False, pooling=POOLING)
        train_vgg_bf = vgg_bottleneck.predict(X_train, batch_size=32, verbose=1)
        valid_vgg_bf = vgg_bottleneck.predict(X_test, batch_size=32, verbose=1)
        
        xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING)
        train_x_bf = xception_bottleneck.predict(X_train, batch_size=32, verbose=1)
        valid_x_bf = xception_bottleneck.predict(X_test, batch_size=32, verbose=1)
        
        X = np.hstack([train_x_bf, train_vgg_bf])
        V = np.hstack([valid_x_bf, valid_vgg_bf])
        
        #X=train_vgg_bf
        #V=valid_vgg_bf
        
        logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=147)
        logreg.fit(X, y_train)
        valid_probs = logreg.predict_proba(V)
        valid_preds = logreg.predict(V)
        print valid_probs
        print valid_preds
        print log_loss(y_test, valid_probs)
        print accuracy_score(y_test, valid_preds)
        
        
f=DataCleaning()
f.forward_pass_Resnt()
        