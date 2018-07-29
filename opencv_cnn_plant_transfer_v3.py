#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:13:18 2018

@author: mohit

"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm #to show progress
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, f1_score



#Define Image Directory
image_dir_test = 'images/plants/test/'
image_dir_train = 'images/plants/train/'

#define the senstivity to control the selection of green color
sensitivity = 30
#define final image size for processing
image_size = 64
'''
define a function to remove background from the image to only leave the green leaves, blurring and normalizing. Followed by resizing the images
to 64 x 64 size

'''

def image_transformation(imageName, sensitivity):
    
    imagePlatCV = cv2.imread(imageName) #read image
    hsvImage = cv2.cvtColor(imagePlatCV, cv2.COLOR_BGR2HSV)
    #define the range for green color
    lower_green = np.array([60 - sensitivity, 100, 50])
    upper_green = np.array([60 + sensitivity, 255, 255])
    # threshold the hsv image to get only green colors
    mask = cv2.inRange(hsvImage, lower_green, upper_green)
    #apply bitwise_and between mask and the original image
    greenOnlyImage = cv2.bitwise_and(imagePlatCV, imagePlatCV, mask=mask)
    #lets define a kernal with ones
    kernel0 = np.ones((15,15), np.uint8)
    #lets apply closing morphological operation
    closing0 = cv2.morphologyEx(greenOnlyImage, cv2.MORPH_CLOSE, kernel0)
    #blur the edges
    blurImage = cv2.GaussianBlur(closing0, (15,15), 0)
    blurImageColor = cv2.cvtColor(blurImage, cv2.COLOR_BGR2RGB)#to make it work with right color
    #resize image
    resizeImage = cv2.resize(blurImageColor, (image_size, image_size), interpolation=cv2.INTER_AREA)
    resizeImage = resizeImage/255 #normalize
    #resizeImage = resizeImage.reshape(image_size,image_size,3) #to make it in right dimensions for the Keras add 3 channel
    print(resizeImage.shape)
    return resizeImage

#define list of plant species
classes = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed'
           , 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
'''
Data extraction: The loop below will create a data list containing image file path name, the classifcation lable 
(0 -11) and the specific plant name

'''
train = [] #data list
for species_lable, speciesName in enumerate(classes):
    for fileName in os.listdir(os.path.join(image_dir_train, speciesName)):
        train.append([image_dir_train + '{}/{}'.format(speciesName, fileName), species_lable, speciesName])
        
        
#convert the list into dataframe using Pandas
trainigDataFrame = pd.DataFrame(train, columns=['FilePath', 'PlantLabel', 'PlantName'])

#Suffle the data
seed = 1234 #define seed to get consistent results
trainigDataFrame = trainigDataFrame.sample(frac=1, random_state=seed)
trainigDataFrame = trainigDataFrame.reset_index()

#Prepare the images for the model by preprocessing

X = np.zeros((trainigDataFrame.shape[0], image_size, image_size, 3)) #array to store image after image_transformfunction

for i, fileName in tqdm(enumerate(trainigDataFrame['FilePath'])):
    print(fileName)
    newImage = image_transformation(fileName, sensitivity)
    X[i] = newImage

#Convert lables to categorical and do one-hot encoding, followed by conversion to numpy array
y = trainigDataFrame['PlantLabel']
y = pd.Categorical(y)
y = pd.get_dummies(y)
y = y.values

'''
Build the initial network from pretrained model that will be used for transfer learning. We will use first two blocks from 
VGG19 pretrained model
'''
#input layer
imageInput = Input(shape=(image_size, image_size, 3), name='input_1')
#Block 1 - layers name same as the layers in pretrained model
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(imageInput)
model = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(model)
model = MaxPool2D(pool_size= (2,2), strides=(2,2), name='block1_pool')(model)

#Block 2

model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(model)
model = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(model)
model = MaxPool2D(pool_size= (2,2), strides=(2,2), name='block2_pool')(model)
model_initial = Model(inputs=imageInput, outputs=model)
model_initial.summary()

#define the dictionary of layers
layer_dict = dict([(layer.name, layer) for layer in model_initial.layers])
print(layer_dict)

#load weights from VGG19 model
weights_path = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'#from https://github.com/fchollet/deep-learning-models/releases/
model_initial.load_weights(weights_path, by_name=True)

#set weights for initial layers
for i, layer in enumerate(model_initial.layers):
    weights = layer.get_weights()
    model_initial.layers[i].set_weights(weights)


model_final = Model(inputs = model_initial.input, outputs = model_initial.output)

features = [] #List to extract features using pretrained model weights
for i in tqdm(X):
    image = np.expand_dims(i, axis=0)
    featurePredict = model_final.predict(image)
    features.append(featurePredict)
    
#convert to numpy array from list    
featuresArray = np.array(features)

#reshape to get in the right shape for keras model
featuresReshape = np.reshape(featuresArray, (featuresArray.shape[0], featuresArray.shape[2], featuresArray.shape[3], featuresArray.shape[4]))
    
print(featuresReshape[1].shape)

    
#split dataset into Train and Test
X_train_dev, X_val, y_train_dev, y_val = train_test_split(featuresReshape, y, test_size=0.10, random_state=seed)
X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size= 0.10, random_state=seed)


#define a classifier function with extracted features as input
def cnn_model():
    classifier = Sequential()
    classifier.add(Conv2D(256, kernel_size=(3,3), padding='same', strides=(1,1), input_shape=(features[1].shape[1], features[1].shape[2],features[1].shape[3]), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same',  activation='relu'))  
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Conv2D(1024, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2,2)))
    classifier.add(Dropout(0.1))
    classifier.add(Flatten())
    classifier.add(Dense(units=1024, activation='relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(units=512, activation='relu'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=12, activation='softmax'))
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    classifier.summary()    
    return classifier

#define classifier
classifier = cnn_model()

batch_size = 64
epochs = 50
checkpoint = ModelCheckpoint('model_vgg19_final.h5', verbose=1, save_best_only=True)

#train model
trainingModel = classifier.fit(x=X_train, y=y_train, batch_size=batch_size, 
                           epochs=epochs, verbose=1, callbacks=[checkpoint], validation_data=(X_dev, y_dev))

#final model
final_model = load_model("model_vgg19_final.h5")

final_loss, final_accuracy = final_model.evaluate(X_val, y_val)
print('Final Loss: {}, Final Accuracy: {}'.format(final_loss, final_accuracy))

# get prediction

y_pred = final_model.predict(X_val)

#confusion matrix and classification report

cm = confusion_matrix(y_val.argmax(axis=1), y_pred.argmax(axis=1))

print(classification_report(y_val.argmax(axis=1), y_pred.argmax(axis=1), target_names=classes))

#plot F1-score vs. Classes

f1Score = f1_score(y_val.argmax(axis=1), y_pred.argmax(axis=1), average=None)

y_pos = np.arange(len(classes))
plt.bar(y_pos, f1Score)
plt.xticks(y_pos, classes)
plt.ylabel('F1 Score')
plt.title('F1 Score of various species after classification')
plt.show()


