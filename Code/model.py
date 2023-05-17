import keras
from keras.utils.np_utils import *
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding, concatenate
from keras.layers import Dense, Input, Flatten, average,Lambda

from keras.layers import *
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers #keras2
from keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from keras.optimizers import *


def get_model1(lr,N,NUM_CHANNEL,NUM_CLASS):
    
    image_input = Input(shape=(N,N,NUM_CHANNEL),)

    image_rep = Conv2D(128,(2,2),padding='same')(image_input)
    image_rep = Activation('relu')(image_rep)
    #image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep0 = image_rep
    
    for i in range(1):
        image_rep = Conv2D(128,(2,2),padding='same')(image_rep0)
        image_rep = Activation('relu')(image_rep)
        image_rep = Dropout(0.2)(image_rep)
        image_rep0 = Lambda(lambda x:x[0]+x[1])([image_rep,image_rep0])
        
    image_rep0 = MaxPool2D((2,2))(image_rep0)
    
    for i in range(1):
        image_rep = Conv2D(128,(2,2),padding='same')(image_rep0)
        image_rep = Activation('relu')(image_rep)
        image_rep = Dropout(0.2)(image_rep)
        image_rep0 = Lambda(lambda x:x[0]+x[1])([image_rep,image_rep0])

    image_rep0 = MaxPool2D((2,2))(image_rep0)

    image_rep = Flatten()(image_rep0)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
#     image_rep = Dense(512,activation='relu')(image_rep)
#     image_rep = Dropout(0.2)(image_rep)
#     image_rep = Dense(512,activation='relu')(image_rep)
#     image_rep = Dropout(0.2)(image_rep)
    logit = Dense(NUM_CLASS,activation='softmax')(image_rep)
    
    model = Model(image_input,logit)
    
    model.compile(loss=['categorical_crossentropy'],
                      optimizer= SGD(lr=lr),
                      metrics=['acc'])

    return model


def get_model2(lr,N,NUM_CHANNEL,NUM_CLASS):
    
    image_input = Input(shape=(28,28,NUM_CHANNEL),)

    image_rep = Conv2D(32,(5,5),)(image_input)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Conv2D(128,(5,5),)(image_rep)
    image_rep = MaxPool2D((2,2))(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    
    image_rep = Flatten()(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    image_rep = Dense(512,activation='relu')(image_rep)
    image_rep = Dropout(0.2)(image_rep)
    logit = Dense(NUM_CLASS,activation='softmax')(image_rep)
    
    model = Model(image_input,logit)
    
    model.compile(loss=['categorical_crossentropy'],
                      optimizer= SGD(lr=lr),
                      metrics=['acc'])

    return model

def get_model(dataset,lr,N,NUM_CHANNEL,NUM_CLASS):
    if dataset == 'CIFAR10':
        return get_model1(lr,N,NUM_CHANNEL,NUM_CLASS)
    else:
        return get_model2(lr,N,NUM_CHANNEL,NUM_CLASS)
