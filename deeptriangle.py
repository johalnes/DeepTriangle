from tensorflow.keras.layers import TimeDistributed, GRU, Dense, RepeatVector, LSTM
from tensorflow.keras.models import Sequential

import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as tfk



def deeptriangle(timesteps, features, names_output=["paid_output","case_reserves_output"]):

    
    tfk.clear_session()

    ay_seq_input  = layers.Input(shape = (timesteps,features), name = 'ay_seq_input' )
    company_code_input = layers.Input(shape = 1, name = "company_input")
    company_code_embedding = layers.Embedding(200,49)(company_code_input)
    company_code_embedding = layers.Flatten()(company_code_embedding)
    company_code_embedding = layers.RepeatVector(timesteps)(company_code_embedding)


    encoded = layers.Masking(mask_value = -99)(ay_seq_input)
    encoded = layers.GRU(128, dropout = 0.2, recurrent_dropout = 0.2)(encoded)

    concat_layer = lambda x: layers.Concatenate()([x, company_code_embedding])
    decoded = layers.RepeatVector(timesteps)(encoded)
    decoded = layers.GRU(128, return_sequences = True, dropout = 0.2, recurrent_dropout = 0.2)(decoded)
    decoded = layers.Lambda(concat_layer)(decoded)

    feature_list = []
    for name in names_output:
        feature_list.append(create_feature_output(name, decoded))
      

    model = keras.Model(
    inputs = [ay_seq_input, company_code_input],
    outputs = feature_list,
        name = "DeepTriangle"
    )

    return model
    
    
def create_feature_output(name, input_layer):

    feature = layers.TimeDistributed(layers.Dense(units = 64, activation = "relu"))(input_layer)
    feature = layers.TimeDistributed(layers.Dropout(rate = 0.2))(feature)
    feature = layers.TimeDistributed(layers.Dense(units = 1, activation = "relu"), name = name)(feature)
    
    return feature
    
    
# masked funktion benyttet af Kuo i Deep Triangle
def masked_mse(missing_value):
    
    def custom_mse(y_true, y_pred):
    # assume 1st dimension is the number of samples
        keep= tfk.cast(tfk.not_equal(y_true, missing_value), tfk.floatx())
        mse = tfk.mean(tfk.square((y_pred-y_true)*keep), axis=2)

        return mse

    return custom_mse