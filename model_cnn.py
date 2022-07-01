import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation
from keras.layers import Reshape, BatchNormalization



def net():
  activation = tf.nn.relu
  # activation = tf.nn.leaky_relu
  padding = 'same'

  ######### Input ########
  input_signal = Input(shape=(1,9000), name='input_signal')
  # print("input_signal:",input_signal.shape)

  ######### CNNs with small filter size at the first layer #########

  cnn0 = Conv1D(
    kernel_size=50,
    filters=64,
    strides=6,
    kernel_regularizer=keras.regularizers.l2(0.001)) 
  s = cnn0(input_signal)
  s = BatchNormalization()(s) 
  s = Activation(activation=activation)(s)

  cnn1 = MaxPool1D(pool_size=8, strides=8)
  s = cnn1(s)

  cnn2 = Dropout(0.5)
  s = cnn2(s)

  cnn3 = Conv1D(kernel_size=8,filters=128,strides=1,padding=padding)
  s = cnn3(s)
  s = BatchNormalization()(s)
  s = Activation(activation=activation)(s)

  cnn4 = Conv1D(kernel_size=8,filters=128,strides=1,padding=padding)
  s = cnn4(s)
  s = BatchNormalization()(s)
  s = Activation(activation=activation)(s)

  cnn5 = Conv1D(kernel_size=8,filters=128,strides=1,padding=padding)
  s = cnn5(s)
  s = BatchNormalization()(s)
  s = Activation(activation=activation)(s)

  cnn6 = MaxPool1D(pool_size=4,strides=4)
  s = cnn6(s)

  cnn7 = Reshape((int(s.shape[1])*int(s.shape[2]),)) # Flatten
  s = cnn7(s)


  ######### CNNs with large filter size at the first layer #########

  cnn8 = Conv1D(
    kernel_size=400,
    filters=64,strides=50,kernel_regularizer=keras.regularizers.l2(0.001))
  l = cnn8(input_signal)
  l = BatchNormalization()(l)
  l = Activation(activation=activation)(l)

  cnn9 = MaxPool1D(pool_size=4, strides=4)
  l = cnn9(l)

  cnn10 = Dropout(0.5)
  l = cnn10(l)

  cnn11 = Conv1D(kernel_size=6,filters=128,strides=1,padding=padding)
  l = cnn11(l)
  l = BatchNormalization()(l)
  l = Activation(activation=activation)(l)

  cnn12 = Conv1D(kernel_size=6,filters=128,strides=1,padding=padding)
  l = cnn12(l)
  l = BatchNormalization()(l)
  l = Activation(activation=activation)(l)

  cnn13 = Conv1D(kernel_size=6,filters=128,strides=1,padding=padding)
  l = cnn13(l)
  l = BatchNormalization()(l)
  l = Activation(activation=activation)(l)

  cnn14 = MaxPool1D(pool_size=2,strides=2)
  l = cnn14(l)

  cnn15 = Reshape((int(l.shape[1])*int(l.shape[2]),))
  l = cnn15(l)


  merged = keras.layers.concatenate([s, l])

  merged = Dense(1024)(merged) 
  merged = Dropout(0.5)(merged) 
  merged = Dense(5,name='merged')(merged)

  softmax = Activation(activation='softmax')(merged)

  model = Model(input_signal,softmax)

  return model

# if __name__ == '__main__':
#   model = net()