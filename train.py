from model_cnn import net
import tensorflow as tf
import datetime
import tensorflow_addons as tfa
import numpy as np

def trainer(X_train, X_valid, y_train, y_valid, 
            epochs=10, 
            batch=32, 
            log_dir="/logs/base_fit/", 
            model = None,
            lr = 1e-4,
            patience=100,
            weights=False
            ):

    num_classes = len(y_train[0])
    
    if weights:
        y_train_ = y_train.argmax(axis=1)
        _ , counts = np.unique(y_train_, return_counts=True)
        # under_represented_class = counts.argmin() #the under represented class
        train_weights = y_train_
        #creation of balanced vector for weights
        #divide the max of the counts by every counts to have the "ratio"
        ratio = max(counts)/counts
        #now divide it by the sum to have a vector that sum up to 1
        weights_ = ratio/sum(ratio)
        sum(weights_) #must be 1
        for i in range(len(weights_)):
            train_weights = np.where(train_weights==i, weights_[i], train_weights)


    if model == None: #if not specified is the first training
        model = net()
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy',
                            'AUC',
                           'Precision',
                           'Recall',
                           tfa.metrics.F1Score(num_classes=num_classes,average='macro'),
                           tfa.metrics.CohenKappa(num_classes=num_classes),
                           tfa.metrics.FBetaScore(num_classes=num_classes,average='macro')
                           ])

    # keras.utils.plot_model(model, show_shapes=True)

    log_directory = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_directory,
                                                histogram_freq=10,
                                                write_graph=True,
                                                write_grads=False,
                                                write_images=False,
                                                embeddings_freq=0,
                                                embeddings_layer_names=None,
                                                embeddings_metadata=None)

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  min_delta=0.0001, 
                                                  patience=patience,
                                                  verbose=1,
                                                  mode='max')    

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M"),
                                              monitor='val_accuracy',
                                              mode='max',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=False)
    if weights:
      print("Using weights")
      history = model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch,
                  verbose=1,
                  sample_weight = train_weights, 
                  validation_data=(X_valid,y_valid),
                  callbacks=[tensorboard,early_stopping, checkpoint])
    else:
        history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch,
                    verbose=1,
                    validation_data=(X_valid,y_valid),
                    callbacks=[tensorboard,early_stopping, checkpoint])
        

    # model.save( log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M"))

    return history

if __name__ == "__main__":
    
    from data_prep import data_loader
    from train import trainer
    import tensorflow as tf
    import os
    # confirm TensorFlow sees the GPU
    from tensorflow.python.client import device_lib
    assert 'GPU' in str(device_lib.list_local_devices())
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    X_train, X_valid, y_train, y_valid = data_loader()

    history = trainer(X_train, X_valid, y_train, y_valid,
                  epochs=1,
                  lr=9e-7, #9e-7
                  batch=100,
                  log_dir="/logs/base_fit/",
                  patiente=200,
                  weights=True
                  # model=tf.keras.models.load_model('/content/drive/MyDrive/prove_DSNL_keras/logs/base_fit2/models/2022_07_18-0904') #0.72355 
                  #  model=tf.keras.models.load_model('/content/drive/MyDrive/prove_DSNL_keras/logs/base_fit2/models/2022_07_18-1922') #0.72141
                  )

    
    