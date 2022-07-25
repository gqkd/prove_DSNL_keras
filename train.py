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
            patiente=100,
            weights=False,
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
                                                  patience=patiente,
                                                  verbose=1,
                                                  mode='min')    

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M"),
                                              monitor='val_accuracy',
                                              mode='max',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=False)
    if weights:
      print("inside if of weights")
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

# if __name__ == "__main__":
