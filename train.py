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
            weights=False,
            decay = 0.0001,
            sched = False
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
                    metrics=['accuracy'
                          #   'AUC',
                          #  'Precision',
                          #  'Recall',
                          #  tfa.metrics.F1Score(num_classes=num_classes,average='macro'),
                          #  tfa.metrics.CohenKappa(num_classes=num_classes),
                          #  tfa.metrics.FBetaScore(num_classes=num_classes,average='macro')
                           ])

    tf.keras.utils.plot_model(model, show_shapes=True)

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
    
    #if .h5 string is deleted at the end it will save in .model format                                                  
    save_weights_at = log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M")+'.h5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights_at,
                                              monitor='val_accuracy',
                                              mode='max',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=False)

    def schedule(epoch, lr):
        decay = 0.00009
        if epoch < 10:
            return lr
        elif epoch > 70:
          return lr
        else:
            return lr * tf.math.exp(-0.1)

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)

    if weights and sched:
      print("Using weights and scheduler")
      history = model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch,
                  verbose=1,
                  sample_weight = train_weights, 
                  validation_data=(X_valid,y_valid),
                  callbacks=[tensorboard, early_stopping, checkpoint,scheduler])
    elif weights and not sched:
      print("Using weights")
      history = model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch,
                  verbose=1,
                  sample_weight = train_weights, 
                  validation_data=(X_valid,y_valid),
                  callbacks=[tensorboard, early_stopping, checkpoint])
    elif not weights and sched:
      print("Using scheduler")
      history = model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch,
                  verbose=1,
                  validation_data=(X_valid,y_valid),
                  callbacks=[tensorboard, early_stopping, checkpoint,scheduler])
    
    elif not weights and not sched:
        history = model.fit(X_train, y_train,
                    epochs=epochs,
                    batch_size=batch,
                    verbose=1,
                    validation_data=(X_valid,y_valid),
                    callbacks=[tensorboard, early_stopping, checkpoint])
        

    # model.save( log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M"))

    return history, model

def trainer_gen(generator_X_train, generator_X_valid,
            k,
            epochs=10, 
            # batch=32, 
            log_dir="/logs/base_fit/", 
            model = None,
            lr = 1e-4,
            patience=100,
            weights=False,
            decay = 0.0001,
            sched = False
            ):

    num_classes = len(generator_X_train.labels)

    if weights:
        y_train_ = list(generator_X_train.labels.values())
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
                    metrics=['accuracy'
                          #   'AUC',
                          #  'Precision',
                          #  'Recall',
                          #  tfa.metrics.F1Score(num_classes=num_classes,average='macro'),
                          #  tfa.metrics.CohenKappa(num_classes=num_classes),
                          #  tfa.metrics.FBetaScore(num_classes=num_classes,average='macro')
                           ])

    tf.keras.utils.plot_model(model, show_shapes=True)

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
    
    #if .h5 string is deleted at the end it will save in .model format                                                  
    save_weights_at = log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M")+'.h5'

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_weights_at,
                                              monitor='val_accuracy',
                                              mode='max',
                                              verbose=1,
                                              save_best_only=True,
                                              save_weights_only=False)

    def schedule(epoch, lr):
        decay = 0.00009
        if epoch < 10:
            return lr
        elif epoch > 70:
          return lr
        else:
            return lr * tf.math.exp(-0.1)

    scheduler = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)

    if weights and sched:
      print("Using weights and scheduler")
      history = model.fit(generator_X_train,
                  epochs=epochs,
                  batch_size=generator_X_train.batch_size,
                  verbose=1,
                  sample_weight = train_weights, 
                  validation_data=(generator_X_valid),
                  callbacks=[tensorboard, early_stopping, checkpoint,scheduler])
    elif weights and not sched:
      print("Using weights")
      history = model.fit(generator_X_train,
                  epochs=epochs,
                  batch_size=generator_X_train.batch_size,
                  verbose=1,
                  sample_weight = train_weights, 
                  validation_data=(generator_X_valid),
                  callbacks=[tensorboard, early_stopping, checkpoint])
    elif not weights and sched:
      print("Using scheduler")
      history = model.fit(generator_X_train,
                  epochs=epochs,
                  batch_size=generator_X_train.batch_size,
                  verbose=1,
                  validation_data=(generator_X_valid),
                  callbacks=[tensorboard, early_stopping, checkpoint,scheduler])
    
    elif not weights and not sched:
        history = model.fit(generator_X_train,
                    epochs=epochs,
                    batch_size=generator_X_train.batch_size,
                    verbose=1,
                    validation_data=(generator_X_valid),
                    callbacks=[tensorboard, early_stopping, checkpoint])
        

    # model.save( log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M"))

    return history, model
# if __name__ == "__main__":
#     pass
    
    