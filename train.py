from model_cnn import net
import tensorflow as tf
import shutil, os
import datetime
import tensorflow_addons as tfa

def trainer(X_train, X_valid, y_train, y_valid, 
            epochs=10, batch=32, log_dir="/logs/base_fit/", model = None):
    num_classes = len(y_train[0])
    if model == None: #if not specified is the first training
        model = net()
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
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

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=0.0001, 
                                                  patience=5,
                                                  verbose=1,
                                                  mode='min')    

    history = model.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch,
                verbose=1,
                validation_data=(X_valid,y_valid),
                callbacks=[tensorboard,early_stopping])
    
    model.save( log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M"))

    return history

