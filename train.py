from model_cnn import net
# from data_prep import data_loader
import tensorflow as tf
import shutil, os
import datetime


def trainer(X_train, X_valid, y_train, y_valid, 
            epochs=10, batch=32, log_dir="/logs/base_fit/" ):


    model = net()
    opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy',
                            'AUC',
                           'Precision',
                           'Recall'])

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

    history = model.fit(X_train, y_train,
                epochs=epochs,
                batch_size=batch,
                verbose=1,
                validation_data=(X_valid,y_valid),
                callbacks=[tensorboard])
    
    model.save( log_dir +'models/'+ datetime.datetime.now().strftime("%Y_%m_%d-%H%M"))

    return history

