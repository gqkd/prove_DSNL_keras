#%%
from model_cnn import net
from data_prep import data_loader
import tensorflow as tf
import keras
import datetime

X_train, X_valid, y_train, y_valid = data_loader()

# %%
model = net()
opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['acc'])

# keras.utils.plot_model(model, show_shapes=True)

log_dir = "/logs/fit_base/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                             histogram_freq=10,
                                             write_graph=True,
                                            write_grads=False,
                                             write_images=False,
                                             embeddings_freq=0,
                                            embeddings_layer_names=None,
                                             embeddings_metadata=None)
# %%
history = model.fit(X_train, y_train,
              epochs=25,
              batch_size=32,
              verbose=1,
              validation_data=(X_valid,y_valid),
              callbacks=[tensorboard])