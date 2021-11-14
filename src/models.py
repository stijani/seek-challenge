from tensorflow.keras import models
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import tensorflow as tf


def callbacks_(patience=2, monitor='val_loss'):
        return [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)]
    

class MLP:
    

    def __init__(self, layers, units, dropout_rate, input_shape, num_classes=3, op_activation='softmax'):
        self.layers = layers
        self.units = units
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.op_activation = op_activation
        
        
    def get_compiled_model(self, opt, metrics, loss_fn):
        model = models.Sequential()
        model.add(Dropout(rate=self.dropout_rate, input_shape=self.input_shape))

        for _ in range(self.layers-1):
            model.add(Dense(units=self.units, activation='relu'))
            model.add(Dropout(rate=self.dropout_rate))

        model.add(Dense(units=self.num_classes, activation=self.op_activation))
        model.compile(optimizer=opt, loss=loss_fn, metrics=metrics)
        return model
    
    
    def train(self, compiled_model, x_train, x_val, y_train, y_val, callbacks, bs, epochs):
        history = compiled_model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            callbacks = callbacks,
                            validation_data = (x_val, y_val),
                            verbose = 2,
                            batch_size = bs)
        
        # printout metrics and loss
        history = history.history
        print('FINAL VAL ACC: {acc}, FINAL VAL LOSS: {loss}'.format(
                acc=history['val_accuracy'][-1], loss=history['val_loss'][-1]))

        # Save model.
        compiled_model.save('../save_models/mlp_model.h5')
        return history
                            
                                
        
        