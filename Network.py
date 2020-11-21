from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

import numpy as np


class MyNetwork:
    def __init__(self, name, data, problem_type, load_from_file=False):
        self.name = name
        self.problem_type = problem_type
        self._unpack_data(data)

        if problem_type == 'classification':
            self.output_activation = "softmax"
            self.n_output_neurons = len(np.unique(self.y_train))
            self.loss = "sparse_categorical_crossentropy"
        else:
            self.output_activation = 'linear'
            self.n_output_neurons = 1
            self.loss = 'mse'

        if load_from_file:
            self.load_model()
        else:
            self.n_hidden_layers = 1
            self.n_hidden_neurons = 10
            self._build_and_compile_model()

    def find_best_model(self, n_hidden_layers, n_hidden_neurons, save=False):
        best_loss = np.inf
        best_hidden_layers = 0
        best_hidden_neurons = 0

        X_split_list = np.split(self.X_train, [int(x) for x in len(self.X_train) / 10 * np.arange(1, 10)])
        y_split_list = np.split(self.y_train, [int(y) for y in len(self.y_train) / 10 * np.arange(1, 10)])

        for layers in n_hidden_layers:
            for neurons in n_hidden_neurons:
                self.n_hidden_layers = layers
                self.n_hidden_neurons = neurons

                losses = []
                for i in range(10):
                    print(f'Layers: {layers}, Neurons: {neurons}, fold: {i + 1}')

                    X_train_tmp, y_train_tmp, X_test_tmp, y_test_tmp = self._cv_split(X_split_list, y_split_list, i)
                    _ = self.train(X_train_tmp, y_train_tmp, build_new=True)

                    losses.append(self.model.evaluate(X_train_tmp, y_train_tmp, verbose=0))

                    # if self.problem_type == 'classification':
                    #     losses.append(log_loss(y_test_tmp, self.predict(X_test_tmp)))
                    # else:
                    #     losses.append(mean_squared_error(y_test_tmp, self.predict(X_test_tmp)))

                if np.mean(losses) < best_loss:
                    print(f'Layers: {layers}, Neurons: {neurons}, Loss: {np.mean(losses)}')
                    best_hidden_layers = layers
                    best_hidden_neurons = neurons
                    best_loss = np.mean(losses)

        print(f'Best parameters - Layers: {best_hidden_layers}, Neurons: {best_hidden_neurons}')
        if save:
            np.save(f'model/{self.name}.txt', [best_hidden_layers, best_hidden_neurons])

        return layers, neurons

    def train(self, X, y, build_new=True):
        if build_new:
            self._build_and_compile_model()

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)

        early_stopping_cb = EarlyStopping(patience=25, restore_best_weights=True)
        r = self.model.fit(X_train, y_train,
                           validation_data=(X_val, y_val),
                           epochs=300, verbose=False, callbacks=[early_stopping_cb])
        print("finished training model")
        return r

    def predict(self, X):
        return self.model.predict(X)

    def load_model(self):
        params = np.loadtxt(f'model/{self.name}.txt')
        self.n_hidden_layers = params[0]
        self.n_hidden_neurons = params[1]
        self._build_and_compile_model()

    def _build_and_compile_model(self):
        i = Input(shape=self.X_train[0].shape)
        x = Dense(self.n_hidden_neurons, activation="relu")(i)
        for _ in range(self.n_hidden_layers - 1):
            x = Dense(self.n_hidden_neurons, activation="relu")(x)
        o = Dense(self.n_output_neurons, activation=self.output_activation)(x)
        model = Model(i, o)
        self.model = model
        self.model.compile(optimizer="adam", loss=self.loss)

    def _unpack_data(self, data):
        self.X_train = data[0]
        self.y_train = data[1]
        self.X_test = data[2]
        self.y_test = data[3]

    def _cv_split(self, X_split_list, y_split_list, n):
        X_train = np.vstack([X_split_list[i] for i in range(10) if i != n])
        y_train = np.vstack([y_split_list[i].reshape(-1, 1) for i in range(10) if i != n])
        X_test = X_split_list[n]
        y_test = y_split_list[n].reshape(-1, 1)

        return X_train, y_train, X_test, y_test
