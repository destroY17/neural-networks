import numpy as np

from scipy.stats.qmc import Sobol
from scipy.interpolate import make_interp_spline

import tensorflow as tf
import tensorflow.keras.layers as layers

import matplotlib.pyplot as plt

X_DIMENSION = 20
ENCODING_DIMENSION = 5
LAYER_DIMENSION = 500
LAYER_COUNT = 3

POINTS_COUNT = 1024
POINTS_COUNT_FOR_TEST = 1024

ITERATIONS_COUNT = 50


class AutoEncoder(tf.keras.Model):
    def __init__(self, input_dimension, encoding_dimension, layer_dimension, layer_count):
        super().__init__()

        self.layer_count = layer_count

        self.encoder = tf.keras.Sequential()
        for _ in range(layer_count):
            self.encoder.add(layers.Dense(layer_dimension, activation='relu'))
        self.encoder.add(layers.Dense(encoding_dimension))

        self.decoder = tf.keras.Sequential()
        for _ in range(layer_count):
            self.decoder.add(layers.Dense(layer_dimension, activation='relu'))
        self.decoder.add(layers.Dense(input_dimension))

    def call(self, x):
        latent_x = self.encoder(x)
        decoded_x = self.decoder(latent_x)
        return decoded_x


def rastrigin(x: np.ndarray) -> float:
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def generate_points(dimension: int, count: int) -> np.ndarray:
    return Sobol(d=dimension).random(n=count)


def get_y_values(x_values: np.ndarray) -> np.ndarray:
    return np.array([rastrigin(x) for x in x_values])


def create_test_data(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    return np.hstack((x_values, y_values.reshape(-1, 1)))


def train_autoencoder(autoencoder: AutoEncoder) -> None:
    x_values = generate_points(X_DIMENSION, POINTS_COUNT_FOR_TEST)
    y_values = get_y_values(x_values)
    train_data = create_test_data(x_values, y_values)

    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(train_data,
                    train_data,
                    validation_split=0.2,
                    epochs=100,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10)
                    ])


def find_optimum(x_values: np.ndarray, y_values: np.ndarray) -> tuple:
    min_index = np.argmin(y_values)
    return x_values[min_index], np.min(y_values)


def print_solution(title: str, vector: np.ndarray, optimum: float) -> None:
    print(f"--- {title} ---")
    print("Решение:")
    print(vector)
    print("Оптимум:")
    print(optimum)
    print()


def calculate_exhaustive_original(iterations_count: int):
    average_original = 0
    x = []
    y = []

    for i in range(iterations_count):
        example_x_values = generate_points(X_DIMENSION, POINTS_COUNT)
        example_y_values = get_y_values(example_x_values)
        example_vector, example_optimum = find_optimum(example_x_values, example_y_values)

        average_original += example_optimum
        x.append(i + 1)
        y.append(example_optimum)

        print("Наилучшее решение, полученное перебором точек в пространстве решений оригинальной функции:")
        print("Xo = " + str(example_vector))
        print("f(Xo) = " + str(example_optimum))

    return x, y


def calculate_exhaustive_autoencoder(autoencoder: AutoEncoder, iterations_count: int):
    average_decoder = 0
    x = []
    y = []

    for i in range(iterations_count):
        example_autoencoder_x_values = generate_points(ENCODING_DIMENSION, POINTS_COUNT)
        decoder_function = autoencoder.decoder
        decoder_all_values = decoder_function(example_autoencoder_x_values)
        decoder_x_values = decoder_all_values[:, :-1]
        decoder_y_values = get_y_values(decoder_x_values)
        decoder_vector, decoder_optimum = find_optimum(decoder_x_values, decoder_y_values)

        average_decoder += decoder_optimum
        x.append(i + 1)
        y.append(decoder_optimum)

        print("Наилучшее решение, полученное перебором точек в редуцированном пространстве:")
        print("Xa = " + str(np.array(decoder_vector)))
        print("f(Xa) = " + str(decoder_optimum))

    return x, y


def interp_spline(x, y):
    spline = make_interp_spline(np.array(x), np.array(y))

    x_interp = np.linspace(np.array(x).min(), np.array(x).max(), 500)
    x_interp = x
    y_interp = spline(x_interp)

    return x_interp, y_interp


def test_exhaustive_autoencoder(layer_count):
    autoencoder = AutoEncoder(input_dimension=X_DIMENSION + 1,
                              encoding_dimension=ENCODING_DIMENSION,
                              layer_dimension=LAYER_DIMENSION,
                              layer_count=layer_count)
    train_autoencoder(autoencoder)
    (x, y) = calculate_exhaustive_autoencoder(autoencoder, ITERATIONS_COUNT)
    return interp_spline(x, y)


def test_exhaustive_original():
    (x, y) = calculate_exhaustive_original(ITERATIONS_COUNT)
    return interp_spline(x, y)


def plot_exhaustive_original(x_original, y_original, color):
    plt.semilogy(x_original, y_original, ':o', color=f'{color}', label='Оригинал')


def plot_exhaustive_autoencoder(x_autoencoder, y_autoencoder, color):
    plt.semilogy(x_autoencoder, y_autoencoder, ':o', color=f'{color}', label='Автоэнкодер')


def create_plot_for_exhaustive():
    plt.figure(figsize=(20, 6))
    plt.xlabel("Номер эксперимента")
    plt.ylabel("Значение оптимума")


def test_exhaustive_search():
    (x_original, y_original) = test_exhaustive_original()
    (x_autoencoder, y_autoencoder) = test_exhaustive_autoencoder(LAYER_COUNT)

    create_plot_for_exhaustive()
    plot_exhaustive_original(x_original, y_original, color='blue')
    plot_exhaustive_autoencoder(x_autoencoder, y_autoencoder, color='red', layer_count=LAYER_COUNT)

    plt.legend()
    plt.show()


def main():
    test_exhaustive_search()


if __name__ == "__main__":
    main()
