import numpy as np
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize
import tensorflow as tf

X_DIMENSION = 20
ENCODING_DIMENSION = 5
LAYER_DIMENSION = 500

POINTS_COUNT = 1024
POINTS_COUNT_FOR_TEST = 1024

ITERATIONS_COUNT = 1000


class AutoEncoder(tf.keras.Model):
    def __init__(self, input_dimension, encoding_dimension, layer_dimension):
        super().__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(encoding_dimension),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_dimension, activation='relu'),
            tf.keras.layers.Dense(input_dimension),
        ])

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


def use_optimize_methods(autoencoder: AutoEncoder):
    def decoder_function_for_optimization(x_value):
        decoder_all_values = decoder_function(np.array(x_value).reshape(-1, ENCODING_DIMENSION))
        decoder_x_values = decoder_all_values[:, :-1]
        decoder_y_values = get_y_values(decoder_x_values)
        return np.array(decoder_y_values).reshape(-1, 1)

    decoder_function = autoencoder.decoder
    average_original = 0
    average_decoder = 0
    for _ in range(ITERATIONS_COUNT):
        example_result = minimize(rastrigin,
                                  generate_points(X_DIMENSION, 1).reshape(-1),
                                  method="nelder-mead")

        decoder_result = minimize(decoder_function_for_optimization,
                                  generate_points(ENCODING_DIMENSION, 1).reshape(-1),
                                  method="nelder-mead")
        decoder_vector = np.array(decoder_result.x).reshape(-1, ENCODING_DIMENSION)

        print_solution("Результат, полученный перебором точек оригинальной функции",
                       example_result.x, example_result.fun)
        print_solution("Результат, полученный перебором точек скрытого пространства",
                       decoder_function(decoder_vector), decoder_result.fun)

        average_original += example_result.fun
        average_decoder += decoder_result.fun

    # print("Средний оптимум методом оптимизаций(оригинал):")
    # print(average_original / ITERATIONS_COUNT)
    #
    # print("Средний оптимум методом оптимизаций(декодер):")
    # print(average_decoder / ITERATIONS_COUNT)


def use_exhaustive_search(autoencoder: AutoEncoder):
    average_original = 0
    average_decoder = 0

    for _ in range(ITERATIONS_COUNT):
        example_x_values = generate_points(X_DIMENSION, POINTS_COUNT)
        example_y_values = get_y_values(example_x_values)
        example_vector, example_optimum = find_optimum(example_x_values, example_y_values)

        example_autoencoder_x_values = generate_points(ENCODING_DIMENSION, POINTS_COUNT)
        decoder_function = autoencoder.decoder
        decoder_all_values = decoder_function(example_autoencoder_x_values)
        decoder_x_values = decoder_all_values[:, :-1]
        decoder_y_values = get_y_values(decoder_x_values)
        decoder_vector, decoder_optimum = find_optimum(decoder_x_values, decoder_y_values)

        # print_solution("Результат, полученный перебором точек оригинальной функции",
        #                example_vector, example_optimum)
        # print_solution("Результат, полученный перебором точек скрытого пространства",
        #                np.array(decoder_vector), decoder_optimum)

        average_original += example_optimum
        average_decoder += decoder_optimum

    print("Средний оптимум, полученный перебором точек оригинальной функции:")
    print(average_original / ITERATIONS_COUNT)

    print("Средний оптимум, полученный перебором точек скрытого пространства:")
    print(average_decoder / ITERATIONS_COUNT)


def main():
    autoencoder = AutoEncoder(input_dimension=X_DIMENSION + 1,
                                 encoding_dimension=ENCODING_DIMENSION,
                                 layer_dimension=LAYER_DIMENSION)
    train_autoencoder(autoencoder)

   # use_optimize_methods(autoencoder)
    use_exhaustive_search(autoencoder)


if __name__ == "__main__":
    main()
