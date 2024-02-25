import numpy as np
from scipy.stats.qmc import Sobol
import autoencoder as ae
import tensorflow as tf

X_DIMENSION = 20
ENCODING_DIMENSION = 5
LAYER_DIMENSION = 100

POINTS_COUNT = 2048


def rastrigin(x: np.ndarray) -> float:
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def generate_points(dimension: int, count: int) -> np.ndarray:
    return Sobol(d=dimension, seed=0).random(n=count)


def get_y_values(x_values: np.ndarray) -> np.ndarray:
    return np.array([rastrigin(x) for x in x_values])


def create_test_data(x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    return np.hstack((x_values, y_values.reshape(-1, 1)))


def train_autoencoder(autoencoder: ae.AutoEncoder, train_data: np.ndarray) -> None:
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(train_data,
                    train_data,
                    validation_split=0.2,
                    epochs=1000,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(patience=10)
                    ])


def find_optimum(x_values: np.ndarray, y_values: np.ndarray) -> tuple:
    min_index = np.argmin(y_values)
    return x_values[min_index], np.min(y_values)


def print_solution(title: str, vector: np.ndarray, optimum: float) -> None:
    print(f"--- {title} ---")
    print("Vector:")
    print(vector)
    print("Optimum:")
    print(optimum)
    print()


def main():
    x_values = generate_points(X_DIMENSION, POINTS_COUNT)
    y_values = get_y_values(x_values)
    test_data = create_test_data(x_values, y_values)

    autoencoder = ae.AutoEncoder(input_dimension=X_DIMENSION + 1,
                                 encoding_dimension=ENCODING_DIMENSION,
                                 layer_dimension=LAYER_DIMENSION)
    train_autoencoder(autoencoder, test_data)

    example_x_values = generate_points(X_DIMENSION, POINTS_COUNT)
    example_y_values = get_y_values(example_x_values)
    example_vector, example_optimum = find_optimum(example_x_values, example_y_values)

    example_autoencoder_x_values = generate_points(ENCODING_DIMENSION, POINTS_COUNT)
    decoder_function = autoencoder.decoder
    decoder_all_values = decoder_function(example_autoencoder_x_values)
    decoder_x_values = decoder_all_values[:, :-1]
    decoder_y_values = get_y_values(decoder_x_values)
    decoder_vector, decoder_optimum = find_optimum(decoder_x_values, decoder_y_values)

    print_solution("Original", example_vector, example_optimum)
    print_solution("Decoder", np.array(decoder_vector), decoder_optimum)


if __name__ == "__main__":
    main()
