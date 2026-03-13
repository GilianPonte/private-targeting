from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import random
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


def _set_random_seed(seed: int | None) -> None:
    """Set random seeds across supported libraries."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)



def _as_1d_array(values: Any, *, name: str) -> np.ndarray:
    """Convert tabular input to a flattened NumPy array."""
    arr = np.asarray(pd.DataFrame(values)).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr



def _as_2d_array(values: Any, *, name: str) -> np.ndarray:
    """Convert tabular input to a 2D NumPy array."""
    arr = np.asarray(pd.DataFrame(values))
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr



def _validate_inputs(X: Any, Y: Any, T: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_arr = _as_2d_array(X, name="X")
    Y_arr = _as_1d_array(Y, name="Y")
    T_arr = _as_1d_array(T, name="T")

    n = X_arr.shape[0]
    if len(Y_arr) != n or len(T_arr) != n:
        raise ValueError(
            "X, Y, and T must contain the same number of observations. "
            f"Received shapes: X={X_arr.shape}, Y={Y_arr.shape}, T={T_arr.shape}."
        )

    unique_t = np.unique(T_arr)
    if not np.all(np.isin(unique_t, [0, 1])):
        raise ValueError(
            "T must be a binary treatment indicator taking values in {0, 1}. "
            f"Observed unique values: {unique_t.tolist()}"
        )

    return X_arr, Y_arr.astype(float), T_arr.astype(int)



def _build_model_factory(input_dim: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=(input_dim,)))
        activation = hp.Choice("activation", ["leaky_relu", "relu"])
        for i in range(hp.Int("num_layers", 1, 4)):
            units = hp.Choice(f"units_{i}", [8, 16, 32, 64, 256, 512])
            model.add(layers.Dense(units=units, activation=None))
            if activation == "leaky_relu":
                model.add(layers.LeakyReLU())
            else:
                model.add(layers.ReLU())

        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["MSE"],
        )
        return model

    return build_model



def _make_checkpoint_callbacks(checkpoint_path: Path, *, patience: int):
    import tensorflow as tf

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        mode="min",
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_path),
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_freq="epoch",
        save_best_only=True,
    )
    return [early_stopping, checkpoint]



def _fit_propensity_model(
    X_train: np.ndarray,
    T_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    clf = LogisticRegression(verbose=0, max_iter=1000)
    clf.fit(X_train, T_train.reshape(-1))
    probabilities = clf.predict_proba(X_test)
    return probabilities[:, 1]



def _check_t_tilde(values: np.ndarray) -> None:
    if np.any(np.isclose(values, 0.0)):
        raise ZeroDivisionError(
            "Encountered one or more near-zero residualized treatment values. "
            "This would make the pseudo outcome unstable. Consider adjusting the "
            "data split, model specification, or regularization."
        )



def cnn(
    X,
    Y,
    T,
    scaling: bool = True,
    batch_size: int = 100,
    epochs: int = 100,
    max_epochs: int = 10,
    folds: int = 5,
    directory: str = "tuner",
    seed: int | None = None,
):
    """
    Causal neural network estimator for average and conditional treatment effects.

    Parameters
    ----------
    X:
        Feature matrix with shape ``(n_samples, n_features)``.
    Y:
        Outcome vector.
    T:
        Binary treatment vector.
    scaling:
        Whether to min-max scale ``X`` to ``[-1, 1]``.
    batch_size:
        Training batch size.
    epochs:
        Number of training epochs.
    max_epochs:
        Maximum number of epochs used by Keras Tuner Hyperband.
    folds:
        Number of cross-fitting folds.
    directory:
        Directory for tuner and checkpoint artifacts.
    seed:
        Random seed.

    Returns
    -------
    tuple
        ``(average_treatment_effect, cate_estimates, tau_hat_model)``
    """
    try:
        import keras_tuner
    except ImportError as e:
        raise ImportError(
            "keras-tuner is required for cnn(). Install with: pip install keras-tuner"
        ) from e

    from tensorflow import keras

    _set_random_seed(seed)
    X_arr, Y_arr, T_arr = _validate_inputs(X, Y, T)

    if folds < 2:
        raise ValueError("folds must be at least 2.")
    if folds > len(X_arr):
        raise ValueError("folds cannot exceed the number of observations.")

    if scaling:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_arr = scaler.fit_transform(X_arr)

    output_dir = Path(directory)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    build_model = _build_model_factory(X_arr.shape[1])
    checkpoint_filepath_mx = checkpoint_dir / "cnn_m_x.keras"
    checkpoint_filepath_taux = checkpoint_dir / "cnn_tau_x.keras"

    mx_callbacks = _make_checkpoint_callbacks(checkpoint_filepath_mx, patience=5)
    tau_callbacks = _make_checkpoint_callbacks(checkpoint_filepath_taux, patience=5)

    print("hyperparameter optimization for yhat")
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs=max_epochs,
        overwrite=True,
        directory=str(output_dir),
        project_name="yhat",
        seed=seed,
    )
    tuner.search(
        X_arr,
        Y_arr,
        epochs=epochs,
        validation_split=0.25,
        verbose=0,
        callbacks=[mx_callbacks[0]],
    )

    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"the optimal architecture is: {best_hps.values}")

    y_tilde_hat = np.array([], dtype=float)
    t_tilde_hat = np.array([], dtype=float)

    cv = KFold(n_splits=folds, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr)):
        _set_random_seed(seed)

        model_m_x = tuner.hypermodel.build(best_hps)
        model_m_x.fit(
            X_arr[train_idx],
            Y_arr[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_arr[test_idx], Y_arr[test_idx]),
            callbacks=mx_callbacks,
            verbose=0,
        )

        model_m_x = keras.models.load_model(checkpoint_filepath_mx)
        m_x = model_m_x.predict(x=X_arr[test_idx], verbose=0).reshape(len(test_idx))

        y_tilde = Y_arr[test_idx].reshape(len(test_idx)) - m_x
        y_tilde_hat = np.concatenate((y_tilde_hat, y_tilde))

        e_x = _fit_propensity_model(X_arr[train_idx], T_arr[train_idx], X_arr[test_idx])
        print(
            f"Fold {fold}: mean(m_x) = {np.round(np.mean(m_x), 2):.2f}, "
            f"sd(m_x) = {np.round(np.std(m_x), 3):.3f} and "
            f"mean(e_x) = {np.round(np.mean(e_x), 2):.2f}, "
            f"sd(e_x) = {np.round(np.std(e_x), 3):.3f}"
        )

        t_tilde = T_arr[test_idx].reshape(len(test_idx)) - e_x
        t_tilde_hat = np.concatenate((t_tilde_hat, t_tilde))

    _check_t_tilde(t_tilde_hat)
    pseudo_outcome = y_tilde_hat / t_tilde_hat
    w_weights = np.square(t_tilde_hat)

    print("hyperparameter optimization for tau hat")
    tuner_tau = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs=max_epochs,
        overwrite=True,
        directory=str(output_dir),
        project_name="tau_hat",
        seed=seed,
    )
    tuner_tau.search(
        X_arr,
        pseudo_outcome,
        epochs=epochs,
        validation_split=0.25,
        verbose=0,
        callbacks=[tau_callbacks[0]],
    )
    best_hps_tau = tuner_tau.get_best_hyperparameters()[0]
    print(f"the optimal architecture is: {best_hps_tau.values}")

    cate_estimates = np.array([], dtype=float)
    print("training for tau hat")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr)):
        _set_random_seed(seed)

        tau_hat = tuner_tau.hypermodel.build(best_hps_tau)
        tau_hat.fit(
            X_arr[train_idx],
            pseudo_outcome[train_idx],
            sample_weight=w_weights[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            callbacks=tau_callbacks,
            validation_data=(X_arr[test_idx], pseudo_outcome[test_idx]),
            verbose=0,
        )

        tau_hat = keras.models.load_model(checkpoint_filepath_taux)
        cate = tau_hat.predict(x=X_arr[test_idx], verbose=0).reshape(len(test_idx))
        print(
            f"Fold {fold}: mean(tau_hat) = {np.round(np.mean(cate), 2):.2f}, "
            f"sd(tau_hat) = {np.round(np.std(cate), 3):.3f}"
        )

        cate_estimates = np.concatenate((cate_estimates, cate))

    average_treatment_effect = float(np.mean(cate_estimates))
    print(f"ATE = {np.round(average_treatment_effect, 4)}")
    return average_treatment_effect, cate_estimates, tau_hat



def _parse_privacy_statement(statement: str) -> tuple[int, float, float, float]:
    """Extract privacy quantities from tensorflow-privacy's printable statement."""
    numbers = [
        float(num) if "." in num else int(num)
        for num in re.findall(r"\d+\.\d+|\d+", statement)
    ]
    if len(numbers) < 9:
        raise RuntimeError(
            "Could not parse the privacy statement returned by tensorflow-privacy. "
            f"Statement was: {statement!r}"
        )
    n, epsilon, noise_multiplier, epsilon_conservative = (
        int(numbers[0]),
        float(numbers[8]),
        float(numbers[2]),
        float(numbers[7]),
    )
    return n, epsilon, noise_multiplier, epsilon_conservative



def pcnn(
    X,
    Y,
    T,
    scaling: bool = True,
    batch_size: int = 100,
    epochs: int = 100,
    max_epochs: int = 1,
    directory: str = "tuner",
    fixed_model: bool = False,
    noise_multiplier: float | None = None,
    seed: int | None = None,
):
    """
    Differentially private causal neural network estimator.

    This function implements the privacy-preserving estimator from the provided code.
    In the paper's terminology, this maps to the DP-CATE training-stage strategy.

    Returns
    -------
    tuple
        ``(
            average_treatment_effect,
            cate_estimates,
            tau_hat_model,
            n,
            epsilon,
            noise_multiplier,
            epsilon_conservative,
        )``
    """
    try:
        import keras_tuner
    except ImportError as e:
        raise ImportError(
            "keras-tuner is required for pcnn(). Install with: pip install keras-tuner"
        ) from e

    try:
        import tensorflow as tf
        import tensorflow_privacy
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import (
            DPKerasAdamOptimizer,
        )
    except ImportError as e:
        raise ImportError(
            "tensorflow-privacy is required for pcnn(). Install with: "
            "pip install tensorflow-privacy"
        ) from e

    if noise_multiplier is None:
        raise ValueError("noise_multiplier must be provided for pcnn().")

    _set_random_seed(seed)
    X_arr, Y_arr, T_arr = _validate_inputs(X, Y, T)

    if (len(X_arr) / 2) % batch_size != 0:
        divisors = [
            i
            for i in range(1, int(math.sqrt((len(X_arr) / 2))) + 1)
            if (len(X_arr) / 2) % i == 0
        ]
        divisors += [(len(X_arr) / 2) // i for i in divisors if (len(X_arr) / 2) // i != i]
        divisors.sort()
        raise ValueError(
            "The chosen batch size does not divide half the sample into a whole number. "
            f"Try one of these batch sizes instead: {np.round(divisors)}"
        )

    statement = tensorflow_privacy.compute_dp_sgd_privacy_statement(
        number_of_examples=len(X_arr),
        batch_size=batch_size,
        num_epochs=epochs,
        noise_multiplier=noise_multiplier,
        delta=1 / len(X_arr),
        used_microbatching=False,
        max_examples_per_user=1,
    )
    print(statement)
    n, epsilon, parsed_noise_multiplier, epsilon_conservative = _parse_privacy_statement(statement)

    if scaling:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_arr = scaler.fit_transform(X_arr)

    output_dir = Path(directory)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def ate_metric(y_true, y_pred):
        del y_true
        return tf.reduce_mean(y_pred, axis=-1)

    def generate_fixed_architecture(X_: np.ndarray):
        model = keras.Sequential()
        model.add(keras.Input(shape=(X_.shape[1],)))

        units = 64
        for _ in range(4):
            model.add(layers.Dense(units, activation="tanh"))
            units = max(units // 2, 1)

        model.add(layers.Dense(1, activation="linear"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["MSE"],
        )
        return model

    build_model = _build_model_factory(X_arr.shape[1])

    idx = np.random.permutation(np.arange(len(X_arr)))
    X_arr = X_arr[idx]
    Y_arr = Y_arr[idx]
    T_arr = T_arr[idx]

    checkpoint_filepath_mx = checkpoint_dir / f"{parsed_noise_multiplier}_m_x.keras"
    checkpoint_filepath_taux = checkpoint_dir / f"{parsed_noise_multiplier}_tau_x.keras"

    mx_callbacks = _make_checkpoint_callbacks(checkpoint_filepath_mx, patience=20)
    tau_callbacks = _make_checkpoint_callbacks(checkpoint_filepath_taux, patience=20)

    print("hyperparameter optimization for yhat")
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective="val_loss",
        max_epochs=max_epochs,
        overwrite=True,
        directory=str(output_dir),
        project_name="yhat",
        seed=seed,
    )
    tuner.search(
        X_arr,
        Y_arr,
        epochs=epochs,
        validation_split=0.25,
        verbose=0,
        callbacks=mx_callbacks,
    )
    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"the optimal architecture is: {best_hps.values}")

    y_tilde_hat = np.array([], dtype=float)
    t_tilde_hat = np.array([], dtype=float)

    cv = KFold(n_splits=2, shuffle=False)
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr)):
        _set_random_seed(seed)

        model_m_x = tuner.hypermodel.build(best_hps)
        model_m_x.fit(
            X_arr[train_idx],
            Y_arr[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_arr[test_idx], Y_arr[test_idx]),
            callbacks=mx_callbacks,
            verbose=0,
        )

        model_m_x = keras.models.load_model(checkpoint_filepath_mx)
        m_x = model_m_x.predict(x=X_arr[test_idx], verbose=0).reshape(len(test_idx))

        y_tilde = Y_arr[test_idx].reshape(len(test_idx)) - m_x
        y_tilde_hat = np.concatenate((y_tilde_hat, y_tilde))

        e_x = _fit_propensity_model(X_arr[train_idx], T_arr[train_idx], X_arr[test_idx])
        print(
            f"Fold {fold}: mean(m_x) = {np.round(np.mean(m_x), 2):.2f}, "
            f"sd(m_x) = {np.round(np.std(m_x), 3):.3f} and "
            f"mean(e_x) = {np.round(np.mean(e_x), 2):.2f}, "
            f"sd(e_x) = {np.round(np.std(e_x), 3):.3f}"
        )

        t_tilde = T_arr[test_idx].reshape(len(test_idx)) - e_x
        t_tilde_hat = np.concatenate((t_tilde_hat, t_tilde))

    _check_t_tilde(t_tilde_hat)
    pseudo_outcome = y_tilde_hat / t_tilde_hat
    w_weights = np.square(t_tilde_hat)

    cate_estimates = np.array([], dtype=float)
    print("training for tau hat")
    for fold, (train_idx, test_idx) in enumerate(cv.split(X_arr)):
        _set_random_seed(seed)

        tau_hat = generate_fixed_architecture(X_arr)
        if fixed_model is False:
            tau_hat = tuner.hypermodel.build(best_hps)

        tau_hat.compile(
            optimizer=DPKerasAdamOptimizer(
                l2_norm_clip=4,
                noise_multiplier=noise_multiplier,
                num_microbatches=batch_size,
                learning_rate=0.001,
            ),
            loss=tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE,
            ),
            metrics=[ate_metric],
        )

        tau_hat.fit(
            X_arr[train_idx],
            pseudo_outcome[train_idx],
            sample_weight=w_weights[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            callbacks=tau_callbacks,
            validation_data=(X_arr[test_idx], pseudo_outcome[test_idx]),
            verbose=0,
        )

        tau_hat = keras.models.load_model(checkpoint_filepath_taux, compile=False)
        cate = tau_hat.predict(x=X_arr[test_idx], verbose=0).reshape(len(test_idx))
        print(
            f"Fold {fold}: mean(tau_hat) = {np.round(np.mean(cate), 2):.2f}, "
            f"sd(tau_hat) = {np.round(np.std(cate), 3):.3f}"
        )

        cate_estimates = np.concatenate((cate_estimates, cate))

    average_treatment_effect = float(np.mean(cate_estimates))
    reverse_idx = np.argsort(idx)
    cate_estimates = cate_estimates[reverse_idx]
    print(f"ATE = {average_treatment_effect}")

    return (
        average_treatment_effect,
        cate_estimates,
        tau_hat,
        n,
        epsilon,
        parsed_noise_multiplier,
        epsilon_conservative,
    )



def ctenn(*args, **kwargs):
    """Paper-aligned alias for :func:`cnn`."""
    return cnn(*args, **kwargs)



def dp_cate(*args, **kwargs):
    """Paper-aligned alias for :func:`pcnn`."""
    return pcnn(*args, **kwargs)


__all__ = ["cnn", "pcnn", "ctenn", "dp_cate"]
