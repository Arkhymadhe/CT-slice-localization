""" Utilities for model instantiation, training, and prediction.
"""

from skorch import NeuralNetRegressor
from skorch import NeuralNetClassifier

from neural_net import MyNet


def prepare_net(task, n_features, n_classes):
    """Return nueral network."""

    net = MyNet(task=task, n_features=n_features, n_classes=n_classes)
    return net


def import_model(
    optim,
    task: str,
    n_features: int,
    n_classes: int,
    lr: float,
    max_epochs: int,
    batch_size: int,
    split: float,
):

    """Import model class."""

    net = prepare_net(task=task, n_features=n_features, n_classes=n_classes)

    if task == "classif":
        final_net = NeuralNetClassifier(
            module=net,
            optimizer=optim,
            optimizer__lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=split,
        )
    else:
        final_net = NeuralNetRegressor(
            module=net,
            optimizer=optim,
            optimizer__lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            train_split=split,
        )
    return final_net


def train_model(model, X, y):
    """Fit an instance of a model."""

    model.fit(X, y)
    return model


def get_predictions(model, features, return_features=True):
    """Obtain predictions for a set of features."""

    pred = model.predict(features)

    return (pred, features) if return_features else pred
