import argparse
import os
import torch
import joblib

from data_ops import read_compress
from data_ops import get_invariant_features, variables
from data_ops import array_to_tensor, split_data

from model_utils import train_model, import_model
from metrics import get_r2_score


def configure_args():
    """Return CLI arguments."""

    args = argparse.ArgumentParser(description="Argument name space for CLI flags.")

    args.add_argument(
        "--data_dir",
        type=str,
        help="Data directory",
        default=os.path.join(os.getcwd().replace("scripts", "data"), "ct-dataset.csv"),
    )

    args.add_argument(
        "--arch_dir",
        type=str,
        help="Directory for generated compressed data file",
        default=os.path.join(os.getcwd().replace("scripts", "data"), "ct-dataset.zip"),
    )

    args.add_argument(
        "--style",
        type=str,
        default="gruvboxd",
        help="Visualization style",
        choices=["gruvboxd", "solarizedd", "solarizedl", "normal", "chesterish"],
    )

    args.add_argument(
        "--epochs", type=int, default=20, help="Number of complete data rounds"
    )

    args.add_argument("--lr", type=float, default=1e-4, help="Convergence rate")

    args.add_argument("--classes", default=None, help="Number of classes")

    args.add_argument("--n_features", type=int, default=348, help="Data dimentionality")

    args.add_argument(
        "--cardinality", type=int, default=50, help="Max number of possible classes"
    )

    args.add_argument(
        "--var_lim", type=float, default=0.8, help="Max allowable limit on variance"
    )

    args.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args.add_argument("--t1", type=float, default=0.2, help="Train split")

    args.add_argument("--t2", type=float, default=0.35, help="Valid split")

    args.add_argument("--split", default=0.2, help="Validation split while fitting?")

    args.add_argument(
        "--valid",
        type=bool,
        choices=[True, False],
        default=True,
        help="Validation split?",
    )

    args.add_argument(
        "--save",
        type=bool,
        default=True,
        choices=[True, False],
        help="Save model artefacts?",
    )

    args.add_argument(
        "--artefact_dir",
        type=str,
        default=os.getcwd().replace("scripts", "artefacts"),
        help="Location for saved model",
    )

    args.add_argument(
        "--task",
        default="regression",
        type=str,
        choices=["classif", "regression"],
        help="Type of experience, E",
    )

    return args


def main():
    print(">>> Configure CLI arguments...")
    args = argparse.ArgumentParser(description="Argument name space for CLI flags.")

    args.add_argument(
        "--data_dir",
        type=str,
        help="Data directory",
        default=os.path.join(os.getcwd().replace("scripts", "data"), "ct-dataset.csv"),
    )

    args.add_argument(
        "--arch_dir",
        type=str,
        help="Directory for generated compressed data file",
        default=os.path.join(os.getcwd().replace("scripts", "data"), "ct-dataset.zip"),
    )

    args.add_argument(
        "--style",
        type=str,
        default="gruvboxd",
        help="Visualization style",
        choices=["gruvboxd", "solarizedd", "solarizedl", "normal", "chesterish"],
    )

    args.add_argument(
        "--epochs", type=int, default=20, help="Number of complete data rounds"
    )

    args.add_argument("--lr", type=float, default=1e-4, help="Convergence rate")

    args.add_argument("--classes", default=None, help="Number of classes")

    args.add_argument("--n_features", type=int, default=348, help="Data dimentionality")

    args.add_argument(
        "--cardinality", type=int, default=50, help="Max number of possible classes"
    )

    args.add_argument(
        "--var_lim", type=float, default=0.8, help="Max allowable limit on variance"
    )

    args.add_argument("--batch_size", type=int, default=16, help="Batch size")

    args.add_argument("--t1", type=float, default=0.2, help="Train split")

    args.add_argument("--t2", type=float, default=0.35, help="Valid split")

    args.add_argument("--split", default=0.2, help="Validation split while fitting?")

    args.add_argument(
        "--valid",
        type=bool,
        choices=[True, False],
        default=True,
        help="Validation split?",
    )

    args.add_argument(
        "--save",
        type=bool,
        default=True,
        choices=[True, False],
        help="Save model artefacts?",
    )

    args.add_argument(
        "--artefact_dir",
        type=str,
        default=os.getcwd().replace("scripts", "artefacts"),
        help="Location for saved model",
    )

    args.add_argument(
        "--task",
        default="regression",
        type=str,
        choices=["classif", "regression"],
        help="Type of experience, E",
    )
    args = args.parse_args()

    print(">> CLI arguments configured!")
    print()

    print(">>> Importing dataset...")
    data, path_to_archive = read_compress(
        path_to_data=args.data_dir, path_to_archive=args.arch_dir
    )

    print(">>> Dataset imported successfully!")
    print()

    print(">>> Data preprocessing underway...")
    drop_labels = get_invariant_features(
        data.iloc[:, :-1], cardinality=args.cardinality, percent=args.var_lim
    )

    X, y = variables(data, labels=drop_labels, target="reference", return_y=True)

    X, y = array_to_tensor(X), array_to_tensor(y)

    print(">>> Data preprocessing complete!")
    print()

    ### Split dataset for training, validation, and testing
    print(">>> Partitioning data into splits...")
    X_train, X_test, y_train, y_test = split_data(X, y, split_size=args.t1)

    if args.valid:
        X_train, X_val, y_train, y_val = split_data(
            X_train, y_train, split_size=args.t2
        )

    print(">>> Data splits created!")
    print()

    ### Instantiate model object
    model = import_model(
        task=args.task,
        n_features=args.n_features,
        n_classes=args.classes,
        optim=torch.optim.SGD,
        lr=args.lr,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        split=None,
    )

    ### Train model
    print(">>> Model training underway...")
    model = train_model(model, X_train, y_train)

    print(">>> Model training complete!")
    print()

    print(">>> Displaying diagnostics...")
    print()

    ### Train scores
    print(">" * 10, " Train Scores ", "<" * 10)
    print(get_r2_score(y_train, model.predict(X_train), num_places=5, text=True))
    print()

    if args.valid:
        ### Validation
        print(">" * 10, " Validation Scores ", "<" * 10)
        print(get_r2_score(y_val, model.predict(X_val), num_places=5, text=True))
        print()

    ### Test scores
    print(">" * 10, " Test Scores ", "<" * 10)
    print(get_r2_score(y_test, model.predict(X_test), num_places=5, text=True))

    if args.save:
        ### Save artefacts
        if not os.path.exists(args.artefact_dir):
            os.makedirs(args.artefact_dir)

        with open(os.path.join(args.artefact_dir, "trained_model.bin"), "wb") as f:
            joblib.dump(model, f)


if __name__ == "__main__":
    main()
