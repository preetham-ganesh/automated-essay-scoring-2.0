import os
import sys
import warnings
import logging
import argparse


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)
warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.FATAL)


import mlflow

from src.utils import set_physical_devices_memory_limit
from src.train import Train


def main():
    print()
    # Parses the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-stu",
        "--set_tracking_uri",
        type=str,
        required=True,
        help="URI where MLFlow server is located.",
    )
    parser.add_argument(
        "-en",
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the MLFlow experiment on which the model should be trained.",
    )
    parser.add_argument(
        "-mv",
        "--model_version",
        type=str,
        required=True,
        help="Version by which the trained model files should be saved as.",
    )
    args = parser.parse_args()

    # Sets memory limit of GPU if found in the system.
    set_physical_devices_memory_limit()

    # Sets tracking URI for MLFlow server.
    mlflow.set_tracking_uri(args.set_tracking_uri)

    # Gets experiment based on name.
    existing_experiment = mlflow.get_experiment_by_name(args.experiment_name)

    # If experiment does not exist, then creates experiment with name & set it.
    if not existing_experiment:
        mlflow.create_experiment(args.experiment_name, artifact_location="")
    mlflow.set_experiment(args.experiment_name)

    # Creates an object for the Train class.
    trainer = Train(args.model_version)

    # Loads model configuration for current model version.
    trainer.load_model_configuration()

    # Loads dataset based on dataset version in the model configuration.
    trainer.load_dataset()

    # Loads model & other utilies for training it.
    trainer.load_model("train")

    # Generates summary and plot for loaded model.
    trainer.generate_model_summary_and_plot(True)

    # Trains & validates the model using train & validation dataset.
    trainer.fit()

    # Loads the model with latest checkpoint.
    trainer.load_model("predict")

    # Tests the model using the test dataset.
    trainer.test_model()

    # Serializes model as TensorFlow module & saves it as MLFlow artifact.
    trainer.serialize_model()


if __name__ == "__main__":
    main()
