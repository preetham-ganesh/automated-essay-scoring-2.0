import os
import re

import tensorflow as tf

from src.utils import load_json_file
from src.dataset import Dataset
from src.model import Model


class Train(object):
    """Trains the automatic essay scoring model based on the configuration."""

    def __init__(self, model_version: str) -> None:
        """Creates object attributes for the Train class.

        Creates object attributes for the Train class.

        Args:
            model_version: A string for the version of the current model.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(model_version, str), "Variable model_version of type 'str'."
        assert re.match(
            r"^\d+\.\d+\.\d+$", model_version
        ), "Variable model_version should be of format '#.#.#'."

        # Initalizes class variables.
        self.model_version = model_version
        self.best_validation_loss = None

    def load_model_configuration(self) -> None:
        """Loads the model configuration file for current version.

        Loads the model configuration file for current version.

        Args:
            None.

        Returns:
            None.
        """
        self.home_directory_path = os.getcwd()
        model_configuration_directory_path = "{}/configs/models/essay_scorer".format(
            self.home_directory_path
        )
        self.model_configuration = load_json_file(
            "v{}".format(self.model_version), model_configuration_directory_path
        )

    def load_dataset(self) -> None:
        """Loads dataset based on model configuration.

        Loads dataset based on model configuration.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes object for the Dataset class.
        self.dataset = Dataset(self.model_configuration)

        # Loads original training data as dataframe.
        self.dataset.load_dataset()

        # Preprocesses texts in the dataset to remove unwanted characters, and duplicates.
        self.dataset.preprocess_dataset()

        # Splits processed essay texts & scores into train, validation & test splits.
        self.dataset.split_dataset()

        # Trains the SentencePiece tokenizer on the processed texts from the dataset.
        self.dataset.train_tokenizer()

        # Converts list of texts & scores into TensorFlow dataset.
        self.dataset.shuffle_slice_dataset()

    def load_model(self, mode: str) -> None:
        """Loads model & other utilies for training.

        Loads model & other utilies for training.

        Args:
            mode: A string for mode by which the should be loaded, i.e., with latest checkpoints or not.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        self.model = Model(self.model_configuration)

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["optimizer"][
                "learning_rate"
            ]
        )

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.checkpoint_directory_path = (
            "{}/models/essay_scorer/v{}/checkpoints".format(
                self.home_directory_path, self.model_version
            )
        )
        checkpoint = tf.train.Checkpoint(model=self.model)
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=self.checkpoint_directory_path, max_to_keep=3
        )

        # If mode is predict, then the trained checkpoint is restored.
        if mode == "predict":
            checkpoint.restore(
                tf.train.latest_checkpoint(self.checkpoint_directory_path)
            )

        print("Finished loading model for current configuration.")
