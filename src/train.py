import os

from src.utils import load_json_file
from src.dataset import Dataset


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
