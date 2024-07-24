import os

import pandas as pd

from typing import Dict, Any, List


class Dataset(object):
    """Loads the dataset based on model configuration."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Creates object attributes for the Dataset class.

        Creates object attributes for the Dataset class.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        # Asserts type & value of the arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initalizes class variables.
        self.model_configuration = model_configuration

    def load_dataset() -> None:
        """Loads original training & testing data as dataframes.

        Loads original training & testing data as dataframes.

        Args:
            None.

        Returns:
            None.
        """
        home_directory_path = os.getcwd()

        # Loads the original train & test dataframes.
        original_train_df = pd.read_csv(
            "{}/data/raw_data/train.csv".format(home_directory_path)
        )
        original_test_df = pd.read_csv(
            "{}/data/raw_data/test.csv".format(home_directory_path)
        )
        print(
            "No. of original examples in the train data: {}".format(
                len(original_train_df)
            )
        )
        print(
            "No. of original examples in the test data: {}".format(
                len(original_test_df)
            )
        )
        print("")
