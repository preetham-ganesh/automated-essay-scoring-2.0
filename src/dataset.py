import os
import re
import unicodedata

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

    def load_dataset(self) -> None:
        """Loads original training & testing data as dataframes.

        Loads original training & testing data as dataframes.

        Args:
            None.

        Returns:
            None.
        """
        home_directory_path = os.getcwd()

        # Loads the original train & test dataframes.
        self.original_train_df = pd.read_csv(
            "{}/data/raw_data/train.csv".format(home_directory_path)
        )
        self.original_test_df = pd.read_csv(
            "{}/data/raw_data/test.csv".format(home_directory_path)
        )
        print(
            "No. of original examples in the train data: {}".format(
                len(self.original_train_df)
            )
        )
        print(
            "No. of original examples in the test data: {}".format(
                len(self.original_test_df)
            )
        )
        print("")

    def preprocess_text(self, text: str) -> str:
        """Preprocess text to remove/normalize unwanted characters.

        Normalizes UNICODE characters to ASCII format in the given text, removes non-ASCII characters, add spaces between
            special characters and removes unwanted spaces.

        Args:
            text: A string for the current text which needs to be processed.

        Returns:
            A string for the processed text with ASCII characters and spaces.
        """
        # Asserts type of input arguments.
        assert isinstance(text, str), "Variable text should be of type 'str'."

        # Removes HTML/XML tags from text.
        html_tags = re.compile(r"<[^>]+>")
        text = html_tags.sub(" ", text)

        # Converts UNICODE characters to ASCII format, and removes non-ASCII characters.
        text = "".join(
            character
            for character in unicodedata.normalize("NFKD", str(text))
            if unicodedata.category(character) != "Mn"
        )
        text = re.sub(r"[^\x00-\x7f]", "", text)

        # Removes unwanted characters from text.
        unwanted_characters = ["`", "\\", "%", "+", "=", "~", "^"]
        n_unwanted_characters = len(unwanted_characters)
        for index in range(n_unwanted_characters):
            text = text.replace(unwanted_characters[index], "")
        text = re.sub(r"[^!@$&()[]{}:;,./?\|'a-zA-Z0-9]+", "", text)

        # Removes beginning, trailing and unwanted spaces found in text.
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text


dataset = Dataset({})
dataset.load_dataset()
dataset.preprocess_dataset()
