import os
import re
import unicodedata

import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.data import from_pandas
import tensorflow as tf

from utils import save_text_file

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
        self.original_df = pd.read_csv(
            "{}/data/raw_data/train.csv".format(home_directory_path)
        )

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

    def preprocess_dataset(self) -> None:
        """Preprocesses texts in the dataset to remove unwanted characters, and duplicates.

        Preprocesses texts in the dataset to remove unwanted characters, and duplicates.

        Args:
            None.

        Returns:
            None.
        """
        # Drops any duplicates of text from the original dataset.
        self.original_df.drop_duplicates(
            subset=["full_text"], keep="first", inplace=True
        )

        # Creates empty list to store processed texts & its scores.
        self.processed_df = list()

        # Iterates across text & score in the original dataset.
        for text, score in zip(
            self.original_df["full_text"], self.original_df["score"]
        ):

            # Preprocess text to remove/normalize unwanted characters.
            text = self.preprocess_text(text)

            # If processed text is empty, then it is skipped.
            if text == "":
                continue

            # Processed text & score is added to the lists.
            self.processed_df.append({"text": text, "score": score})

        # Converts list of dictionaries into dataframe.
        self.processed_df = pd.DataFrame.from_records(self.processed_df)

    def split_dataset(self) -> None:
        """Splits processed essay texts & scores into train, validation & test splits.

        Splits processed essay texts & scores into train, validation & test splits.

        Args:
            None.

        Returns:
            None.
        """
        # Splits processed texts & scores into train, validation, & test splits.
        (
            self.train_df,
            self.validation_df,
        ) = train_test_split(
            self.processed_df,
            test_size=self.model_configuration["dataset"]["split_percentage"][
                "validation"
            ],
            shuffle=True,
            random_state=42,
        )
        (
            self.train_df,
            self.test_df,
        ) = train_test_split(
            self.train_df,
            test_size=self.model_configuration["dataset"]["split_percentage"]["test"],
            shuffle=True,
            random_state=42,
        )

        # Logs train, validation & test datasets to mlflow server as inputs.
        mlflow.log_input(
            from_pandas(
                self.train_df,
                name="{}_v{}_train".format(
                    self.model_configuration["dataset"]["name"],
                    self.model_configuration["dataset"]["version"],
                ),
            )
        )
        mlflow.log_input(
            from_pandas(
                self.validation_df,
                name="{}_v{}_validation".format(
                    self.model_configuration["dataset"]["name"],
                    self.model_configuration["dataset"]["version"],
                ),
            )
        )
        mlflow.log_input(
            from_pandas(
                self.test_df,
                name="{}_v{}_test".format(
                    self.model_configuration["dataset"]["name"],
                    self.model_configuration["dataset"]["version"],
                ),
            )
        )

        # Computes no. of examples per data split.
        self.n_train_examples = len(self.train_df)
        self.n_validation_examples = len(self.validation_df)
        self.n_test_examples = len(self.test_df)

    def train_tokenizer(self) -> None:
        """Trains the SentencePiece tokenizer on the processed texts from the dataset.

        Trains the SentencePiece tokenizer on the processed texts from the dataset.

        Args:
            None.

        Returns:
            None.
        """
        home_directory_path = os.getcwd()

        # Combines list of strings into a single string.
        combined_text = "\n".join(self.train_df["text"])

        # Saves string as a text file.
        save_text_file(combined_text, "temp", "")

        # Train a SentencePiece model using the temporary file.
        spm.SentencePieceTrainer.train(
            input="temp.txt",
            model_prefix="{}/models/v{}/{}".format(
                home_directory_path,
                self.model_configuration["model"]["version"],
                self.model_configuration["tokenizer"]["name"],
            ),
            vocab_size=8000,
        )

        # Logs the trained tokenizer model.
        mlflow.log_artifact(
            "{}/models/v{}/{}.model".format(
                home_directory_path,
                self.model_configuration["model"]["version"],
                self.model_configuration["tokenizer"]["name"],
            ),
            "v{}".format(self.model_configuration["model"]["version"]),
        )

        # Deletes the combined text file.
        os.remove("temp.txt")

    def shuffle_slice_dataset(self) -> None:
        """Converts list of texts & scores into TensorFlow dataset.

        Converts list of texts & scores into TensorFlow dataset & slices them based on batch size.

        Args:
            None.

        Returns:
            None.
        """
        # Zips texts & scores into single tensor, and shuffles it.
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (list(self.train_df["text"]), list(self.train_df["score"]))
        ).shuffle(len(self.train_df))
        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (list(self.validation_df["text"]), list(self.validation_df["score"]))
        ).shuffle(len(self.validation_df))
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            (list(self.test_df["text"]), list(self.test_df["score"]))
        ).shuffle(len(self.test_df))

        # Deletes unwanted variables.
        del (self.train_df, self.validation_df, self.test_df)

        # Slices the combined dataset based on batch size, and drops remainder values.
        self.batch_size = self.model_configuration["model"]["batch_size"]
        self.train_dataset = self.train_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.validation_dataset = self.validation_dataset.batch(
            self.batch_size, drop_remainder=True
        )
        self.test_dataset = self.test_dataset.batch(
            self.batch_size, drop_remainder=True
        )

        # Computes number of steps per epoch for all dataset.
        self.n_train_steps_per_epoch = self.n_train_examples // self.batch_size
        self.n_validation_steps_per_epoch = (
            self.n_validation_examples // self.batch_size
        )
        self.n_test_steps_per_epoch = self.n_test_examples // self.batch_size


dataset = Dataset(
    {
        "dataset": {"split_percentage": {"test": 0.05, "validation": 0.05}},
        "tokenizer": {"vocab_size": 8192, "name": "tokenizer"},
    }
)
dataset.load_dataset()
dataset.preprocess_dataset()
dataset.split_dataset()
dataset.train_tokenizer()
