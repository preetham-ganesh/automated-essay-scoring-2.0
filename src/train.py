import os
import re
import time

import tensorflow as tf
import mlflow

from src.utils import load_json_file, check_directory_path_existence
from src.dataset import Dataset
from src.model import RNNClassifier, TransformerClassifier


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
        model_configuration_directory_path = "{}/configs".format(
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

        # Updates model configuration with trained tokenizer vocab size.
        if self.model_configuration["model"]["type"] == "rnn":
            self.model_configuration["model"]["layers"]["configuration"]["embedding_0"][
                "input_dim"
            ] = (self.dataset.spp.get_piece_size() + 2)
        elif self.model_configuration["model"]["type"] == "transformer":
            self.model_configuration["model"]["layers"]["configuration"][
                "vocab_size"
            ] = (self.dataset.spp.get_piece_size() + 2)

    def load_model(self, mode: str) -> None:
        """Loads model & other utilies for training.

        Loads model & other utilies for training.

        Args:
            mode: A string for mode by which the should be loaded, i.e., with latest checkpoints or not.

        Returns:
            None.
        """
        # Loads model for current model configuration.
        if self.model_configuration["model"]["type"] == "rnn":
            self.model = RNNClassifier(self.model_configuration)
        elif self.model_configuration["model"]["type"] == "transformer":
            self.model = TransformerClassifier(self.model_configuration)

        # Loads the optimizer.
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.model_configuration["model"]["optimizer"][
                "learning_rate"
            ]
        )

        # Creates checkpoint manager for the neural network model and loads the optimizer.
        self.checkpoint_directory_path = "{}/models/v{}/checkpoints".format(
            self.home_directory_path, self.model_version
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

    def generate_model_summary_and_plot(self, plot: bool) -> None:
        """Generates summary & plot for loaded model.

        Generates summary & plot for loaded model.

        Args:
            pool: A boolean value to whether generate model plot or not.

        Returns:
            None.
        """
        # Builds plottable graph for the model.
        model = self.model.build_graph()

        # Compiles the model to log the model summary.
        model_summary = list()
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = "\n".join(model_summary)
        mlflow.log_text(
            model_summary,
            "v{}/summary.txt".format(self.model_configuration["version"]),
        )

        # Creates the following directory path if it does not exist.
        self.reports_directory_path = check_directory_path_existence(
            "models/v{}/reports".format(self.model_version)
        )

        # Plots the model & saves it as a PNG file.
        if plot:
            tf.keras.utils.plot_model(
                model,
                "{}/model_plot.png".format(self.reports_directory_path),
                show_shapes=True,
                show_layer_names=True,
                expand_nested=True,
            )

            # Logs the saved model plot PNG file.
            mlflow.log_artifact(
                "{}/model_plot.png".format(self.reports_directory_path),
                "v{}".format(self.model_configuration["version"]),
            )

    def initialize_metric_trackers(self) -> None:
        """Initializes trackers which computes the mean of all metrics.

        Initializes trackers which computes the mean of all metrics.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.validation_loss = tf.keras.metrics.Mean(name="validation_loss")
        self.train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")
        self.validation_accuracy = tf.keras.metrics.Mean(name="validation_accuracy")

    def compute_loss(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes loss for the current batch using actual & predicted values.

        Computes loss for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor for target batch of generated mask images.
            predicted_batch: A tensor for batch of outputs predicted by the model for input batch.

        Returns:
            A tensor for the loss computed on comparing target & predicted batch.
        """
        # Computes loss for the current batch using actual values and predicted values.
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = self.loss_object(target_batch, predicted_batch)
        return loss

    def compute_accuracy(
        self, target_batch: tf.Tensor, predicted_batch: tf.Tensor
    ) -> tf.Tensor:
        """Computes accuracy for the current batch using actual & predicted values.

        Computes accuracy for the current batch using actual & predicted values.

        Args:
            target_batch: A tensor which contains the actual values for the current batch.
            predicted_batch: A tensor which contains the predicted values for the current batch.

        Returns:
            A tensor for the accuracy of current batch.
        """
        # Computes accuracy for the current batch using actual values and predicted values.
        accuracy = tf.keras.metrics.categorical_accuracy(target_batch, predicted_batch)
        return accuracy

    @tf.function
    def train_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Trains model using current input & target batches.

        Trains model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for training the model.
            target_batch: A tensor for the target text from the current batch for training and validating the model.

        Returns:
            None.
        """
        # Initializes the hidden states & starting probabilities from the model for each batch.
        hidden_state_m, hidden_state_c, probabilities = (
            self.model.initialize_other_inputs(
                target_batch.shape[0],
                self.model_configuration["model"]["n_classes"],
                self.model_configuration["model"]["layers"]["configuration"]["lstm_0"][
                    "units"
                ],
            )
        )

        # Iterates across tokenized & encoded subphrases in input batch.
        loss = 0
        accuracy = 0
        with tf.GradientTape() as tape:
            n_subsequences = 0
            for id_0 in range(
                0, input_batch.shape[1], self.model_configuration["model"]["max_length"]
            ):
                # Predicts output for current subphrase & computes loss & accuracy.
                probabilities = self.model(
                    [
                        input_batch[
                            :,
                            id_0 : id_0
                            + self.model_configuration["model"]["max_length"],
                        ],
                        hidden_state_m,
                        hidden_state_c,
                        probabilities,
                    ],
                    training=True,
                    masks=None,
                )[0]
                loss += self.compute_loss(target_batch, probabilities)
                accuracy += self.compute_accuracy(target_batch, probabilities)
                n_subsequences += 1

        # Computes batch loss & accuracy.
        batch_loss = loss / n_subsequences
        batch_accuracy = accuracy / n_subsequences

        # Computes gradients using loss and model variables.
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Uses optimizer to apply the computed gradients on the combined model variables.
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Appends batch loss & accuracy to main metrics.
        self.train_loss(batch_loss)
        self.train_accuracy(batch_accuracy)

    def validation_step(self, input_batch: tf.Tensor, target_batch: tf.Tensor) -> None:
        """Validates model using current input & target batches.

        Validates model using current input & target batches.

        Args:
            input_batch: A tensor for the input text from the current batch for validating the model.
            target_batch: A tensor for the target text from the current batch for validating the model.

        Returns:
            None.
        """
        # Initializes the hidden states from the decoder for each batch.
        hidden_state_m, hidden_state_c, probabilities = (
            self.model.initialize_other_inputs(
                target_batch.shape[0],
                self.model_configuration["model"]["n_classes"],
                self.model_configuration["model"]["layers"]["configuration"]["lstm_0"][
                    "units"
                ],
            )
        )

        # Iterates across tokenized & encoded subphrases in input batch.
        loss = 0
        accuracy = 0
        n_subsequences = 0
        for id_0 in range(
            0, input_batch.shape[1], self.model_configuration["model"]["max_length"]
        ):
            # Predicts output for current subphrase & computes loss & accuracy.
            probabilities = self.model(
                [
                    input_batch[
                        :,
                        id_0 : id_0 + self.model_configuration["model"]["max_length"],
                    ],
                    hidden_state_m,
                    hidden_state_c,
                    probabilities,
                ]
            )[0]
            loss += self.compute_loss(target_batch, probabilities)
            accuracy += self.compute_accuracy(target_batch, probabilities)
            n_subsequences += 1

        # Computes batch loss & accuracy.
        batch_loss = loss / n_subsequences
        batch_accuracy = accuracy / n_subsequences

        # Appends batch loss & accuracy to main metrics.
        self.validation_loss(batch_loss)
        self.validation_accuracy(batch_accuracy)

    def reset_trackers(self) -> None:
        """Resets states for trackers before the start of each epoch.

        Resets states for trackers before the start of each epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.train_loss.reset_state()
        self.validation_loss.reset_state()
        self.train_accuracy.reset_state()
        self.validation_accuracy.reset_state()

    def train_model_per_epoch(self, epoch: int) -> None:
        """Trains the model using train dataset for current epoch.

        Trains the model using train dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Iterates across batches in the train dataset.
        for batch, (texts, scores) in enumerate(
            self.dataset.train_dataset.take(self.dataset.n_train_steps_per_epoch)
        ):
            batch_start_time = time.time()

            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(texts.numpy()), list(scores.numpy())
            )

            # Trains the model using the current input and target batch.
            self.train_step(input_batch, target_batch)
            print(
                "Epoch={}, Batch={}, Train loss={}, Train accuracy={}, Time taken={} sec.".format(
                    epoch,
                    batch,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.train_accuracy.result().numpy(), 3)),
                    round(time.time() - batch_start_time, 3),
                )
            )

        # Logs train metrics for current step.
        mlflow.log_metrics(
            {
                "train_loss": self.train_loss.result().numpy(),
                "train_accuracy": self.train_accuracy.result().numpy(),
            },
            step=epoch,
        )
        print()

    def validate_model_per_epoch(self, epoch: int) -> None:
        """Validates the model using validation dataset for current epoch.

        Validates the model using validation dataset for current epoch.

        Args:
            epoch: An integer for the number of current epoch.

        Returns:
            None.
        """
        # Iterates across batches in the validation dataset.
        for batch, (texts, scores) in enumerate(
            self.dataset.validation_dataset.take(
                self.dataset.n_validation_steps_per_epoch
            )
        ):
            batch_start_time = time.time()

            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(texts.numpy()), list(scores.numpy())
            )

            # Validates the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)
            print(
                "Epoch={}, Batch={}, Validation loss={}, Validation accuracy={}, Time taken={} sec.".format(
                    epoch,
                    batch,
                    str(round(self.validation_loss.result().numpy(), 3)),
                    str(round(self.validation_accuracy.result().numpy(), 3)),
                    round(time.time() - batch_start_time, 3),
                )
            )

        # Logs validation metrics for current epoch.
        mlflow.log_metrics(
            {
                "validation_loss": self.validation_loss.result().numpy(),
                "validation_accuracy": self.validation_accuracy.result().numpy(),
            },
            step=epoch,
        )
        print()

    def save_model(self) -> None:
        """Saves the model after checking performance metrics in current epoch.

        Saves the model after checking performance metrics in current epoch.

        Args:
            None.

        Returns:
            None.
        """
        self.manager.save()
        print("Checkpoint saved at {}.".format(self.checkpoint_directory_path))

    def early_stopping(self) -> bool:
        """Stops the model from learning further if the performance has not improved from previous epoch.

        Stops the model from learning further if the performance has not improved from previous epoch.

        Args:
            None.

        Returns:
            None.
        """
        # If epoch = 1, then best validation loss is replaced with current validation loss, & the checkpoint is saved.
        if self.best_validation_loss is None:
            self.patience_count = 0
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is higher than current validation loss, the best validation loss is replaced with
        # current validation loss, & the checkpoint is saved.
        elif self.best_validation_loss > str(
            round(self.validation_loss.result().numpy(), 3)
        ):
            self.patience_count = 0
            print(
                "Best validation loss changed from {} to {}".format(
                    str(self.best_validation_loss),
                    str(round(self.validation_loss.result().numpy(), 3)),
                )
            )
            self.best_validation_loss = str(
                round(self.validation_loss.result().numpy(), 3)
            )
            self.save_model()

        # If best validation loss is not higher than the current validation loss, then the number of times the model
        # has not improved is incremented by 1.
        elif self.patience_count < 2:
            self.patience_count += 1
            print("Best validation loss did not improve.")
            print("Checkpoint not saved.")

        # If the number of times the model did not improve is greater than 4, then model is stopped from training.
        else:
            return False
        return True

    def fit(self) -> None:
        """Trains & validates the loaded model using train & validation dataset.

        Trains & validates the loaded model using train & validation dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Initializes TensorFlow trackers which computes the mean of all metrics.
        self.initialize_metric_trackers()

        # Iterates across epochs for training the neural network model.
        for epoch in range(1, self.model_configuration["model"]["epochs"] + 1):
            epoch_start_time = time.time()

            # Resets states for training and validation metrics before the start of each epoch.
            self.reset_trackers()

            # Trains the model using batces in the train dataset.
            self.train_model_per_epoch(epoch)

            # Validates the model using batches in the validation dataset.
            self.validate_model_per_epoch(epoch)
            print(
                "Epoch={}, Train loss={}, Validation loss={}, Train Accuracy={}, Validation Accuracy={}, "
                "Time taken={} sec.".format(
                    epoch,
                    str(round(self.train_loss.result().numpy(), 3)),
                    str(round(self.validation_loss.result().numpy(), 3)),
                    str(round(self.train_accuracy.result().numpy(), 3)),
                    str(round(self.validation_accuracy.result().numpy(), 3)),
                    round(time.time() - epoch_start_time, 3),
                )
            )

            # Stops the model from learning further if the performance has not improved from previous epoch.
            model_training_status = self.early_stopping()
            if not model_training_status:
                print(
                    "Model did not improve after 4th time. Model stopped from training further."
                )
                print("")
                break
            print("")

    def test_model(self) -> None:
        """Tests the trained model using the test dataset.

        Tests the trained model using the test dataset.

        Args:
            None.

        Returns:
            None.
        """
        # Resets states for validation metrics.
        self.reset_trackers()

        # Iterates across batches in the train dataset.
        for batch, (texts, scores) in enumerate(
            self.dataset.test_dataset.take(self.dataset.n_test_steps_per_epoch)
        ):
            # Loads input & target sequences for current batch as tensors.
            input_batch, target_batch = self.dataset.load_input_target_batches(
                list(texts.numpy()), list(scores.numpy())
            )

            # Tests the model using the current input and target batch.
            self.validation_step(input_batch, target_batch)

        # Logs test metrics for current epoch.
        mlflow.log_metrics(
            {
                "test_loss": self.validation_loss.result().numpy(),
                "test_accuracy": self.validation_accuracy.result().numpy(),
            }
        )
        print(
            "Test loss={}.".format(str(round(self.validation_loss.result().numpy(), 3)))
        )
        print(
            "Test accuracy={}.".format(
                str(round(self.validation_accuracy.result().numpy(), 3))
            ),
        )
        print("")

    def serialize_model(self) -> None:
        """Serializes model as TensorFlow module & saves it as MLFlow artifact.

        Serializes model as TensorFlow module & saves it as MLFlow artifact.

        Args:
            None.

        Returns:
            None.
        """
        # Defines input shapes for exported model's input signature.
        input_0_shape = [None, self.model_configuration["model"]["max_length"]]
        input_1_shape = [
            None,
            self.model_configuration["model"]["layers"]["configuration"]["lstm_0"][
                "units"
            ],
        ]
        input_2_shape = [
            None,
            self.model_configuration["model"]["layers"]["configuration"]["lstm_0"][
                "units"
            ],
        ]
        input_3_shape = [None, self.model_configuration["model"]["n_classes"]]

        class ExportModel(tf.Module):
            """Exports trained tensorflow model as tensorflow module for serving."""

            def __init__(self, model: tf.keras.Model) -> None:
                """Initializes the variables in the class.

                Initializes the variables in the class.

                Args:
                    model: A tensorflow model for the model trained with latest checkpoints.

                Returns:
                    None.
                """
                # Initializes class variables.
                self.model = model

            @tf.function(
                input_signature=[
                    tf.TensorSpec(shape=input_0_shape, dtype=tf.int32),
                    tf.TensorSpec(shape=input_1_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=input_2_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=input_3_shape, dtype=tf.float32),
                ]
            )
            def predict(
                self,
                x: tf.Tensor,
                hidden_state_m: tf.Tensor,
                hidden_state_c: tf.Tensor,
                probabilities: tf.Tensor,
            ):
                """Inputs are passed through the model for prediction.

                Inputs are passed through the model for prediction.

                Args:
                    x: A tensor for the input at the current timestep.
                    hidden_state_m: A tensor for the hidden state m passed through RNN layers in the model.
                    hidden_state_c: A tensor for the hidden state c passed through RNN layers in the model.
                    probabilities: A tensor for the output from the previous input batch.

                Return:
                    A tensor for the output predicted by the encoder for the current image.
                """
                prediction = self.model(
                    [x, hidden_state_m, hidden_state_c, probabilities]
                )
                return prediction

        # Exports trained tensorflow model as tensorflow module for serving.
        exported_model = ExportModel(self.model)

        # Saves the tensorflow object created from the loaded model.
        home_directory_path = os.getcwd()
        tf.saved_model.save(
            exported_model,
            "{}/models/v{}/serialized".format(home_directory_path, self.model_version),
        )

        # Logs serialized model as artifact.
        mlflow.log_artifacts(
            "{}/models/v{}/serialized".format(home_directory_path, self.model_version),
            "v{}/model".format(self.model_configuration["version"]),
        )

        # Logs updated model configuration as artifact.
        mlflow.log_dict(self.model_configuration, "v{}.json".format(self.model_version))
