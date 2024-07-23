import tensorflow as tf

from typing import Dict, Any, List


class Model(tf.keras.Model):
    """A tensorflow model to automatically score essays."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the layers in the classification model.

        Initializes the layers in the classification model, by adding embedding, LSTM, dropout & dense layers.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        super(Model, self).__init__()

        # Asserts type of input arguments.
        assert isinstance(
            model_configuration, dict
        ), "Variable model_configuration should be of type 'dict'."

        # Initializes class variables.
        self.model_configuration = model_configuration
        self.model_layers = dict()

        # Iterates across layers in the layers arrangement.
        self.model_layers = dict()
        for name in self.model_configuration["model"]["layers"]["arrangement"]:
            config = self.model_configuration["model"]["layers"]["configuration"][name]

            # If layer's name is like 'embedding_', an Embedding layer is initialized based on layer configuration.
            if name.split("_")[0] == "embedding":
                self.model_layers[name] = tf.keras.layers.Embedding(
                    input_dim=config["input_dim"],
                    output_dim=config["output_dim"],
                    name=name,
                )

            # If layer's name is like 'lstm_', an LSTM layer is initialized based on layer configuration.
            elif name.split("_")[0] == "lstm":
                self.model_layers[name] = tf.keras.layers.LSTM(
                    units=config["units"],
                    return_state=config["return_state"],
                    return_sequences=config["return_sequences"],
                    name=name,
                )

            # If layer's name is like 'dense_', a Dense layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dense":
                self.model_layers[name] = tf.keras.layers.Dense(
                    units=config["units"], activation=config["activation"], name=name
                )

            # If layer's name is like 'dropout_', a Dropout layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dropout":
                self.model_layers[name] = tf.keras.layers.Dropout(rate=config["rate"])

            # If layer's name is like 'concatenate_', a Concatenate layer is initialized based on layer configuration.
            elif name.split("_")[0] == "concat":
                self.model_layers[name] = tf.keras.layers.Concatenate(
                    axis=config["axis"], name=name
                )

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = False,
        masks: List[tf.Tensor] = None,
    ) -> List[tf.Tensor]:
        """Input tensor is passed through the layers in the model.

        Input tensor is passed through the layers in the model.

        Args:
            inputs: A list for the inputs from the input batch.
            training: A boolean value for the flag of training/testing state.
            masks: A tensor for the masks from the input batch.

        Returns:
            A tensor for the processed output from the components in the layer.
        """
        # Asserts type & values of the input arguments.
        assert isinstance(inputs, list), "Variable inputs should be of type 'list'."
        assert isinstance(training, bool), "Variable training should be of type 'bool'."
        assert (
            isinstance(masks, list) or masks is None
        ), "Variable masks should be of type 'list' or masks should have value as 'None'."

        # Extracts inputs from list of tensors.
        x = inputs[0]
        hidden_state_m = inputs[1]
        hidden_state_c = inputs[2]
        previous_result = inputs[3]

        # Iterates across the layers arrangement, and predicts the output for each layer.
        for name in self.model_configuration["model"]["layers"]["arrangement"]:
            config = self.model_configuration["model"]["layers"]["configuration"][name]

            # If layer's name is like 'dropout_', the following output is predicted.
            if name.split("_")[0] == "dropout":
                x = self.model_layers[name](x, training=training)

            # If layer's name is like 'lstm_', the following output is predicted.
            elif name.split("_")[0] == "lstm":
                if config["return_state"] == True:
                    x, hidden_state_m, hidden_state_c = self.model_layers[name](
                        x, initial_state=[hidden_state_m, hidden_state_c]
                    )
                else:
                    x = self.model_layers[name](
                        x, initial_state=[hidden_state_m, hidden_state_c]
                    )

            # If layer's name is like 'concat_', the following output is predicted.
            elif name.split("_")[0] == "concat":
                x = self.model_layers[name]([x, previous_result])

            # Else, the following output is predicted.
            else:
                x = self.model_layers[name](x)
        return [x]
