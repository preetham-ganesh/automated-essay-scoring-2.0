import tensorflow as tf
import numpy as np

from typing import Dict, Any, List


class RNNClassifier(tf.keras.Model):
    """A tensorflow RNN classification model to automatically score essays."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the layers in the classification model.

        Initializes the layers in the classification model, by adding embedding, LSTM, dropout & dense layers.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        super(RNNClassifier, self).__init__()

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
                self.model_layers[name] = tf.keras.layers.Dropout(
                    rate=config["rate"], name=name
                )

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
        probabilities = inputs[3]

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
                x = self.model_layers[name]([x, probabilities])

            # Else, the following output is predicted.
            else:
                x = self.model_layers[name](x)
        return [x]

    def initialize_other_inputs(
        self, batch_size: int, n_classes: int, rnn_size: int
    ) -> List[tf.Tensor]:
        """Initializes hidden states m & c used in the LSTM layer for each batch.

        Initializes hidden states m & c used in the LSTM layer for each batch.

        Args:
            batch_size: An integer for the size of training/testing batch.
            n_classes: An integer for the no. of neurons in the last layer.
            rnn_size: An integer for the no. of neurons in the RNN layer.

        Returns:
            A list of tensors for the hidden states m & c and probabilities used in the RNN layer.
        """
        # Checks type & values of arguments.
        assert isinstance(
            batch_size, int
        ), "Variable batch_size should be of type 'int'."
        assert isinstance(rnn_size, int), "Variable rnn_size should be of type 'int'."

        # Creates empty tensors probabilities for hidden states h & c.
        probabilities = tf.ones((batch_size, n_classes)) / n_classes
        hidden_state_m = tf.zeros((batch_size, rnn_size))
        hidden_state_c = tf.zeros((batch_size, rnn_size))
        return [hidden_state_m, hidden_state_c, probabilities]

    def build_graph(self) -> tf.keras.Model:
        """Builds plottable graph for the model.

        Builds plottable graph for the model.

        Args:
            None.

        Returns:
            None.
        """
        # Creates the input layer using the model configuration.
        inputs = [
            tf.keras.layers.Input(
                shape=(self.model_configuration["model"]["max_length"],)
            ),
            tf.keras.layers.Input(
                shape=(
                    self.model_configuration["model"]["layers"]["configuration"][
                        "lstm_0"
                    ]["units"],
                )
            ),
            tf.keras.layers.Input(
                shape=(
                    self.model_configuration["model"]["layers"]["configuration"][
                        "lstm_0"
                    ]["units"],
                )
            ),
            tf.keras.layers.Input(
                shape=(self.model_configuration["model"]["n_classes"],)
            ),
        ]

        # Creates an object for the tensorflow model and returns it.
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))


class PositionalEmbedding(tf.keras.layers.Layer):
    """A tensorflow layer to compute position encoding for input."""

    def __init__(self, n_max_positions: int, units: int) -> None:
        """Initialize the PositionalEmbedding layer.

        Initializes the layer by generating the positional encoding matrix.

        Args:
            n_max_positions: An integer for the maximum no. of positions to create positional encodings.
            units: An integer for the dimensionality of the model.

        Returns:
            None.
        """
        super(PositionalEmbedding, self).__init__()

        # Initializes class variables.
        self.units = tf.cast(units, dtype=tf.float32)
        self.pos_encoding = self.positional_encoding(n_max_positions, units)

    def get_angles(self, positions: np.ndarray, indices: np.ndarray):
        """Calculate the angle rates for the positional encoding.

        Computes the angles used in the positional encoding for each position and dimension of the model.

        Args:
            positions: A numpy array of positions with shape (position, 1).
            indices: A numpy array of dimension indices with shape (1, d_model).

        Returns:
            A numpy array containing the calculated angles with shape (position, d_model).
        """
        angle_rates = 1 / np.power(10000, (2 * (indices // 2)) / self.units)
        return positions * angle_rates

    def positional_encoding(self, n_max_positions: int, units: int) -> tf.Tensor:
        """Generate the positional encoding matrix.

        Creates a matrix of positional encodings based on the input position and model dimensionality.

        position (int): The maximum number of positions (sequence length) for which to create positional encodings.
        d_model (int): The dimensionality of the model (number of encoding dimensions).

        Args:
            n_max_positions: An integer for the maximum no. of positions to create positional encodings.
            units: An integer for the dimensionality of the model.

        Returns:
            A tensor containing the positional encoding matrix with shape (1, position, d_model).
        """
        angle_rads = self.get_angles(
            np.arange(n_max_positions)[:, np.newaxis], np.arange(units)[np.newaxis, :]
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x: tf.Tensor):
        """Applies positional encoding to the input tensor.

        Applies positional encoding to the first tensor in the list. The encoding is scaled by the square root
        of the number of units and then added to the input tensor.

        Args:
            x: A tensor to which positional encoding will be applied.

        Returns:
            A tensor with positional encoding applied.
        """
        x *= tf.math.sqrt(self.units)
        x += self.pos_encoding[:, : tf.shape(x)[1], :]
        return x

    def compute_output_shape(self, input_shape: tuple[int]):
        """Computes the output shape of the layer.

        Computes the output shape of the layer.

        Args:
            input_shape: A tuple of integers for input shape of the array.

        Returns:
            A tuple of integers for the output shape.
        """
        return input_shape


class MultiHeadAttention(tf.keras.layers.Layer):
    """A tensorflow layer to compute multi-head attention."""

    def __init__(self, units: int, n_heads: int) -> None:
        """Initializes the MultiHeadAttention layer.

        Initializes the MultiHeadAttention layer.

        Args:
            units: An integer for the dimensionality of the model.
            n_heads: An integer for the number of attention heads.

        Returns:
            None.
        """
        super(MultiHeadAttention, self).__init__()

        # Initializes class variables.
        self.units = units
        self.n_heads = n_heads
        self.depth = units // n_heads
        self.w_q = tf.keras.layers.Dense(units)
        self.w_k = tf.keras.layers.Dense(units)
        self.w_v = tf.keras.layers.Dense(units)
        self.dense_0 = tf.keras.layers.Dense(units)

    def split_heads(self, x: tf.Tensor):
        """Splits the last dimension into (n_heads, depth).

        Splits the last dimension into (n_heads, depth).

        Args:
            x: A tensor for which last dimension is split into (n_heads, depth).

        Returns:
            A reshaped input tensor with shape (batch_size, n_heads, seq_len, depth).
        """
        x = tf.reshape(x, shape=(x.shape[0], -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(
        self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor = None
    ):
        """Calculates the scaled dot-product attention.

        Calculates the scaled dot-product attention for q, k, and v.

        Args:
            q: A tensor for Query of shape (..., seq_len_q, depth).
            k: A tensor for Key of shape (..., seq_len_k, depth).
            v: A tensor for Value of shape (..., seq_len_v, depth_v).
            mask: A list of tensors for mask shape broadcastable to (..., seq_len_q, seq_len_k).

        Returns:
            A tensor for the output tensor of the attention mechanism.
        """
        # Performs matrix multiplication of q and k, transposing k for correct dimensions.
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Calculates scaling factor based on the depth of k.Scales the attention logits by the square root of the depth.
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Adds the mask to the scaled attention logits, if provided, to prevent attending to certain positions.
        if mask is not None:
            scaled_attention_logits += mask[0] * -1e9

        # Apply the softmax function to get the attention weights & computes the output.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output

    def call(self, q: tf.Tensor, k: tf.Tensor, v: tf.Tensor, mask: tf.Tensor = None):
        """Applies the multi-head attention mechanism to the input.

        Applies the multi-head attention mechanism to the input.

        Args:
            inputs: A list of tensors for input to the layer.
            training: A boolean indicating whether the call is in training mode. Default is False.
            masks: A list of mask tensors. Default is None.

        Returns:
            tf.Tensor: Output tensor after applying the multi-head attention mechanism.
        """
        # Extracts Query, Key, and Value tensors from inputs.
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Splits the Query, Key, and Value tensors into multiple heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Computes scaled attention for Query, Key & Value tensor using the masks.
        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (q.shape[0], -1, self.units))
        output = self.dense_0(concat_attention)
        return output

    def compute_output_shape(self, input_shape: tuple[int]):
        """Computes the output shape of the layer.

        Computes the output shape of the layer.

        Args:
            input_shape: A tuple of integers for input shape of the array.

        Returns:
            A tuple of integers for the output shape.
        """
        return input_shape


class EncoderLayer(tf.keras.layers.Layer):
    """A tensorflow layer for encoder in transformer classifier."""

    def __init__(
        self,
        units: int,
        n_heads: int,
        ff_units: int,
        rate: float,
        epsilon: float = 1e-6,
    ) -> None:
        """Initializes the EncoderLayer class.

        Initializes the EncoderLayer class.

        Args:
            units: An integer for the no. of units in the model.
            n_heads: An integer for the no. of attention heads in the multi-head attention mechanism.
            ff_units: An integer for the number of units in the feed-forward network.
            rate: A floating point value for the dropout rate.
            epsilon: A floating point value to prevent division by zero in LayerNormalization.

        Returns:
            None.
        """
        super(EncoderLayer, self).__init__()

        # Initializes the class variables.
        self.units = units
        self.ff_units = ff_units
        self.mha = MultiHeadAttention(units, n_heads)
        self.ffn = self.point_wise_feed_forward_network()
        self.layer_norm_0 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.dropout_0 = tf.keras.layers.Dropout(rate)
        self.dropout_1 = tf.keras.layers.Dropout(rate)

    def point_wise_feed_forward_network(self) -> tf.keras.Model:
        """Creates a point-wise feed-forward network.

        Creates a point-wise feed-forward network.

        Args:
            None.

        Returns:
            A tensorflow model for the point-wise feed-forward network.
        """
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.units, activation="relu"),
                tf.keras.layers.Dense(self.ff_units),
            ]
        )

    def call(self, x: tf.Tensor, training: bool = False, mask: tf.Tensor = None):
        """Applies the encoder layer to the input tensor.

        Applies the encoder layer to the input tensor.

        Args:
            x: A tensor for the input.
            training: A boolean flag indicating whether the layer should behave in training mode or inference mode.
            mask: A tensor representing the mask to be applied to the attention mechanism. Defaults to None.

        Returns:
            tf.Tensor: The output tensor after applying the encoder layer transformations.
        """
        # Applies multi-head attention. Applies layer normalization to the sum of the input and attention output.
        attention_output = self.mha(x, x, x, mask)
        attention_output = self.dropout_0(attention_output, training=training)
        output_0 = self.layer_norm_0(x + attention_output)

        # Applies the feed-forward network to the normalized output. Applies layer normalization to the sum of the
        # previous output and the feed-forward network output.
        ffn_output = self.ffn(output_0)
        ffn_output = self.dropout_1(ffn_output, training=training)
        output_1 = self.layer_norm_1(output_0 + ffn_output)
        return output_1


class TransformerClassifier(tf.keras.Model):
    """A tensorflow transformer classification model to automatically score essays."""

    def __init__(self, model_configuration: Dict[str, Any]) -> None:
        """Initializes the layers in the classification model.

        Initializes the layers in the classification model, by adding embedding, dropout & dense layers.

        Args:
            model_configuration: A dictionary for the configuration of model's current version.

        Returns:
            None.
        """
        super(TransformerClassifier, self).__init__()

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
            config = self.model_configuration["model"]["layers"]["configuration"]

            # If layer's name is like 'embedding_', an Embedding layer is initialized based on layer configuration.
            if name.split("_")[0] == "embedding":
                self.model_layers[name] = tf.keras.layers.Embedding(
                    input_dim=config["vocab_size"],
                    output_dim=config["units"],
                    name=name,
                )

            # If layer's name is like 'posembedding_', an PositionalEmbedding layer is initialized based on layer
            # configuration.
            elif name.split("_")[0] == "posembedding":
                self.model_layers[name] = PositionalEmbedding(
                    n_max_positions=config["n_max_positions"],
                    units=config["units"],
                )

            # If layer's name is like 'dropout_', a Dropout layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dropout":
                self.model_layers[name] = tf.keras.layers.Dropout(
                    rate=config["rate"], name=name
                )

            # If layer's name is like 'mha_', a Multi-head attention layer is initialized based on layer configuration.
            elif name.split("_")[0] == "mha":
                self.model_layers[name] = MultiHeadAttention(
                    units=config["units"], n_heads=config["n_heads"]
                )

            # If layer's name is like 'layernorm', a Normalization layer is initialized based on layer configuration.
            elif name.split("_")[0] == "layernorm":
                self.model_layers[name] = tf.keras.layers.LayerNormalization(
                    epsilon=config["epsilon"], name=name
                )

            # If layer's name is like 'add', an Add layer is initialized based on layer configuration.
            elif name.split("_")[0] == "add":
                self.model_layers[name] = tf.keras.layers.Add()

            # If layer's name is like 'dense_', a Dense layer is initialized based on layer configuration.
            elif name.split("_")[0] == "dense":
                self.model_layers[name] = tf.keras.layers.Dense(
                    units=config["units"], activation=config["activation"], name=name
                )

    def call(
        self,
        inputs: List[tf.Tensor],
        training: bool = False,
        masks: List[tf.Tensor] = None,
    ):
        """"""
        # Extracts inputs from list of tensors.
        x = inputs[0]
        padding_mask = masks[0]

        # Iterates across the layers arrangement, and predicts the output for each layer.
        for name in self.model_configuration["model"]["layers"]["arrangement"]:
            config = self.model_configuration["model"]["layers"]["configuration"]

            # If layer's name is like 'dropout_', the following output is predicted.
            if name.split("_")[0] == "dropout":
                x = self.model_layers[name](x, training=training)

            #
            elif name.split("_")[0] == "mha":
                attention_out = self.model_layers[name](x, x, x, padding_mask)
                print(name, attention_out.shape)

            else:
                x = self.model_layers[name](x)
            print(name, x.shape)
        return x


inp = tf.ones((64, 50))
enc_padding_mask = tf.cast(tf.math.equal(inp, 0), tf.float32)
enc_padding_mask = enc_padding_mask[:, tf.newaxis, tf.newaxis, :]

model_config = {
    "model": {
        "layers": {
            "arrangement": ["embedding_0", "posembedding_0", "dropout_0", "mha_0"],
            "configuration": {
                "vocab_size": 4096,
                "units": 512,
                "rate": 0.3,
                "n_max_positions": 4096,
                "n_heads": 8,
            },
        }
    }
}

model = TransformerClassifier(model_config)
_ = model([inp], training=True, masks=[enc_padding_mask])
print(model.summary())
