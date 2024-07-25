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
