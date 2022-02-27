from pytorch_lightning.callbacks import Callback


class CustomCallback(Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """Called when the train epoch ends."""

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the val epoch begins."""

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, trainer, pl_module):
        """Called when the test epoch begins."""

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
