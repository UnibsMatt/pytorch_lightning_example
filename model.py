import logging

import pytorch_lightning as pl
import torch
from torch.nn.modules import CrossEntropyLoss, Linear, ReLU, Sequential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.CRITICAL)


class DummyModel(pl.LightningModule):
    """
    This module will create a dummy dataset based on distribution of random number
    """

    def __init__(self, n_class: int):
        """
        Init method for dummy model
        Args:
            n_class:
        """
        super().__init__()
        self.ll = Sequential(
            Linear(64, 32),
            ReLU(),
            Linear(32, n_class),
        )

        self.criterion = CrossEntropyLoss()

    def forward(self, x):
        return self.ll(x)

    def training_step(self, batch, batch_index):
        logging.info("Entering training_step")
        # x, y taken from dataset. Here both are tensor
        # x -> double tensor
        # y -> long tensor
        x, y = batch
        output = self(x.float())
        return {"out": output, "label": y}

    def training_step_end(self, outputs):
        logging.info("Entering training_step_end")
        out, y = outputs["out"], outputs["label"]
        loss = self.criterion(out, y)
        # return {"loss":loss, "parma1":1, "param2": 2,}
        return loss

    def training_epoch_end(self, outputs) -> None:
        logging.info("Entering epoch end")
        """
        In outputs now we have all the output returned in the training_step_end as a dict.
        In our case all the losses from the batches.
        If in the previous methods we returned
        
        return {"loss":loss, "parma1":1, "param2": 2, ...}
        now outputs would have had all these parameters x batch       
        
        Outputs is a list of dict in training even if we pass a tensor. The cast to dict is implicit
        """
        all_epoch_losses = [out["loss"] for out in outputs]
        final_loss = torch.stack(all_epoch_losses).mean()
        logging.warning(f"Loss: {final_loss}")

    def validation_step(self, batch, batch_index):
        logging.info("Entering validation_step")
        # x, y taken from dataset. Here both are tensor
        # x -> double tensor
        # y -> long tensor
        x, y = batch
        output = self(x.float())
        return {"out": output, "label": y}

    def validation_step_end(self, outputs):
        logging.info("Entering validation_step_end")
        out, y = outputs["out"], outputs["label"]
        val_loss = self.criterion(out, y)
        """
        Returning a tensor just for demonstration. In the validation the cast to dict must be explicit.
        In this case in validation_epochend we have a list of tensor.
        If we want to pass a dict then we must retrive the loss as a dict
        """
        return val_loss

    def validation_epoch_end(self, outputs) -> None:
        final_validation_loss = torch.stack(outputs).mean()
        logging.warning(f"Loss: {final_validation_loss}")
        self.log("val_loss", final_validation_loss)

    """ The same appened for the tests steps
    def test_step(self, a, b):
        pass

    def test_step_end(self, a,b):
        pass

    def test_epoch_end(self, outputs) -> None:
        pass
        
    """

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
