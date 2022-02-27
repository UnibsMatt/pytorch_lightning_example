## Pytorch-lightning version 1.1.2

There are 4 main components for the new lightning version that are slightly different from the previous one:

- [Model](#Model)
- [Training](#Training)
- [Datamodule](#Datamodule)
- [Callbacks](#Callbacks)

# Model

The model is similar to the previous version of lightning.  

```python
import pytorch_lightning as pl
from torch.nn.modules import Sequential, Linear, ReLU, CrossEntropyLoss
import torch
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.CRITICAL)


class DummyModel(pl.LightningModule):
    def __init__(self, n_class):
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
        x, y = batch
        output = self(x)
        return {"out": output, "label": y}

    def training_step_end(self, outputs):
        out, y = outputs["out"], outputs["label"]
        loss = self.criterion(out, y)
        return loss

    def training_epoch_end(self, outputs) -> None:
        all_epoch_losses = [out["loss"] for out in outputs]
        final_loss = torch.stack(all_epoch_losses).mean()
        logging.warning(f"Loss: {final_loss}")

    def validation_step(self, batch, batch_index):
        x, y = batch
        output = self(x)
        return {"out": output, "label": y}

    def validation_step_end(self, outputs):
        return val_loss

    def validation_epoch_end(self, outputs) -> None:
        pass
    def test_step(self, a, b):
        pass

    def test_step_end(self, a,b):
        pass

    def test_epoch_end(self, outputs) -> None:
        pass
    
    def configure_optimizers(self):
        pass

```
The required implementations are:
- forward()
- configure_optimizer()

Training/validation dataset are moved in the [datamodule](#datamodule) part.

---

# Datamodule

Here we can find all the dataloader that will be pass at training time.

```python 
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from fake_dataset import FakeDataset


class CustomDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

    def prepare_data(self):
        # download, tokenize or operations that use disk and it's done on a single gpu in a distributed scenario
        # DO NOT ASSIGN self.variable in this methods!!!!!!!!!!
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = FakeDataset(mode="train")
            self.val_dataset = FakeDataset(mode="validation")
            self.test_dataset = FakeDataset(mode="test")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=64)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=64)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=64)

```
If not specified, the basic Trainer will call the setup method in this class with stage=="fit".  
Otherwise more control can be achieved by manually call the setup methods with th desired stages see [examples](#examples).
  

---

# Callbacks

The callbacks are used in order to control the training flow in all his part. 
We can configure hooks that are invoked at the setup, during the validation phase or after the training epoch.  

```python
from pytorch_lightning.callbacks import Callback


class CustomCallback(Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch begins."""
        pass

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """Called when the train epoch ends."""
        pass

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called when the val epoch begins."""
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the val epoch ends."""
        pass

    def on_test_epoch_start(self, trainer, pl_module):
        """Called when the test epoch begins."""
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the test epoch ends."""
        pass
```

---

# Training

In the training part I use hydra as configuration manager.  
Adding custom callbacks can be helpfull to controll the execution flow.  
 


```python
import pytorch_lightning as pl
from datamodule import CustomDataModule
from model import DummyModel
from pytorch_lightning.callbacks import ModelCheckpoint
from callbacks import CustomCallback
import hydra


@hydra.main(config_path="configuration", config_name="base_config.yaml")
def train(params):
    data_m = CustomDataModule()
    model = DummyModel(params.model.n_classes)
    custom_call = CustomCallback()

    checkpoint_p = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        verbose=True,
        filename="modello-figo-{epoch}--{val_loss:0.4f}",
        save_top_k=-1,
        period=5,

    )

    trainer = pl.Trainer(
        gpus = [0, 1, 2, 3],
        accelerator="ddp",
        callbacks=[checkpoint_p, custom_call],
        min_epochs=1,
        max_epochs=50,
        weights_summary="top",
    )

    trainer.fit(model, data_m)


if __name__ == "__main__":
    train()

```

---

# Examples

Here we define the stage for the datamodule in order to control which dataloader we want to use.  

```python
import pytorch_lightning as pl
from datamodule import CustomDataModule
from model import DummyModel

data_m = CustomDataModule()
data_m.prepare_data()
data_m.setup(stage="eval")

model = DummyModel(2)

trainer = pl.Trainer(gpus = [0])

trainer.fit(model, data_m)
```

For the usage of the callbacks we use a model checkpoint callbacks.

```python
checkpoint_p = ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    verbose=True,
    #dirpath="results",
    filename="modello-figo-{epoch}--{val_loss:0.4f}",
    save_top_k=-1,
    # save every 5 epoch
    period=5,
)
```
This allow the user to save the model that achieve the smaller val_loss.  
In order to monitor the value "val_loss", we must use in the training loop the methods

```python
self.log("name_of_var_to_log", value_to_log)
```
For example:
```python
    def validation_epoch_end(self, outputs) -> None:
        final_validation_loss = torch.stack(outputs).mean()
        self.log("val_loss", final_validation_loss)
```