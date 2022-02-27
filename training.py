import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks import CustomCallback
from datamodule import CustomDataModule
from model import DummyModel


@hydra.main(config_path="configuration", config_name="base_config.yaml")
def train(params):
    data_m = CustomDataModule()
    model = DummyModel(params.model.n_classes)

    # #################################### CALLBACKS ###############################################################
    custom_call = CustomCallback()

    checkpoint_p = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        verbose=True,
        filename="modello-{epoch}--{val_loss:0.4f}",
        save_top_k=-1,
        # save every 5 epoch
        period=5,
    )

    ###############################################################################################################

    """
    distributed_backend as been substitute by accelerator -> distributed_backend="ddp"
                                                          -> accelerator="ddp"
    """
    trainer = pl.Trainer(
        # gpu used
        # gpus = [0, 1, 2, 3]
        # default_root_dir="results/",
        # accelerator="ddp",
        callbacks=[checkpoint_p, custom_call],
        min_epochs=1,
        max_epochs=50,
        # if the training is fast this can be good for the system
        progress_bar_refresh_rate=20,
        # summary of the model. Params and structure
        # weights_sum = ["full", None, "top"]
        weights_summary="top",
        # this is call at the end of the fit!
        profiler="simple",
    )

    trainer.fit(model, data_m)


if __name__ == "__main__":
    train()
