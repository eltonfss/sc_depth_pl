from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from config import get_opts
from data_modules import VideosDataModule
from SC_Depth import SC_Depth
from SC_DepthV2 import SC_DepthV2
from SC_DepthV3 import SC_DepthV3
from SC_DepthV3_ER import SC_DepthV3_ER
from er_buffer import ExperienceReplayBuffer
from er_buffer_initializer import initialize_er_buffer

if __name__ == '__main__':
    hparams = get_opts()

    # pl model
    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    elif hparams.model_version == 'v3':
        system = SC_DepthV3(hparams)
    elif hparams.model_version == 'v3_with_er':
        er_buffer = ExperienceReplayBuffer(hparams.er_buffer_size)
        replay_dataset_name = hparams.replay_dataset_name
        replay_dataset_dir = hparams.replay_dataset_dir
        assert replay_dataset_name and replay_dataset_dir and replay_dataset_name != hparams.dataset_name
        initialize_er_buffer(hparams, [(replay_dataset_name, replay_dataset_dir)], er_buffer)
        system = SC_DepthV3_ER(hparams, er_buffer)

    # pl data module
    dm = VideosDataModule(hparams)

    # pl logger
    logger = TensorBoardLogger(
        save_dir="ckpts",
        name=hparams.exp_name
    )

    # save checkpoints
    ckpt_dir = 'ckpts/{}/version_{:d}'.format(
        hparams.exp_name, logger.version)
    checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                          filename='{epoch}-{val_loss:.4f}',
                                          monitor='val_loss',
                                          mode='min',
                                          save_last=True,
                                          save_weights_only=True,
                                          save_top_k=3)

    # restore from previous checkpoints
    if hparams.ckpt_path is not None:
        print('load pre-trained model from {}'.format(hparams.ckpt_path))
        system = system.load_from_checkpoint(
            hparams.ckpt_path, strict=False, hparams=hparams)

    # set up trainer
    print('hparams.num_epochs', hparams.num_epochs)
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=hparams.num_epochs,
        limit_train_batches=hparams.epoch_size,
        limit_val_batches=200 if hparams.val_mode == 'photo' else 1.0,
        num_sanity_val_steps=5,
        callbacks=[checkpoint_callback],
        logger=logger,
        benchmark=True
    )

    trainer.fit(system, dm)
