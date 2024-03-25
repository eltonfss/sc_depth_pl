from typing import List, Any, Tuple

from er_buffer import ExperienceReplayBuffer
from sc_depth_data_module import SCDepthDataModule
from pytorch_lightning import LightningModule, Trainer


def initialize_er_buffer(hparams: Any, datasets: List[Tuple[str, str]], er_buffer: ExperienceReplayBuffer):
    for dataset_name, dataset_dir in datasets:
        sc_depth_params = hparams.copy()
        sc_depth_params.dataset_name = dataset_name
        sc_depth_params.dataset_ir = dataset_dir
        data_module = SCDepthDataModule(sc_depth_params)
        er_initializer = ExperienceReplayBufferInitializer(dataset_name, er_buffer)
        Trainer().fit(model=er_initializer, datamodule=data_module)


class ExperienceReplayBufferInitializer(LightningModule):

    def __init__(self, dataset_name, er_buffer: ExperienceReplayBuffer, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset_name = dataset_name
        self._er_buffer = er_buffer

    def training_step(self, batch, batch_idx):
        self._er_buffer.add_batch_to_buffer((self._dataset_name, batch_idx, batch))
