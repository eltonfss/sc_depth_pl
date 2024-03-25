from er_buffer import ExperienceReplayBuffer
from SC_DepthV3 import SC_DepthV3


class SC_DepthV3_ER(SC_DepthV3):

    def __init__(self, hparams, er_buffer: ExperienceReplayBuffer):
        super(SC_DepthV3_ER, self).__init__(hparams)
        self._er_buffer = er_buffer
        self._er_size = hparams.er_size
        self._er_frequency = hparams.er_frequency
        self._dataset_name = self.hparams.hparams.dataset_name
        self._batch_count = 0

    def training_step(self, batch, batch_idx):

        # replay batches from buffer
        if self._batch_count > 0 and self._batch_count % self._er_frequency == 0:
            er_batches_info = self._er_buffer.get_batches_from_buffer(self._er_size)
            for (dataset, er_batch_idx, er_batch) in er_batches_info:
                super(SC_DepthV3_ER, self).training_step(er_batch, er_batch_idx)

        # train with current batch and add it to buffer
        super(SC_DepthV3_ER, self).training_step(batch, batch_idx)
        self._er_buffer.add_batch_to_buffer((self._dataset_name, batch_idx, batch))
        self._batch_count += 1
