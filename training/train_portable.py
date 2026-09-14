"""Run unchanged benchmark learning code without shared-memory batch transport.

Workers return NumPy arrays through the multiprocessing queue. Tensor conversion
and pinning occur in the parent, avoiding /dev/shm while retaining four workers,
original sampling order, worker seeds and batch size.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from training import train_benchmark as benchmark


def collate_numpy(samples):
    batch = {'image': np.stack([s['image'] for s in samples]),
             'image_id': [s['image_id'] for s in samples]}
    if 'label' in samples[0]:
        batch['label'] = np.asarray([s['label'] for s in samples], dtype=np.int64)
    return batch


def initialize_worker(worker_id):
    benchmark.seed_worker(worker_id)
    import cv2
    cv2.setNumThreads(1)


class NumpyTransportLoader(DataLoader):
    def __iter__(self):
        for batch in super().__iter__():
            for key in ('image', 'label'):
                if key in batch:
                    value = torch.from_numpy(batch[key])
                    batch[key] = value.pin_memory() if torch.cuda.is_available() else value
            yield batch


def make_loader(dataset, *, batch_size, shuffle, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return NumpyTransportLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=False, persistent_workers=workers > 0,
        worker_init_fn=initialize_worker, generator=generator,
        collate_fn=collate_numpy,
    )


if __name__ == '__main__':
    benchmark.make_loader = make_loader
    benchmark.main()
