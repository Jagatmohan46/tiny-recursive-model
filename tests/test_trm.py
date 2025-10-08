import pytest

import torch

def test_trm():
    from tiny_recursive_model.trm import TinyRecursiveModel
    from tiny_recursive_model.mlp_mixer_1d import MLPMixer1D
    from torch.optim import AdamW

    network = MLPMixer1D(dim = 512, depth = 2, seq_len = 1024)

    trm = TinyRecursiveModel(
        dim = 512,
        num_tokens = 256,
        network = network
    )

    optim = AdamW(trm.parameters(), lr = 1e-4)

    seq = torch.randint(0, 256, (2, 1024))
    answer = torch.randint(0, 256, (2, 1024))

    outputs, latents = trm.get_initial()

    for _ in range(3):
        loss, losses, outputs, latents, pred, halt = trm(seq, outputs, latents, labels = answer)

        loss.backward()
        optim.step()
        optim.zero_grad()
