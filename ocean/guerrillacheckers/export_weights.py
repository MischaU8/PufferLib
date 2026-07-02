#!/usr/bin/env python3
"""Export a guerrillacheckers torch checkpoint to puffernet flat weights.

The demo client (guerrillacheckers.c) rebuilds the default policy from
config/guerrillacheckers.ini -- Affine encoder -> MinGRU x num_layers ->
Affine decoder -- and reads tensors from the flat file in exactly this order:

    encoder.encoder.weight   (hidden, obs)
    encoder.encoder.bias     (hidden,)
    decoder.decoder.weight   (actions, hidden)
    decoder.decoder.bias     (actions,)
    network.layers.<i>.weight (3*hidden, hidden) for i in 0..num_layers-1

The value head is not exported; the client only needs action logits.

Usage:
    python ocean/guerrillacheckers/export_weights.py CHECKPOINT.bin [OUT.bin]
"""
import sys

import numpy as np
import torch

OBS_SIZE = 120
NUM_ACTIONS = 256
DEFAULT_OUT = 'resources/guerrillacheckers/guerrillacheckers_weights.bin'


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    checkpoint = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    sd = torch.load(checkpoint, map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in sd.items()}

    gru_keys = sorted(k for k in sd if k.startswith('network.layers.'))
    order = [
        'encoder.encoder.weight',
        'encoder.encoder.bias',
        'decoder.decoder.weight',
        'decoder.decoder.bias',
    ] + gru_keys

    hidden = sd['encoder.encoder.weight'].shape[0]
    assert sd['encoder.encoder.weight'].shape == (hidden, OBS_SIZE)
    assert sd['decoder.decoder.weight'].shape == (NUM_ACTIONS, hidden)
    for k in gru_keys:
        assert sd[k].shape == (3 * hidden, hidden), (k, sd[k].shape)

    flat = np.concatenate([
        sd[k].detach().numpy().astype(np.float32).ravel() for k in order
    ])
    flat.tofile(out)
    print(f'{out}: {flat.size} floats ({flat.nbytes} bytes), '
        f'hidden={hidden}, mingru_layers={len(gru_keys)}')


if __name__ == '__main__':
    main()
