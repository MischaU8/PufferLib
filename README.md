![figure](https://pufferai.github.io/source/resource/header.png)

[![Discord](https://dcbadge.vercel.app/api/server/spT4huaGYV?style=plastic)](https://discord.gg/spT4huaGYV)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez)](https://twitter.com/jsuarez)

PufferLib is a fast and sane reinforcement learning library that can train tiny, super-human models in seconds. The included learning algorithm, hyperparameter tuning, and simulation methods are the product of our own research. All our tools are free and open source. Need a high performance environment for your application? We build them professionally and offer training + extended support. Contact jsuarez🐡puffer🐡ai.

All of our documentation is hosted at [puffer.ai](https://puffer.ai "PufferLib Documentation"). @jsuarez5341 on [Discord](https://discord.gg/puffer) for support. Post there before opening issues. We're always looking for new contributors!

## macOS

On macOS, `build.sh` defaults to the CPU PyTorch backend because CUDA is not available. Install OpenMP first:

```bash
brew install libomp
```

Homebrew installs `libomp` as keg-only. `build.sh` detects the standard Homebrew paths automatically; for a nonstandard install, set `CPPFLAGS` and `LDFLAGS` to your `libomp` include and lib directories.

MPS is available as an opt-in torch device for policy and training tensors while keeping environment stepping on CPU:

```bash
PUFFER_TORCH_DEVICE=mps puffer train breakout
```

Use `PUFFER_TORCH_DEVICE=cpu` to force the CPU torch backend, or `PUFFER_TORCH_DEVICE=auto` to choose CUDA, MPS, then CPU in that order.

### Native Metal rollout hybrid

The Metal hybrid is an opt-in macOS-only path that keeps CPU as the default. It runs rollout policy forward and sampling through the native Metal backend scheduled by Python, then trains PPO with PyTorch tensors on MPS. On an M1 Max, this path is learning-verified and has measured about 3.1x PyTorch-MPS throughput.

The `python` rollout path is **shape-parametric**: it works for any environment using the default PufferLib policy (`DefaultEncoder -> N-layer MinGRU -> DefaultDecoder`), deriving obs/hidden sizes, the MinGRU layer count, and the action space from the env and policy at runtime (hidden up to 512, any num_layers >= 1). It supports discrete (1..8 heads), multi-discrete, and continuous (Gaussian) action spaces. Verified on Breakout (hidden=64, 2 layers, 3 actions), Pong (hidden=32, 1 layer), Chess (hidden=512, 3 layers, 97 actions), Minimal (multi-discrete `[9,5]`), and Drone (4 continuous dims). The `native` (CPU-env scheduler) path remains Breakout-specific.

Build the CPU extension and native Metal rollout dylib, then run:

```bash
bash build.sh breakout
bash build.sh breakout --metal-native
PUFFER_TORCH_DEVICE=cpu PUFFER_METAL_ROLLOUT=python PUFFER_METAL_TRAIN_DEVICE=mps puffer train breakout
```

Kept flags:

- `PUFFER_METAL_ROLLOUT=python|native`: opt in to the Metal rollout; `python` is the recommended (env-agnostic) hybrid, `native` is the Breakout-only scheduler retained for parity checks.
- `PUFFER_METAL_TRAIN_DEVICE=cpu|mps`: choose the PPO training tensor device for a Metal rollout; unset inherits the normal torch device.
- `PUFFER_TORCH_DEVICE=cpu|mps|cuda|auto`: choose the base PyTorch device; CPU remains the default on macOS.
- `PUFFER_METAL_DIAGNOSTICS=0|1`: enable detailed Metal/train timing and native rollout parity diagnostics.
- `PUFFER_TORCH_NUM_THREADS=<n>`: set PyTorch intra-op CPU threads for measurement and tuning.

Then the standard install test works without Docker or CUDA:

```bash
bash build.sh breakout
puffer train breakout
puffer eval breakout --load-model-path latest
```

### Known issues (environment compatibility)

Some Ocean environments do not build or run on the current tree. These are
upstream issues independent of this macOS/Metal work — the affected envs fail to
compile their `_C` extension, so they cannot train on **any** backend
(CPU/MPS/Metal):

- **~15 envs predate the 4.0 binding API** and still `#include "../env_binding.h"`
  (removed in PufferLib 4.0). They need migrating to the current `vecenv.h`
  `Dict*` binding: tracked in
  [PufferAI/PufferLib#542](https://github.com/PufferAI/PufferLib/issues/542)
  (asteroids, battle, boids, chain_mdp, checkers, convert_circle, impulse_wars,
  matsci, memory, onestateworld, onlyfish, shared_pool, tactical, template,
  tmaze). Migration is happening per-env, e.g.
  [PufferAI/PufferLib#545](https://github.com/PufferAI/PufferLib/pull/545)
  (impulse_wars). A few more (benchmark, blastar, convert, snake, whisker_racer)
  are partway migrated and still fail on `OBS_TENSOR_T`/`rng`.
- **Some env headers have additional C errors** beyond the binding (e.g.
  checkers' `c_render`):
  [PufferAI/PufferLib#324](https://github.com/PufferAI/PufferLib/issues/324).
- **x86-only AVX2/FMA flags break the Apple Silicon build** for some targets
  (e.g. craftax_classic); fix in flight:
  [PufferAI/PufferLib#590](https://github.com/PufferAI/PufferLib/pull/590).
- **Several configs carry Protein-sweep float `num_layers`** (e.g. cartpole),
  which would crash `MinGRU(**policy_kwargs)` via `range(float)` on any backend.
  This patch fixes it: `load_policy` rounds integer policy hyperparameters
  (`num_layers`, `hidden_size`) to match the sweep's own `is_integer` rounding,
  so these envs (cartpole, etc.) now train normally. Not yet tracked upstream.

The Metal rollout itself supports any environment using the default
`DefaultEncoder -> N-layer MinGRU -> DefaultDecoder` policy with discrete,
multi-discrete, or continuous action spaces; the limitations above are about
building the env, not the Metal path.

## Star to puff up the project!

<a href="https://star-history.com/#pufferai/pufferlib&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=pufferai/pufferlib&type=Date" />
 </picture>
</a>
