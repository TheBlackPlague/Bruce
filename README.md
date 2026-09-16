<h1 align="center">
    🦈 Bruce
</h1>

<h3 align="center">
    Neural Network Trainer Interface for StockDory
</h3>

### 🌐 Overview

* 🧠 High-Level Neural Network Training
* 🚀 GPU Accelerated
* 📊 TensorBoard Graphs
* 💾 Reliable Checkpointing and Resume
* 🔄 Multiple Dataset Formats
* ⚙️ Simple TOML Configuration
* 👌 Free and Open Source

Bruce is a modern Rust interface intended for training neural networks to be used inside the 
[StockDory](https://github.com/TheBlackPlague/StockDory) chess engine. It is designed to act as a tiny frontend layer on
top of the existing [Bullet](https://github.com/jw1912/bullet) machinery, making it easier to configure, monitor, and 
reproduce training runs. Additionally, Bruce is able to handle checkpoint creation and continuation of a training run 
from a previously saved checkpoint.

### 🎮 Building

> [!NOTE]
> Bruce is primarily a GPU-accelerated neural-network training interface. As such, a supported GPU backend and its 
> corresponding development toolkit are required for training. By default, Bruce uses and recommends the CUDA backend.

The recommended build environment is:

* 👽 Git >= 2.30
* 🦀 Rust >= 1.88
* 🟢 NVIDIA CUDA Toolkit

Clone and build Bruce with:

```bash
# Clone the Bruce repository.
git clone https://github.com/TheBlackPlague/Bruce.git
cd Bruce

# Build an optimized CUDA version.
cargo build --release --locked
```

If CUDA is not automatically discoverable, set `CUDA_PATH` to the CUDA Toolkit installation before building:

```bash
export CUDA_PATH=/usr/local/cuda
cargo build --release --locked
```

> [!IMPORTANT]
> Bruce additionally exposes Bullet's ROCm and Metal but they can differ in platform support and hardware validation.
> You may compile them as so:
> ```aiignore
> # AMD ROCm
> cargo build --release --locked --no-default-features --features rocm
> 
> # Apple Metal
> cargo build --release --locked --no-default-features --features metal
> ```

A GPU-independent build can also be produced for operations which do not require training (such as data validation & 
conversion):

```bash
cargo build --release --locked --no-default-features
```

### 🧠 Training

Bruce training is controlled through a TOML configuration file. Before beginning a run, the configuration and dataset 
can be checked with:

```bash
./target/release/bruce check --config config/training.toml
```

Training can then be started with:

```bash
./target/release/bruce train --config config/training.toml
```

During training, Bruce provides a live terminal interface showing the current training state and important metrics. 
Depending on the configuration, the training state is periodically stored in checkpoints containing as much information
to reliably resume a run.

A previous checkpoint can be resumed with:

```bash
./target/release/bruce train --config config/training.toml \
    --resume checkpoints/<checkpoint>
```

Completed networks are exported directly into the `.nnue` format expected by MantaRay.

### 🔄 Dataset Conversion

Bruce supports datasets stored in several formats understood by Bullet and related chess-engine tooling. Datasets can be
converted using Bruce's `convert` command:

```bash
./target/release/bruce convert \
    --from sf \
    --to bullet \
    --input data.binpack \
    --output training.bullet
```

This allows for datasets originating from various different engines to be brought into a common format. Depending on the
task at hand, this can be super beneficial and allow you to combine a lot of data into a single training run.

### ⚙️ Configuration

Training behavior is defined entirely through TOML configuration.

More advanced schedules can be composed without restarting training, allowing multiphase training plans to remain part 
of a single reproducible run.

See the provided [`config`](config) directory for an example configuration.

### 📑 Terms of Use

Bruce is licensed under the [GNU AGPL v3.0](LICENSE).

[Bullet](https://github.com/jw1912/bullet) and other libraries/frameworks used by Bruce are developed separately by 
their respective contributors and are subject to their own licensing terms.
