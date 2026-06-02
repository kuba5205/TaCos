# TaCos
TaCos is a lightweight, highly stable Transformer-based language model implemented in PyTorch.

---


This script may be freely tested, used, and modified for private purposes. Commercial use is expressly prohibited (based on CC BY-NC 4.0).

---


# TaCos: Tau-Cosine Attention Language Model

TaCos is a lightweight, highly stable Transformer-based language model implemented in PyTorch. It replaces standard dot-product attention with a **Tau-Cosine Attention** mechanism, normalizing query and key vectors to a unit hypersphere. This approach eliminates gradient explosion, stabilizes early training, and provides explicit control over the model's "focus" via a temperature parameter ($\tau$).

## 🚀 Key Features

* **Tau-Cosine Attention:** Computes bounded similarity scores $[-1, 1]$ rather than unbounded dot products, leading to much smoother convergence.
* **Temperature Control ($\tau$):** The `TAU_TEMPERATUR` hyperparameter acts as a strictness valve. Lower values create a "laser focus" on specific token relationships, while higher values flood the model with broader context.
* **Highly Stable:** Immune to standard Transformer activation spikes. Handles higher learning rates (e.g., $5\times 10^{-4}$) natively without complex warmup schedules.
* **GELU Activations:** Replaces standard ReLU in the FeedForward layers for a smoother gradient landscape.
* **Hardware Agnostic:** Automatically detects and scales to CUDA GPUs, with a seamless fallback to CPU for local testing.

## 🧠 The Math: Why it works

In standard attention, raw magnitudes dictate the scores:


$$\text{Attention} = \text{softmax}(Q K^T) V$$

If vectors grow too large, the softmax function flatlines, killing gradients. TaCos solves this by applying $L_2$ normalization across the channel dimension, forcing all vectors onto a unit hypersphere. The raw dot product becomes pure cosine similarity, bounded between $-1$ and $1$.

We then apply the Temperature ($\tau$) to scale the angular boundaries before softmax:


$$\text{TauCos Attention} = \text{softmax}\left(\frac{\frac{Q}{\|Q\|_2} \cdot \left(\frac{K}{\|K\|_2}\right)^T}{\tau}\right) V$$

## 📦 Installation

Ensure you have Python installed, then install the required dependencies:

```bash
pip install torch tqdm matplotlib

```

## 🛠️ Usage

**1. Prepare your training data:**
Place a plain text file containing your training corpus at the path defined in the script, or update the `file_path` variable:

```python
file_path = '/path/to/your/input.txt'

```

**2. Run the training loop:**

```bash
python taucosnet.py

```

The script will automatically:

* Process the text and build a character-level vocabulary.
* Initialize the model and check for existing checkpoints in the `tau_checkpoints/` directory.
* Begin training, displaying a live `matplotlib` graph of the training and validation loss.
* Output periodic text generation snippets so you can observe the model learning the syntax of your dataset in real-time.

## ⚙️ Hyperparameters

You can adjust the core behavior directly in the script:

* `TAU_TEMPERATUR` (Default: `0.1`): Lower = strict k-NN style matching; Higher = looser, creative context matching.
* `BATCH_SIZE` (Default: `128`): Stabilized for memory alignment.
* `BLOCK_SIZE` (Default: `256`): The context window (how far back the model looks).
* `EMBED_DIM` (Default: `384`): The embedding dimension size.
* `NUM_HEADS` (Default: `6`): Number of parallel attention heads.

## 📂 Ignoring Checkpoints in Git

If you are tracking your experiments with Git, ensure you add the following to your `.gitignore` file to prevent committing massive tensor files:

```text
# Ignore Checkpoints
*.pth
tau_checkpoints/

# Ignore Datasets
input.txt

# Python Cache
__pycache__/

```
