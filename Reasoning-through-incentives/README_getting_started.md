# GVPO Implementation

Minimal implementation of **Group Variance Policy Optimization (GVPO)** for LLM post-training.

## Key Algorithm

GVPO improves upon GRPO by incorporating KL constraints directly into gradient weights:

```python
# GRPO (baseline)
advantages = (rewards - rewards.mean()) / rewards.std()

# GVPO (this implementation)
advantages = (rewards - rewards.mean()) - beta * (log_ratio - log_ratio.mean())
loss = -beta * (advantages * log_probs).sum() / (k - 1)
```

## Mathematical Foundation

**Weight Formula:**
```
w_i = (R(x,y_i) - R̄) - β(log(π_θ/π_θ') - log(π_θ/π_θ')̄)
```

**Loss Function:**
```
L = -β * Σ w_i * log π_θ(y_i|x) / (k-1)
```

The `(k-1)` factor is the Bessel correction for unbiased variance estimation.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from gvpo_trainer import GVPOTrainer

# Load models
model = AutoModelForCausalLM.from_pretrained("your-model")
ref_model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Initialize trainer
trainer = GVPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    beta=0.1,  # KL constraint coefficient
    num_samples_per_prompt=4
)

# Training step
prompts = ["Solve: 2 + 2 = ?"]
completions = [["4", "Four", "2+2=4", "The answer is 4"]]
rewards = [[1.0, 0.8, 0.9, 0.95]]

metrics = trainer.train_step(prompts, completions, rewards)
```

### Run Example

```bash
python train_gvpo.py
```

## Key Differences from GRPO

| Feature | GRPO | GVPO |
|---------|------|------|
| **Advantage** | `(R - R̄) / σ_R` | `(R - R̄) - β(log π_θ/π_θ' - mean)` |
| **KL Constraint** | External penalty | Built into weights |
| **Normalization** | Std division | Only centering |
| **Stability** | Requires clipping | Inherently stable |

## Hyperparameters

- **beta (β)**: KL constraint strength (default: 0.1)
  - Paper shows robustness across [0.01, 0.5]
  - Lower β = more exploration
  - Higher β = stay closer to reference policy

- **num_samples_per_prompt (k)**: Responses per prompt (default: 4)
  - Paper tests k ∈ [2, 32]
  - Higher k improves performance but increases compute

## Reward Function

Replace `dummy_reward_function` in `train_gvpo.py` with your actual reward model:

```python
def reward_function(prompt: str, completion: str) -> float:
    # For math: use verifier to check correctness
    # For general: use reward model (e.g., from RLHF)
    return score
```

## Architecture

```
gvpo_trainer.py      # Core GVPO algorithm
train_gvpo.py        # Training script
requirements.txt     # Dependencies
README.md           # Documentation
```

## Citation

Based on the GVPO paper's mathematical formulation:
- Zero-sum weight constraint eliminates partition function
- No importance sampling needed (unlike PPO/GRPO)
- Theoretical guarantee of convergence to optimal policy

## Performance

Paper results on Qwen2.5-Math-7B:

| Benchmark | GRPO | GVPO | Improvement |
|-----------|------|------|-------------|
| AIME2024 | 14.79 | 20.72 | +40% |
| AMC | 55.42 | 62.65 | +13% |
| MATH500 | 80.00 | 83.80 | +5% |

## License

MIT
