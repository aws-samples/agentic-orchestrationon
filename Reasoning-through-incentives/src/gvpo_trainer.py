"""
GVPO (Group Variance Policy Optimization) Trainer
Based on the paper's mathematical formulation and TRL's architecture
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class GVPOTrainer:
    """
    GVPO Trainer implementing the core algorithm:
    
    Key Formula:
    w_i = (R(x,y_i) - R̄) - β(log(π_θ/π_θ') - log(π_θ/π_θ')̄)
    Loss = -β * Σ w_i * log π_θ(y_i|x) / (k-1)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        beta: float = 0.1,
        num_samples_per_prompt: int = 4,
        max_length: int = 512,
        learning_rate: float = 1e-6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.ref_model = ref_model.to(device)
        self.ref_model.eval()  # Reference model stays frozen
        
        self.tokenizer = tokenizer
        self.beta = beta
        self.k = num_samples_per_prompt
        self.max_length = max_length
        self.device = device
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
    def compute_log_probs(
        self, 
        model: PreTrainedModel, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for sequences"""
        with torch.no_grad() if model == self.ref_model else torch.enable_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute log probs
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Mask padding tokens
            mask = attention_mask[..., 1:].contiguous()
            token_log_probs = token_log_probs * mask
            
            # Sum over sequence length
            sequence_log_probs = token_log_probs.sum(dim=-1)
            
        return sequence_log_probs
    
    def compute_gvpo_loss(
        self,
        prompts: List[str],
        completions: List[List[str]],
        rewards: List[List[float]]
    ) -> Dict[str, torch.Tensor]:
        """
        Core GVPO loss computation
        
        Args:
            prompts: List of prompts [batch_size]
            completions: List of k completions per prompt [batch_size, k]
            rewards: List of k rewards per prompt [batch_size, k]
        """
        batch_size = len(prompts)
        total_loss = 0.0
        stats = {"loss": [], "advantages": [], "kl_div": []}
        
        for i in range(batch_size):
            prompt = prompts[i]
            k_completions = completions[i]
            k_rewards = torch.tensor(rewards[i], device=self.device)
            
            # Tokenize prompt + completions
            full_texts = [prompt + comp for comp in k_completions]
            encodings = self.tokenizer(
                full_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Compute log probs from current and reference models
            log_probs_new = self.compute_log_probs(
                self.model, 
                encodings.input_ids, 
                encodings.attention_mask
            )
            log_probs_old = self.compute_log_probs(
                self.ref_model,
                encodings.input_ids,
                encodings.attention_mask
            )
            
            # Compute log ratio: log(π_θ/π_θ')
            log_ratio = log_probs_new - log_probs_old
            
            # GVPO advantage computation (key difference from GRPO)
            # w_i = (R_i - R̄) - β((log_ratio_i - log_ratio_mean))
            reward_centered = k_rewards - k_rewards.mean()
            log_ratio_centered = log_ratio - log_ratio.mean()
            
            advantages = reward_centered - self.beta * log_ratio_centered
            
            # GVPO loss with Bessel correction (k-1)
            # Loss = -β * Σ w_i * log π_θ(y_i|x) / (k-1)
            loss = -self.beta * (advantages * log_probs_new).sum() / (self.k - 1)
            
            total_loss += loss
            
            # Track statistics
            stats["loss"].append(loss.item())
            stats["advantages"].append(advantages.mean().item())
            stats["kl_div"].append(log_ratio.mean().item())
        
        avg_loss = total_loss / batch_size
        
        return {
            "loss": avg_loss,
            "mean_advantage": torch.tensor(stats["advantages"]).mean(),
            "mean_kl": torch.tensor(stats["kl_div"]).mean()
        }
    
    def train_step(
        self,
        prompts: List[str],
        completions: List[List[str]],
        rewards: List[List[float]]
    ) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        metrics = self.compute_gvpo_loss(prompts, completions, rewards)
        loss = metrics["loss"]
        
        loss.backward()
        self.optimizer.step()
        
        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
    
    def generate_completions(
        self,
        prompts: List[str],
        num_return_sequences: Optional[int] = None
    ) -> List[List[str]]:
        """Generate k completions per prompt"""
        if num_return_sequences is None:
            num_return_sequences = self.k
            
        self.model.eval()
        all_completions = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                completions = [
                    self.tokenizer.decode(out[inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    for out in outputs
                ]
                all_completions.append(completions)
        
        return all_completions
