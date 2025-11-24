"""
Example training script for GVPO
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gvpo_trainer import GVPOTrainer
from typing import List, Tuple


def dummy_reward_function(prompt: str, completion: str) -> float:
    """
    Placeholder reward function - replace with actual reward model
    For math problems, this would be a verifier checking correctness
    """
    # Simple heuristic: reward longer, more detailed responses
    return len(completion.split()) / 100.0


def prepare_training_data() -> List[Tuple[str, List[str], List[float]]]:
    """
    Prepare training data in format: (prompt, [completions], [rewards])
    In practice, load from dataset
    """
    # Example math prompts
    prompts = [
        "Solve: What is 15 * 23?",
        "Find the derivative of f(x) = x^2 + 3x + 2",
        "Calculate the area of a circle with radius 5"
    ]
    
    return [(p, [], []) for p in prompts]  # Completions generated during training


def main():
    # Configuration
    model_name = "gpt2"  # Replace with your model
    beta = 0.1
    num_samples = 4
    num_epochs = 3
    batch_size = 2
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model and reference model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Initialize GVPO trainer
    trainer = GVPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        beta=beta,
        num_samples_per_prompt=num_samples,
        learning_rate=1e-6
    )
    
    print(f"\nGVPO Training Configuration:")
    print(f"  Beta (β): {beta}")
    print(f"  Samples per prompt (k): {num_samples}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}\n")
    
    # Training loop
    training_data = prepare_training_data()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_metrics = {"loss": [], "kl": [], "advantage": []}
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i + batch_size]
            prompts = [item[0] for item in batch]
            
            # Generate completions
            print(f"  Generating {num_samples} completions per prompt...")
            completions = trainer.generate_completions(prompts)
            
            # Compute rewards
            print(f"  Computing rewards...")
            rewards = []
            for prompt, comps in zip(prompts, completions):
                prompt_rewards = [dummy_reward_function(prompt, c) for c in comps]
                rewards.append(prompt_rewards)
            
            # Training step
            print(f"  Training step...")
            metrics = trainer.train_step(prompts, completions, rewards)
            
            epoch_metrics["loss"].append(metrics["loss"])
            epoch_metrics["kl"].append(metrics["mean_kl"])
            epoch_metrics["advantage"].append(metrics["mean_advantage"])
            
            print(f"    Loss: {metrics['loss']:.4f} | "
                  f"KL: {metrics['mean_kl']:.4f} | "
                  f"Advantage: {metrics['mean_advantage']:.4f}")
        
        # Epoch summary
        avg_loss = sum(epoch_metrics["loss"]) / len(epoch_metrics["loss"])
        avg_kl = sum(epoch_metrics["kl"]) / len(epoch_metrics["kl"])
        print(f"  Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, KL={avg_kl:.4f}\n")
    
    # Save model
    output_dir = "./gvpo_model"
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
