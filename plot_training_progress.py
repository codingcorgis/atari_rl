#!/usr/bin/env python3
"""
Script to plot reward/timestep for training progress in SpaceInvaders PPO training.
This script shows training rewards over time using available TensorBoard data.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def load_tensorboard_data(log_dir="./logs/tensorboard_logs"):
    """
    Load data from TensorBoard logs.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        
    Returns:
        dict: Dictionary containing all training metrics
    """
    print(f"Loading TensorBoard data from: {log_dir}")
    
    # Find the most recent log directory
    log_dirs = glob.glob(os.path.join(log_dir, "*"))
    if not log_dirs:
        print(f"No TensorBoard logs found in {log_dir}")
        return None
    
    # Get the most recent log directory
    latest_log = max(log_dirs, key=os.path.getctime)
    print(f"Using log directory: {latest_log}")
    
    # Load the event accumulator
    ea = EventAccumulator(latest_log)
    ea.Reload()
    
    # Get all available tags
    tags = ea.Tags()
    print(f"Available tags: {tags}")
    
    data = {}
    
    # Load scalar data
    for tag in tags['scalars']:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data

def plot_reward_progress(data, save_path="./training_plots"):
    """
    Plot reward/timestep for training progress.
    
    Args:
        data: Dictionary containing training metrics
        save_path: Directory to save plots
    """
    if data is None:
        print("No data to plot")
        return
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    fig_width = 14
    fig_height = 10
    
    # Create a comprehensive plot showing all available reward-related metrics
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
    fig.suptitle('Training Progress - SpaceInvaders PPO', fontsize=16, fontweight='bold')
    
    # 1. Training Episode Rewards
    if 'rollout/ep_rew_mean' in data:
        ax = axes[0, 0]
        steps = data['rollout/ep_rew_mean']['steps']
        values = data['rollout/ep_rew_mean']['values']
        
        ax.plot(steps, values, 'b-', linewidth=3, alpha=0.8, label='Training Reward')
        ax.set_title('Training Episode Rewards')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add statistics
        if values:
            mean_reward = np.mean(values)
            final_reward = values[-1]
            improvement = values[-1] - values[0]
            ax.text(0.02, 0.98, f'Mean: {mean_reward:.2f}\nFinal: {final_reward:.2f}\nImprovement: {improvement:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Evaluation Rewards (if available)
    if 'eval/mean_reward' in data:
        ax = axes[0, 1]
        steps = data['eval/mean_reward']['steps']
        values = data['eval/mean_reward']['values']
        
        ax.plot(steps, values, 'r-', linewidth=3, alpha=0.8, label='Evaluation Reward')
        ax.set_title('Evaluation Rewards')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Evaluation Reward')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add statistics
        if values:
            mean_reward = np.mean(values)
            final_reward = values[-1]
            improvement = values[-1] - values[0]
            ax.text(0.02, 0.98, f'Mean: {mean_reward:.2f}\nFinal: {final_reward:.2f}\nImprovement: {improvement:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Episode Lengths
    if 'rollout/ep_len_mean' in data:
        ax = axes[1, 0]
        steps = data['rollout/ep_len_mean']['steps']
        values = data['rollout/ep_len_mean']['values']
        
        ax.plot(steps, values, 'g-', linewidth=3, alpha=0.8, label='Episode Length')
        ax.set_title('Episode Lengths')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Length')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if values:
            mean_length = np.mean(values)
            final_length = values[-1]
            ax.text(0.02, 0.98, f'Mean: {mean_length:.1f}\nFinal: {final_length:.1f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Combined Training vs Evaluation
    ax = axes[1, 1]
    if 'rollout/ep_rew_mean' in data:
        steps = data['rollout/ep_rew_mean']['steps']
        values = data['rollout/ep_rew_mean']['values']
        ax.plot(steps, values, 'b-', linewidth=3, alpha=0.8, label='Training')
    
    if 'eval/mean_reward' in data:
        eval_steps = data['eval/mean_reward']['steps']
        eval_values = data['eval/mean_reward']['values']
        ax.plot(eval_steps, eval_values, 'r-', linewidth=3, alpha=0.8, label='Evaluation')
    
    ax.set_title('Training vs Evaluation')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Reward')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_progress.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed reward progression plot
    if 'rollout/ep_rew_mean' in data:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height//2))
        fig.suptitle('Detailed Reward Progression - SpaceInvaders PPO', fontsize=16, fontweight='bold')
        
        steps = data['rollout/ep_rew_mean']['steps']
        values = data['rollout/ep_rew_mean']['values']
        
        # Plot the main reward line
        ax.plot(steps, values, 'b-', linewidth=3, alpha=0.8, label='Training Reward')
        
        # Add markers for key points
        ax.scatter(steps[0], values[0], color='red', s=100, zorder=5, label=f'Start: {values[0]:.2f}')
        ax.scatter(steps[-1], values[-1], color='green', s=100, zorder=5, label=f'End: {values[-1]:.2f}')
        
        # Find and mark the best point
        best_idx = np.argmax(values)
        ax.scatter(steps[best_idx], values[best_idx], color='orange', s=100, zorder=5, 
                  label=f'Best: {values[best_idx]:.2f}')
        
        # Add trend line
        if len(values) > 1:
            z = np.polyfit(steps, values, 1)
            p = np.poly1d(z)
            ax.plot(steps, p(steps), '--', color='gray', alpha=0.7, label=f'Trend')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Mean Episode Reward')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.legend()
        
        # Add summary statistics
        improvement = values[-1] - values[0]
        total_improvement = max(values) - min(values)
        ax.text(0.02, 0.98, f'Total Improvement: {improvement:.2f}\nBest Improvement: {total_improvement:.2f}\nFinal Reward: {values[-1]:.2f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'detailed_reward_progression.png'), dpi=300, bbox_inches='tight')
        plt.show()

def print_training_summary(data):
    """
    Print a summary of training progress.
    
    Args:
        data: Dictionary containing training metrics
    """
    if data is None:
        print("No data to summarize")
        return
    
    print("\n" + "="*60)
    print("TRAINING PROGRESS SUMMARY")
    print("="*60)
    
    # Training rewards
    if 'rollout/ep_rew_mean' in data:
        values = data['rollout/ep_rew_mean']['values']
        steps = data['rollout/ep_rew_mean']['steps']
        
        print(f"Training Episode Rewards:")
        print(f"  Initial reward: {values[0]:.2f}")
        print(f"  Final reward: {values[-1]:.2f}")
        print(f"  Best reward: {max(values):.2f}")
        print(f"  Worst reward: {min(values):.2f}")
        print(f"  Average reward: {np.mean(values):.2f}")
        print(f"  Total improvement: {values[-1] - values[0]:.2f}")
        print(f"  Training steps: {steps[-1]:,}")
        print(f"  Data points: {len(values)}")
    
    # Evaluation rewards
    if 'eval/mean_reward' in data:
        values = data['eval/mean_reward']['values']
        steps = data['eval/mean_reward']['steps']
        
        print(f"\nEvaluation Rewards:")
        print(f"  Initial reward: {values[0]:.2f}")
        print(f"  Final reward: {values[-1]:.2f}")
        print(f"  Best reward: {max(values):.2f}")
        print(f"  Worst reward: {min(values):.2f}")
        print(f"  Average reward: {np.mean(values):.2f}")
        print(f"  Total improvement: {values[-1] - values[0]:.2f}")
        print(f"  Evaluation steps: {steps[-1]:,}")
    
    # Episode lengths
    if 'rollout/ep_len_mean' in data:
        values = data['rollout/ep_len_mean']['values']
        
        print(f"\nEpisode Lengths:")
        print(f"  Initial length: {values[0]:.1f}")
        print(f"  Final length: {values[-1]:.1f}")
        print(f"  Average length: {np.mean(values):.1f}")
        print(f"  Min length: {min(values):.1f}")
        print(f"  Max length: {max(values):.1f}")
    
    print("="*60)

def main():
    """Main function to load and plot training progress."""
    print("Loading training data and plotting reward progress...")
    
    # Load TensorBoard data
    data = load_tensorboard_data()
    
    if data is None:
        print("No training data found. Make sure you have run training first.")
        return
    
    # Print summary
    print_training_summary(data)
    
    # Create plots
    plot_reward_progress(data)
    
    print("\nPlots saved to ./training_plots/")
    print("Training progress analysis complete!")

if __name__ == "__main__":
    main() 