#!/usr/bin/env python3
"""
Analyze the trained SpaceInvaders policy to understand action distribution
and why the ship might not be moving side to side.
"""

import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import PPO
import os
from collections import Counter

def analyze_policy_actions(model_path="./logs/best_model/best_model.zip", num_episodes=5):
    """Analyze what actions the trained policy is taking."""
    print(f"Analyzing policy from: {model_path}")
    
    # Load the model
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model not found at {model_path}")
        return
    
    # Create environment
    env = gym.make('ALE/SpaceInvaders-v5')
    env = gym.wrappers.AtariPreprocessing(env, 
                                        frame_skip=1,
                                        grayscale_obs=True,
                                        scale_obs=True,
                                        terminal_on_life_loss=True)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    # Action names for better understanding
    action_names = {
        0: "NOOP",
        1: "FIRE", 
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE"
    }
    
    all_actions = []
    episode_rewards = []
    episode_lengths = []
    
    print(f"\nRunning {num_episodes} episodes to analyze policy behavior...")
    print("=" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        episode_actions = []
        total_reward = 0
        step_count = 0
        
        print(f"\nEpisode {episode + 1}:")
        
        while not (done or truncated) and step_count < 1000:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Convert action to integer if it's a numpy array
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
                
            episode_actions.append(action)
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            # Print first few actions to see what's happening
            if step_count <= 10:
                print(f"  Step {step_count}: Action {action} ({action_names[action]}) - Reward: {reward}")
        
        # Count actions in this episode
        action_counts = Counter(episode_actions)
        print(f"  Episode length: {step_count}")
        print(f"  Total reward: {total_reward}")
        print(f"  Action distribution:")
        for action, count in sorted(action_counts.items()):
            percentage = (count / len(episode_actions)) * 100
            print(f"    {action_names[action]}: {count} times ({percentage:.1f}%)")
        
        all_actions.extend(episode_actions)
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
    
    env.close()
    
    # Overall analysis
    print("\n" + "=" * 60)
    print("OVERALL ANALYSIS:")
    print("=" * 60)
    
    total_action_counts = Counter(all_actions)
    total_steps = len(all_actions)
    
    print(f"Total steps analyzed: {total_steps}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f}")
    print(f"Average episode reward: {np.mean(episode_rewards):.1f}")
    
    print(f"\nOverall action distribution:")
    for action, count in sorted(total_action_counts.items()):
        percentage = (count / total_steps) * 100
        print(f"  {action_names[action]}: {count} times ({percentage:.1f}%)")
    
    # Movement analysis
    movement_actions = [2, 3, 4, 5]  # RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
    movement_count = sum(total_action_counts[action] for action in movement_actions)
    movement_percentage = (movement_count / total_steps) * 100
    
    print(f"\nMovement Analysis:")
    print(f"  Movement actions (RIGHT/LEFT/RIGHTFIRE/LEFTFIRE): {movement_count} times ({movement_percentage:.1f}%)")
    print(f"  Stationary actions (NOOP/FIRE): {total_steps - movement_count} times ({100 - movement_percentage:.1f}%)")
    
    if movement_percentage < 10:
        print(f"\n⚠️  WARNING: Very low movement rate ({movement_percentage:.1f}%)")
        print("   This suggests the policy might be stuck in a local optimum")
        print("   Consider: increasing exploration, adjusting reward structure, or longer training")
    elif movement_percentage < 30:
        print(f"\n⚠️  CAUTION: Low movement rate ({movement_percentage:.1f}%)")
        print("   The policy might benefit from more movement exploration")
    else:
        print(f"\n✅ Good movement rate ({movement_percentage:.1f}%)")

def test_random_actions():
    """Test random actions to see what normal movement looks like."""
    print("\n" + "=" * 60)
    print("TESTING RANDOM ACTIONS FOR COMPARISON:")
    print("=" * 60)
    
    env = gym.make('ALE/SpaceInvaders-v5')
    env = gym.wrappers.AtariPreprocessing(env, 
                                        frame_skip=1,
                                        grayscale_obs=True,
                                        scale_obs=True,
                                        terminal_on_life_loss=True)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    
    action_names = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}
    
    obs, _ = env.reset()
    done, truncated = False, False
    step_count = 0
    actions = []
    
    print("Random action sequence (first 20 steps):")
    while not (done or truncated) and step_count < 20:
        action = env.action_space.sample()
        actions.append(action)
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {step_count + 1}: Action {action} ({action_names[action]}) - Reward: {reward}")
        step_count += 1
    
    env.close()
    
    action_counts = Counter(actions)
    print(f"\nRandom action distribution:")
    for action, count in sorted(action_counts.items()):
        percentage = (count / len(actions)) * 100
        print(f"  {action_names[action]}: {count} times ({percentage:.1f}%)")

if __name__ == "__main__":
    # Analyze the trained policy
    analyze_policy_actions()
    
    # Test random actions for comparison
    test_random_actions() 