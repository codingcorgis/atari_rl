#!/usr/bin/env python3
"""
PPO Training Script for SpaceInvaders
This script trains a PPO agent on the SpaceInvaders environment
with progress tracking and periodic video recording.
GPU/CUDA acceleration enabled for faster training.
Addresses delayed reward issue with better exploration and reward shaping.
"""

import os
import time
import numpy as np
import gymnasium as gym
import ale_py  # Import to register ALE environments
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env, make_atari_env
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv
)

import matplotlib.pyplot as plt
import torch
from custom_reward_wrapper import CustomRewardWrapper


# Set random seed for reproducibility
set_random_seed(42)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

from stable_baselines3.common.callbacks import BaseCallback

def linear_schedule(initial_value: float, final_value: float = 0.0):
    """
    Linear learning rate schedule.
    :param initial_value: The initial value.
    :param final_value: The final value (default 0.0).
    :return: A function that takes the remaining progress and returns the current value.
    """
    def func(progress_remaining: float) -> float:
        """
        Progress_remaining will be 1.0 at the beginning and 0.0 at the end of training.
        """
        return final_value + (initial_value - final_value) * progress_remaining
    return func

# CustomRewardWrapper is imported from custom_reward_wrapper.py

class ActionDiversityCallback(BaseCallback):
    """Callback to monitor and encourage action diversity."""
    
    def __init__(self, check_freq=5000, min_diversity=0.3, log_dir="./logs", verbose=0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.min_diversity = min_diversity
        self.log_dir = log_dir
        self.action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.total_actions = 0
        
        print("Action diversity monitoring initialized.")
        
    def _on_step(self):
        """Called after each training step."""
        return True
        
    def _on_rollout_end(self):
        """Called at the end of a rollout."""
        # Aggregate actions from the rollout buffer
        if self.model is not None and hasattr(self.model, 'rollout_buffer'):
            actions_in_rollout = self.model.rollout_buffer.actions.flatten()
            for action in actions_in_rollout:
                # Ensure action is an integer
                if isinstance(action, np.ndarray):
                    action = int(action.item())
                else:
                    action = int(action)
                if action in self.action_counts:
                    self.action_counts[action] += 1
            self.total_actions += len(actions_in_rollout)

        # Check action diversity every check_freq steps
        if self.num_timesteps % self.check_freq == 0 and self.num_timesteps > 0:
            self._check_action_diversity()
        
        return True
    
    def _check_action_diversity(self):
        """Check if the policy is using diverse actions."""
        if self.total_actions == 0:
            return
            
        # Calculate diversity (how many different actions are used)
        used_actions = sum(1 for count in self.action_counts.values() if count > 0)
        diversity = used_actions / 6.0  # 6 possible actions
        
        print(f"\nAction Diversity Check at {self.num_timesteps} steps:")
        print(f"  Used actions: {used_actions}/6")
        print(f"  Diversity: {diversity:.2f}")
        
        # Print action distribution
        for action, count in self.action_counts.items():
            if count > 0:
                percentage = (count / self.total_actions) * 100
                action_names = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}
                print(f"    {action_names[action]}: {percentage:.1f}%")
        
        if diversity < self.min_diversity:
            print(f"⚠️  WARNING: Low action diversity ({diversity:.2f})")
            print("   Consider increasing exploration or adjusting training parameters")

class VideoRecorderCallback(BaseCallback):
    """Custom callback to record videos during training."""
    
    def __init__(self, episode_freq=25, video_length=0, log_dir="./logs", verbose=0):
        super().__init__(verbose)
        self.episode_freq = episode_freq
        self.video_length = video_length
        self.log_dir = log_dir
        self.video_count = 0
        self.last_episode_count = 0
        
        # Create video directory
        os.makedirs(os.path.join(log_dir, "videos"), exist_ok=True)
        
        print(f"Video recording initialized. Videos will be saved every {episode_freq} episodes.")
        
    def _on_step(self):
        """Called after each training step."""
        return True
        
    def _on_rollout_end(self):
        """Called at the end of a rollout."""
        # Get current episode count from the training environment
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            first_env = self.training_env.envs[0]
            if hasattr(first_env, 'get_episode_rewards'):
                episode_rewards = first_env.get_episode_rewards()
                current_episode_count = len(episode_rewards)
                
                # Check if we should record a video (every episode_freq episodes)
                if (current_episode_count >= self.episode_freq and 
                    current_episode_count // self.episode_freq > self.last_episode_count // self.episode_freq):
                    self._record_video()
                    self.last_episode_count = current_episode_count
        
        return True
    
    def _record_video(self):
        """Record a video of the current policy until episode ends."""
        try:
            # Get current episode count for the video filename
            current_episode_count = 0
            if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
                first_env = self.training_env.envs[0]
                if hasattr(first_env, 'get_episode_rewards'):
                    episode_rewards = first_env.get_episode_rewards()
                    current_episode_count = len(episode_rewards)
            
            print(f"\nRecording video at episode {current_episode_count} ({self.num_timesteps} steps)...")
            
            # Create a temporary environment for video recording that matches training environment exactly
            # Create the base environment with render mode for video capture
            env = gym.make('ALE/SpaceInvaders-v5', 
                         repeat_action_probability=0.25,
                         render_mode='rgb_array')
            
            # Apply the same wrappers as training environment exactly
            # Apply standard Atari wrappers explicitly
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            
            # Apply Atari preprocessing explicitly
            env = gym.wrappers.AtariPreprocessing(
                env,
                frame_skip=1,  # Already handled by MaxAndSkipEnv
                screen_size=84,
                grayscale_obs=True,
                scale_obs=True,
                terminal_on_life_loss=False  # Already handled by EpisodicLifeEnv
            )
            
            # Apply custom reward wrapper
            env = CustomRewardWrapper(env)
            
            # Apply frame stacking
            env = gym.wrappers.FrameStackObservation(env, stack_size=6)
            
            # Ensure observation compatibility with CnnPolicy
            if env.observation_space.shape[-1] == 1:
                class SqueezeWrapper(gym.ObservationWrapper):
                    def __init__(self, env):
                        super().__init__(env)
                        old_shape = env.observation_space.shape
                        new_shape = old_shape[:-1]
                        self.observation_space = gym.spaces.Box(
                            low=env.observation_space.low.reshape(new_shape),
                            high=env.observation_space.high.reshape(new_shape),
                            dtype=env.observation_space.dtype
                        )
                    
                    def observation(self, obs):
                        return obs.squeeze(-1)
                
                env = SqueezeWrapper(env)
            
            # Record video
            video_path = os.path.join(self.log_dir, "videos", f"space_invaders_episode_{current_episode_count}.avi")
            frames = []
            
            # Reset environment
            obs, _ = env.reset()
            done, truncated = False, False
            step_count = 0
            
            # Record frames until episode ends (no step limit)
            while not (done or truncated):
                # Get action from the current model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Take step
                obs, reward, done, truncated, info = env.step(action)
                
                # Capture frame
                frame = env.render()
                frames.append(frame)
                
                step_count += 1
                
                # Safety check: prevent infinite loops (max 5000 steps)
                if step_count > 5000:
                    print(f"Warning: Episode exceeded 5000 steps, stopping recording")
                    break
            
            env.close()
            
            # Save video
            if frames:
                self._save_frames_as_video(frames, video_path)
                print(f"Video saved to: {video_path} ({step_count} steps, {len(frames)} frames)")
                self.video_count += 1
            else:
                print("No frames captured for video")
                
        except Exception as e:
            print(f"Error recording video: {e}")
    
    def _save_frames_as_video(self, frames, video_path):
        """Save frames as a video file."""
        try:
            import cv2
            
            if not frames:
                return
                
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
        except ImportError:
            print("OpenCV not available, saving frames as images instead")
            # Fallback: save as images
            for i, frame in enumerate(frames):
                plt.imsave(f"{video_path.replace('.avi', '')}_frame_{i}.png", frame)

class ProgressCallback(BaseCallback):
    """Custom callback to track training progress with proper episode counting."""
    
    def __init__(self, log_dir="./logs", verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.start_time = time.time()
        self.last_episode_count = 0
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        
        print("Progress tracking initialized with proper episode counting.")
        
    def _on_step(self):
        """Called after each training step."""
        return True
        
    def _on_rollout_end(self):
        """Called at the end of a rollout."""
        # Get episode count from the training environment's monitor
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            # Try to get episode count from the first environment's monitor
            first_env = self.training_env.envs[0]
            if hasattr(first_env, 'get_episode_rewards'):
                episode_rewards = first_env.get_episode_rewards()
                current_episode_count = len(episode_rewards)
                
                # Only print if we have new episodes
                if current_episode_count > self.last_episode_count:
                    new_episodes = current_episode_count - self.last_episode_count
                    self.last_episode_count = current_episode_count
                    
                    # Print progress every 10 episodes
                    if current_episode_count % 10 == 0:
                        elapsed_time = time.time() - self.start_time
                        print(f"Episode {current_episode_count}: Time elapsed: {elapsed_time/60:.1f} min")
        
        return True

def make_env():
    """Explicitly create and wrap the SpaceInvaders environment with standard Atari wrappers, custom reward wrapper, and frame stacking."""
    def _make_env():
        env = gym.make('ALE/SpaceInvaders-v5', repeat_action_probability=0.25)

        # Apply standard Atari wrappers explicitly
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        # Apply Atari preprocessing explicitly
        env = gym.wrappers.AtariPreprocessing(
            env,
            frame_skip=1,  # Already handled by MaxAndSkipEnv
            screen_size=84,
            grayscale_obs=True,
            scale_obs=True,
            terminal_on_life_loss=False  # Already handled by EpisodicLifeEnv
        )

        # Apply custom reward wrapper
        env = CustomRewardWrapper(env)

        # Apply frame stacking
        env = gym.wrappers.FrameStackObservation(env, stack_size=6)

        # Ensure observation compatibility with CnnPolicy
        if env.observation_space.shape[-1] == 1:
            class SqueezeWrapper(gym.ObservationWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    old_shape = env.observation_space.shape
                    new_shape = old_shape[:-1]
                    self.observation_space = gym.spaces.Box(
                        low=env.observation_space.low.reshape(new_shape),
                        high=env.observation_space.high.reshape(new_shape),
                        dtype=env.observation_space.dtype
                    )

                def observation(self, obs):
                    return obs.squeeze(-1)

            env = SqueezeWrapper(env)

        # Wrap environment with Monitor for logging
        env = Monitor(env)

        return env

    return _make_env

def train_ppo():
    """Train PPO agent on SpaceInvaders with GPU acceleration, improved exploration, and linear schedules."""
    print("Starting PPO training on SpaceInvaders-v5 with OPTIMIZED hyperparameters")
    print("=" * 60)
    print("NOTE: SpaceInvaders has delayed rewards - agent needs ~120 steps to see rewards")
    print("This is normal behavior for Atari games with complex reward structures")
    print("CUSTOM REWARD SYSTEM (from custom_reward_wrapper.py):")
    print("  - Score bonus: +2.0x for scoring points")
    print("  - Life loss penalty: -100 for losing a life")
    print("  - Inaction penalty: -0.1 after 5 consecutive NOOPs")
    print("  - Focused on event-based rewards to prevent reward hacking")
    print("STANDARD ATARI WRAPPERS (explicit setup):")
    print("  - NoopResetEnv: Random NOOP actions (0-30) at episode start")
    print("  - MaxAndSkipEnv: Skip 4 frames, take max of last 2 frames")
    print("  - EpisodicLifeEnv: End episode on life loss")
    print("  - AtariPreprocessing: Frame resizing, grayscale, normalization")
    print("  - CustomRewardWrapper: Enhanced reward shaping")
    print("  - FrameStackObservation: 6-frame temporal context")
    print("  - Monitor: Episode tracking and logging")
    print("OPTIMIZED TRAINING PARAMETERS:")
    print("  - Learning rate: 2.5e-4 to 1e-5 (linear schedule)")
    print("  - Entropy coefficient: 0.04 (increased for exploration)")
    print("  - Clip range: 0.2 (fixed)")
    print("  - N steps: 2048 (increased for stability)")
    print("  - Batch size: 256 (larger for better gradients)")
    print("  - N epochs: 10 (more epochs)")
    print("  - Target KL: Removed (using standard PPO-Clip)")
    print("  - Network: Deeper architecture [512, 256]")
    print("=" * 60)
    
    # Create environment using make_vec_env with custom environment function
    env = make_vec_env(
        make_env(),
        n_envs=8,  # Increased for better exploration
        seed=44
    )
    
    # Create model with optimized hyperparameters and linear schedules
    model = PPO(
        "CnnPolicy",  # Use CNN policy for image observations
        env,
        learning_rate=linear_schedule(2.5e-4, 1e-5),  # Slightly lower learning rate for more stable learning
        n_steps=2048,  # Increased for better stability
        batch_size=256,  # Larger batch size for better gradient estimates
        n_epochs=10,  # More epochs for better learning
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,  # Fixed clip range to avoid function issues
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.04,  # Increased entropy coefficient for better exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,  # Disable SDE for discrete actions
        tensorboard_log="./logs/tensorboard_logs",
        verbose=1,
        device=device,  # Use GPU device
        policy_kwargs={
            "normalize_images": False,  # Images are already normalized
            "features_extractor_kwargs": {"features_dim": 512},
            "net_arch": [dict(pi=[512, 256], vf=[512, 256])]  # Deeper networks
        }
    )
    
    # Create directories
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/checkpoints", exist_ok=True)
    os.makedirs("./logs/videos", exist_ok=True)
    
    # Create callbacks
    progress_callback = ProgressCallback(log_dir="./logs")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path="./logs/checkpoints",
        name_prefix="ppo_space_invaders"
    )
    
    # Create evaluation environment with the same wrapper
    eval_env = make_vec_env(
        make_env(),
        n_envs=1,
        seed=42
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/results",
        eval_freq=2500,
        deterministic=True,
        render=False
    )
    
    # Create video recording callback
    video_callback = VideoRecorderCallback(
        episode_freq=100,  # Record every 25 episodes
        video_length=0,  # 0 means record until episode ends
        log_dir="./logs"
    )
    
    # Create action diversity callback
    diversity_callback = ActionDiversityCallback(
        check_freq=5000,
        min_diversity=0.3,
        log_dir="./logs"
    )
    
    # Train the model
    print("Training started with ENHANCED parameters:")
    print(f"  Policy: CnnPolicy with deeper architecture")
    print(f"  Learning rate: 2.5e-4 to 1e-5 (linear schedule)")
    print(f"  Batch size: 256 (larger for better gradients)")
    print(f"  N steps: 2048 (increased for stability)")
    print(f"  N epochs: 10 (more epochs)")
    print(f"  Entropy coefficient: 0.04 (increased for exploration)")
    print(f"  Clip range: 0.2 (fixed)")
    print(f"  Target KL: Removed (using standard PPO-Clip)")
    print(f"  Device: {device}")
    print(f"  Video recording: Every 100 episodes (until episode ends)")
    print(f"  Action diversity monitoring: Every 5000 steps")
    print(f"  Expected: Delayed rewards after ~120 steps")
    print(f"  CUSTOM REWARDS: Event-based rewards for scoring and survival")
    print("=" * 60)
    
    start_time = time.time()
    
    # Combine all callbacks
    callbacks = [progress_callback, checkpoint_callback, eval_callback, video_callback, diversity_callback]
    
    model.learn(
        total_timesteps=120000000,  # Increased training time for better convergence
        callback=callbacks,
        progress_bar=True
    )
    
    # Save final model
    model.save("./logs/final_model")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/60:.1f} minutes!")
    print("Final model saved to ./logs/final_model")
    print("Best model saved to ./logs/best_model")
    print("Videos saved to ./logs/videos")

def test_environment():
    """Test the SpaceInvaders environment to verify it's working correctly."""
    print("Testing SpaceInvaders environment...")
    
    # Use the same environment setup as training
    env = make_env()()
    
    # Test environment
    obs, _ = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}")
    
    env.close()
    print("Environment test completed successfully!")

def list_available_games():
    """List available Atari games."""
    import gymnasium as gym
    import ale_py
    
    print("Available Atari games:")
    atari_games = [env for env in gym.envs.registry.keys() if 'ALE/' in env]
    for game in sorted(atari_games):
        print(f"  {game}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PPO Training for SpaceInvaders')
    parser.add_argument('--mode', choices=['train', 'test', 'list'], default='train',
                       help='Mode to run: train, test, or list games')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_ppo()
    elif args.mode == 'test':
        test_environment()
    elif args.mode == 'list':
        list_available_games() 