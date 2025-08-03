#!/usr/bin/env python3
"""
PPO Training Script for SpaceInvaders
This script trains a PPO agent on the SpaceInvaders environment
with a refined setup for robust and effective learning.
"""

import os
import time
from typing import Callable

import gymnasium as gym
import torch
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (EvalCallback,
                                                CheckpointCallback, BaseCallback)
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecFrameStack

# It's best practice to have all imports at the top
import ale_py
from custom_reward_wrapper import CustomRewardWrapper # Assuming your wrapper is in this file

# --- SCRIPT CONFIGURATION ---
ENV_ID = "ALE/SpaceInvaders-v5"
LOG_DIR = "./logs"
MODEL_NAME = "ppo_space_invaders"
TOTAL_TIMESTEPS = 10_000_000 # A realistic training budget for Atari

# --- SETUP ---
set_random_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- HELPER FUNCTIONS AND CLASSES ---

def linear_schedule(initial_value: float, final_value: float = 1e-5) -> Callable[[float], float]:
    """Linear learning rate schedule that decays to a final value."""
    def func(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)
    return func

# Your CustomRewardWrapper should be in its own file (custom_reward_wrapper.py)

class ActionDiversityCallback(BaseCallback):
    """Callback to monitor action diversity during training."""
    def __init__(self, check_freq: int = 2048 * 5, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.action_names = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"}

    def _on_rollout_end(self) -> None:
        """Checks action diversity at the end of each rollout."""
        # This check is now triggered based on the number of rollouts to be less frequent
        if self.n_calls > 0 and self.n_calls % self.check_freq == 0:
            actions = self.model.rollout_buffer.actions.flatten()
            total_actions = len(actions)
            if total_actions == 0: return

            action_counts = {k: np.count_nonzero(actions == k) for k in self.action_names.keys()}
            
            print(f"\n--- Action Diversity at {self.num_timesteps} steps ---")
            for action_id, name in self.action_names.items():
                percentage = (action_counts.get(action_id, 0) / total_actions) * 100
                print(f"  {name:<10}: {percentage:.2f}%")
            print("-----------------------------------------")

    def _on_step(self) -> bool:
        return True

class VideoRecorderCallback(BaseCallback):
    """Custom callback to record high-fidelity videos during training."""
    def __init__(self, video_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, video_folder: str = 'videos/'):
        super().__init__()
        self.video_env = video_env
        self.render_freq = render_freq
        self.n_eval_episodes = n_eval_episodes
        self.video_folder = video_folder
        os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.render_freq == 0:
            print(f"\nRecording video at {self.num_timesteps} steps...")
            frames = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.video_env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, done, _, _ = self.video_env.step(action)
                    frame = self.video_env.render()
                    frames.append(frame)
            
            video_path = os.path.join(self.video_folder, f"{MODEL_NAME}_step_{self.num_timesteps}.mp4")
            self._save_frames_as_video(frames, video_path)
        return True

    def _save_frames_as_video(self, frames, video_path):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved to: {video_path}")

def make_env(env_id: str, seed: int, wrapper_kwargs: dict = None):
    """Creates and wraps the Atari environment using the SB3 helper."""
    def _init():
        # The make_atari_env helper handles all standard wrappers correctly.
        env = make_atari_env(
            env_id,
            wrapper_class=CustomRewardWrapper,
            env_kwargs={'repeat_action_probability': 0.25, **(wrapper_kwargs or {})}
        )
        # We do not apply FrameStack here, it will be applied to the VecEnv
        env.seed(seed)
        return env
    return _init

def train_ppo():
    """Main function to configure and run the PPO training."""
    print("--- Starting PPO Training ---")

    # --- Create Environments ---
    # Create the vectorized training environment
    train_env = make_vec_env(
        make_env(ENV_ID, seed=42),
        n_envs=8,
        seed=42
    )
    # Apply FrameStack to the vectorized environment
    train_env = VecFrameStack(train_env, n_stack=6)

    # Create the evaluation environment (single instance)
    eval_env = make_vec_env(
        make_env(ENV_ID, seed=1337),
        n_envs=1,
        seed=1337
    )
    eval_env = VecFrameStack(eval_env, n_stack=6)

    # Create a separate, high-fidelity environment for video recording
    video_env = make_atari_env(
        ENV_ID,
        wrapper_class=CustomRewardWrapper,
        env_kwargs={'repeat_action_probability': 0.25, 'frameskip': 1}
    )
    video_env = VecFrameStack(video_env, n_stack=6)


    # --- MODEL HYPERPARAMETERS ---
    model = PPO(
        "CnnPolicy",
        train_env,
        learning_rate=linear_schedule(2.5e-4, 1e-5),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None, # Use PPO-Clip, not PPO-Penalty
        tensorboard_log=f"{LOG_DIR}/tensorboard_logs/",
        verbose=1,
        device=device,
        policy_kwargs={
            "normalize_images": False,
            "net_arch": [dict(pi=[512, 256], vf=[512, 256])]
        }
    )

    # --- CALLBACKS ---
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f"{LOG_DIR}/checkpoints/",
        name_prefix=MODEL_NAME
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{LOG_DIR}/best_model/",
        log_path=f"{LOG_DIR}/results/",
        eval_freq=25_000,
        deterministic=True,
        render=False
    )
    video_callback = VideoRecorderCallback(
        video_env,
        render_freq=100_000,
        video_folder=f"{LOG_DIR}/videos/"
    )
    
    callbacks = [checkpoint_callback, eval_callback, ActionDiversityCallback(), video_callback]

    # --- TRAINING ---
    print("--- Starting model training ---")
    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True
    )
    end_time = time.time()

    # --- FINAL SAVE ---
    model.save(f"{LOG_DIR}/{MODEL_NAME}_final.zip")
    print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Final model saved to {LOG_DIR}/{MODEL_NAME}_final.zip")

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    train_ppo()
