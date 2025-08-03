#!/usr/bin/env python3
"""
Fine-tuning Script for SpaceInvaders PPO Agent
This script loads the best model from a previous training run and continues
training with a lower, fixed learning rate for improved and more stable performance.
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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

# It's best practice to have all imports at the top
import ale_py
from custom_reward_wrapper import CustomRewardWrapper # Assuming your wrapper is in this file

# --- SCRIPT CONFIGURATION ---
ENV_ID = "ALE/SpaceInvaders-v5"
LOG_DIR = "./logs_fine_tune"  # Use a separate log directory for fine-tuning
MODEL_NAME = "ppo_space_invaders_fine_tuned"
TOTAL_TIMESTEPS = 5_000_000  # A reasonable budget for fine-tuning
FRAME_SKIP_RATE = 2
# --- THE FIX IS HERE ---
# This value MUST match the FrameStack size used to train the original model.
# The error message indicates the original model used 6.
FRAME_STACK_SIZE = 6 

# --- SETUP ---
set_random_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# --- HELPER FUNCTIONS AND CLASSES ---

class VideoRecorderCallback(BaseCallback):
    """
    Callback to record high-fidelity videos using a robust dual-environment setup.
    """
    def __init__(self, render_freq: int, video_folder: str = 'videos/'):
        super().__init__()
        self.render_freq = render_freq
        self.video_folder = video_folder
        os.makedirs(self.video_folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.render_freq == 0:
            self._record_video()
        return True

    def _record_video(self):
        print(f"\nRecording high-fidelity video at {self.num_timesteps} steps...")

        # Create a policy_env that is IDENTICAL to the training env
        # Pass is_for_training=True because the model expects EpisodicLifeEnv
        policy_env = make_env(ENV_ID, seed=999, is_for_training=True)()

        # Create a render_env for high-fidelity visuals
        render_env = gym.make(ENV_ID, render_mode='rgb_array', repeat_action_probability=0.25, frameskip=1)
        
        frames = []
        try:
            obs, _ = policy_env.reset()
            render_env.reset()
            done_render = False
            while not done_render:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done_policy, _, _ = policy_env.step(action)
                
                if done_policy:
                    obs, _ = policy_env.reset()

                for _ in range(FRAME_SKIP_RATE):
                    prev_frame = render_env.render()
                    _, _, done_render, truncated_render, _ = render_env.step(action)
                    new_frame = render_env.render()
                    combined_frame = np.maximum(prev_frame, new_frame)
                    frames.append(combined_frame)
                    if done_render or truncated_render:
                        break
        finally:
            policy_env.close()
            render_env.close()

        video_path = os.path.join(self.video_folder, f"{MODEL_NAME}_step_{self.num_timesteps}.mp4")
        self._save_frames_as_video(frames, video_path)

    def _save_frames_as_video(self, frames, video_path):
        if not frames: return
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved to: {video_path}")

def make_env(env_id: str, seed: int, is_for_training: bool = True):
    """Creates and wraps the Atari environment using modern Gymnasium wrappers."""
    def _init():
        env_kwargs = {'repeat_action_probability': 0.25}
        env = gym.make(env_id, frameskip=1, **env_kwargs)
        
        env = gym.wrappers.AtariPreprocessing(
            env, 
            noop_max=30, 
            frame_skip=FRAME_SKIP_RATE, 
            terminal_on_life_loss=is_for_training,
            grayscale_newaxis=True
        )
        
        env = CustomRewardWrapper(env)
        env = Monitor(env)
        return env
    return _init

def fine_tune_ppo():
    """Main function to configure and run the PPO fine-tuning."""
    print("--- Starting PPO Fine-tuning ---")
    
    # Check if best model exists
    best_model_path = './logs/best_model/best_model.zip'
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found at {best_model_path}")
        print("Please run the main training script first to generate a model to fine-tune.")
        return
    
    print(f"Loading best model from: {best_model_path}")

    # --- Create Environments ---
    train_env = make_vec_env(
        make_env(ENV_ID, seed=42, is_for_training=True),
        n_envs=8,
        seed=42,
        vec_env_cls=SubprocVecEnv
    )
    train_env = VecFrameStack(train_env, n_stack=FRAME_STACK_SIZE)

    eval_env = make_vec_env(
        make_env(ENV_ID, seed=1337, is_for_training=False),
        n_envs=1,
        seed=1337
    )
    eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK_SIZE)

    # --- Load the best model and set new, lower learning rate ---
    print("Loading pre-trained model for fine-tuning...")
    # Set a new, small, fixed learning rate for fine-tuning
    fine_tune_lr = 1e-5 
    
    model = PPO.load(
        best_model_path, 
        env=train_env, 
        device=device,
        custom_objects={"learning_rate": fine_tune_lr} # Set the new learning rate on load
    )
    
    print(f"Fine-tuning with a fixed learning rate of: {model.learning_rate}")

    # --- CALLBACKS ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=f"{LOG_DIR}/checkpoints/",
        name_prefix=MODEL_NAME
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{LOG_DIR}/best_model/",
        log_path=f"{LOG_DIR}/results/",
        eval_freq=10_000,
        deterministic=True,
        render=False
    )
    video_callback = VideoRecorderCallback(
        render_freq=20_000,
        video_folder=f"{LOG_DIR}/videos/"
    )
    
    callbacks = [checkpoint_callback, eval_callback, video_callback]

    # --- FINE-TUNING ---
    print("--- Starting model fine-tuning ---")
    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False  # Important: don't reset timesteps for fine-tuning
    )
    end_time = time.time()

    # --- FINAL SAVE ---
    model.save(f"{LOG_DIR}/{MODEL_NAME}_final.zip")
    print(f"Fine-tuning finished in {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Fine-tuned model saved to {LOG_DIR}/{MODEL_NAME}_final.zip")
    print(f"Best fine-tuned model saved to {LOG_DIR}/best_model/")

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    fine_tune_ppo()
