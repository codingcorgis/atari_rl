#!/usr/bin/env python3
"""
PPO Evaluation Script for SpaceInvaders
This script evaluates a trained PPO agent and records a high-fidelity video.
It uses a dual-environment setup to ensure the policy's observations are correct
while rendering every frame to capture flickering objects and maintain correct speed.
"""

import os
import cv2
import ale_py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import (
    NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv
)
from custom_reward_wrapper import CustomRewardWrapper # Make sure this file is in the same directory

# --- Custom Wrapper to Fix Observation Shape ---
class SqueezeChannelWrapper(gym.ObservationWrapper):
    """Squeezes the last dimension of the observation if it's 1."""
    def __init__(self, env):
        super().__init__(env)
        if self.observation_space.shape[-1] == 1:
            squeezed_shape = self.observation_space.shape[:-1]
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=squeezed_shape, dtype=self.observation_space.dtype
            )

    def observation(self, obs):
        if obs.shape[-1] == 1:
            return obs.squeeze(axis=-1)
        return obs

def make_policy_env():
    """Creates the environment with the exact wrapper stack used for training."""
    env = gym.make('ALE/SpaceInvaders-v5', repeat_action_probability=0.25)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=2)
    # NOTE: EpisodicLifeEnv IS used here to match the training condition exactly.
    # The main loop will handle playing through all lives for the video.
    env = EpisodicLifeEnv(env) 
    env = FireResetEnv(env)
    env = gym.wrappers.AtariPreprocessing(
        env, frame_skip=1, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=False
    )
    env = CustomRewardWrapper(env)
    env = gym.wrappers.FrameStackObservation(env, 6)
    env = SqueezeChannelWrapper(env)
    return env

# --- 1. Load the Model ---
model_path = './logs/best_model/best_model.zip'
if not os.path.exists(model_path):
    print(f"Best model not found at {model_path}. Using final model instead.")
    model_path = './logs/final_model.zip'
    if not os.path.exists(model_path):
        print(f"Error: No model found at {model_path}. Please train a model first.")
        exit()
        
print(f"Loading model from: {model_path}")
model = PPO.load(model_path)

# --- 2. Create Two Environments ---
# The policy_env is for the agent's "brain" and must match the training setup exactly.
print("Creating policy environment...")
policy_env = make_policy_env()

# The render_env is for our "eyes" and has no frame skipping to capture every detail.
print("Creating high-fidelity render environment...")
render_env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array', repeat_action_probability=0.25, frameskip=1)
# We only need the Noop and Fire wrappers for correct reset behavior
render_env = NoopResetEnv(render_env, noop_max=30)
render_env = FireResetEnv(render_env)

# --- 3. Record Gameplay Frames ---
video_path = "best_spaceinvaders_policy.mp4"
frames = []
try:
    # Reset both environments to ensure they start in sync
    obs, _ = policy_env.reset()
    render_env.reset()

    print("Recording video... Press Ctrl+C to stop.")
    # Use a simple while True loop, termination is handled inside
    while True:
        # Get an action from the agent based on the preprocessed observation
        action, _ = model.predict(obs, deterministic=True)

        # Step the policy environment to get the next observation for the agent
        obs, reward, done, truncated, info = policy_env.step(action)
        
        # If the policy_env says an episode is done (due to life loss), reset it
        if done or truncated:
            obs, _ = policy_env.reset()

        # Step the render environment for `skip` frames, applying the same action.
        # This simulates the frame skipping while letting us render every single frame.
        # This is the key to fixing the speed and capturing flickering objects.
        skip_rate = 2 # Must match the MaxAndSkipEnv skip rate from training
        done_render = False
        for _ in range(skip_rate):
            # Use the flicker fix by combining frames before and after the step
            prev_frame = render_env.render()
            _, _, done_render, truncated_render, _ = render_env.step(action)
            new_frame = render_env.render()
            combined_frame = np.maximum(prev_frame, new_frame)
            frames.append(combined_frame)

            # If the render_env (the "real" game) is over, break the inner loop
            if done_render or truncated_render:
                break
        
        # If the render_env is over, break the main loop
        if done_render or truncated_render:
            print("Render environment finished.")
            break
        
except KeyboardInterrupt:
    print("\nRecording stopped by user.")
finally:
    policy_env.close()
    render_env.close()

# --- 4. Save Video Using OpenCV ---
if frames:
    print(f"Saving {len(frames)} frames to video...")
    height, width, _ = frames[0].shape
    # Use 'mp4v' codec and save at 60 FPS to match the Atari's native speed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 60.0, (width, height))

    for frame in frames:
        # OpenCV expects frames in BGR format, so we convert from RGB
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
    print(f"Video successfully saved to {video_path}")
else:
    print("No frames were recorded.")
