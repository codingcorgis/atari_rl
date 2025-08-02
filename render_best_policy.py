import gymnasium as gym
import ale_py
import cv2
import numpy as np
from stable_baselines3 import PPO
import os

# Load the best model
model_path = './logs/best_model/best_model.zip'
if not os.path.exists(model_path):
    print(f"Best model not found at {model_path}. Using final model instead.")
    model_path = './logs/final_model.zip'
model = PPO.load(model_path)

# Create the environment
#env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
env = gym.make('SpaceInvadersNoFrameskip-v4', render_mode='rgb_array')

env = gym.wrappers.AtariPreprocessing(env, frame_skip=1, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

# Prepare video writer
video_path = "best_spaceinvaders_policy.avi"
frames = []
obs, _ = env.reset()
# Get the very first frame to start
prev_frame = env.render()
done, truncated = False, False

while not (done or truncated):
    # Predict and take a step in the environment
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    # Render the new frame AFTER the step
    new_frame = env.render()
    # Combine the frame from BEFORE the step and the one AFTER the step
    # This captures all flickering objects
    combined_frame = np.maximum(prev_frame, new_frame)
    frames.append(combined_frame)
    
    # Update the previous frame for the next loop iteration
    prev_frame = new_frame

env.close()
# Save video using OpenCV
height, width, _ = frames[0].shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_path, fourcc, 30, (width, height))
for frame in frames:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()

print(f"Video saved to {video_path}")