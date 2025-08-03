#!/usr/bin/env python3
"""
Refined and Focused Custom Reward Wrapper for SpaceInvaders.

This version simplifies the reward structure to provide the clearest possible
signals for the most important behaviors: scoring points and staying alive.
"""

import gymnasium as gym

class CustomRewardWrapper(gym.Wrapper):
    """
    A refined reward wrapper that focuses on critical, event-based rewards
    to prevent reward hacking and encourage robust learning.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0
        self.last_lives = 0
        self.consecutive_noops = 0

    def reset(self, **kwargs):
        """Reset the environment and all tracking variables."""
        obs, info = self.env.reset(**kwargs)
        self.last_score = 0
        self.last_lives = info.get('lives', 0)
        self.consecutive_noops = 0
        return obs, info

    def step(self, action):
        """Take a step and apply the rebalanced, event-based reward function."""
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        current_score = original_reward
        current_lives = info.get('lives', 0)

        # --- 1. Primary Positive Reward: MASSIVE Score Bonus ---
        # The reward for hitting an alien needs to be significant enough to
        # justify the risk of moving and shooting.
        score_bonus = 0
        score_delta = current_score - self.last_score
        if score_delta > 0:
            # We make the bonus a substantial fraction of the life penalty.
            # A standard alien is now worth 50-300 points in reward.
            score_bonus = score_delta * 5.0

        # --- 2. Primary Negative Reward: Life Loss Penalty ---
        # This remains the main driver for learning survival.
        if current_lives < self.last_lives:
            life_penalty = -100.0
        else:
            life_penalty = 0

        # Update state for the next step
        self.last_score = current_score
        self.last_lives = current_lives

        # The total reward is now simple, powerful, and event-driven.
        # The agent must learn that the only way to get a positive total reward
        # is to score points while avoiding the life penalty.
        total_reward = score_bonus + life_penalty

        return obs, total_reward, terminated, truncated, info


def test_custom_reward():
    """A simple function to test the wrapper and see the rewards in action."""
    import ale_py

    env = gym.make('ALE/SpaceInvaders-v5')
    env = CustomRewardWrapper(env)

    obs, _ = env.reset()
    total_reward = 0

    print("--- Testing RefinedRewardWrapper ---")
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Print any significant reward event
        if reward != 0:
            print(f"Step {i:03d}: Action={action}, Reward={reward:6.2f}, Total={total_reward:6.2f}, Lives={info.get('lives', 0)}")

        if done or truncated:
            print("--- Episode Finished ---")
            break

    env.close()
    print(f"\nFinal total reward after 200 random steps: {total_reward:.2f}")

if __name__ == "__main__":
    test_custom_reward()
