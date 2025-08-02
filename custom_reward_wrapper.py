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
        """Take a step and apply the refined, event-based reward function."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_score = info.get('score', 0)
        current_lives = info.get('lives', 0)

        # --- Primary Positive Reward: Scoring Points ---
        # A strong, direct reward for the main objective.
        # We multiply the score delta to make it more significant than small penalties.
        score_bonus = (current_score - self.last_score) * 2.0

        # --- Primary Negative Reward: Losing a Life ---
        # A large, unambiguous penalty that is the main driver for learning survival.
        life_penalty = -100.0 if current_lives < self.last_lives else 0

        # --- Nudge Penalty: Discourage Inaction ---
        # A small penalty to prevent the agent from getting stuck doing nothing.
        inaction_penalty = 0
        if action == 0: # NOOP action
            self.consecutive_noops += 1
            # Apply penalty only after several consecutive NOOPs
            if self.consecutive_noops > 5:
                inaction_penalty = -0.1
        else:
            self.consecutive_noops = 0

        # --- Combine the clear, event-based rewards ---
        # Note: We remove the constant 'survival' and 'movement' bonuses.
        # The agent must now learn to move and survive as a *strategy* to
        # get the score_bonus and avoid the life_penalty.
        custom_reward = (
            score_bonus +
            life_penalty +
            inaction_penalty
        )

        # Update state for the next step
        self.last_score = current_score
        self.last_lives = current_lives

        return obs, custom_reward, terminated, truncated, info


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
