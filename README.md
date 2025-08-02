# Atari Reinforcement Learning Training Setup

This repository contains scripts for training PPO agents on Atari games, specifically SpaceInvaders, with custom reward wrappers and comprehensive training utilities.

## Setup

1. **Virtual Environment**: A Python virtual environment called `rl_env` has been created
2. **Dependencies**: All necessary packages have been installed including:
   - stable-baselines3
   - gymnasium[atari]
   - opencv-python
   - matplotlib
   - rl-zoo3

## Available Training Scripts

### 1. SpaceInvaders Training (`train_space_invaders.py`)
A comprehensive PPO training script for SpaceInvaders with progress tracking, video recording, and custom reward wrapper.

**Usage:**
```bash
# Activate virtual environment
rl_env\Scripts\activate

# List available Atari games
python train_space_invaders.py --mode list

# Test the environment
python train_space_invaders.py --mode test

# Train the agent
python train_space_invaders.py --mode train

# Evaluate a trained model
python train_space_invaders.py --mode evaluate --model-path ./logs/final_model
```

## Core Files

### Training Scripts
- **`train_space_invaders.py`** - Main training script for SpaceInvaders with PPO
- **`custom_reward_wrapper.py`** - Custom reward wrapper for enhanced training

### Analysis and Visualization
- **`plot_training_progress.py`** - Script to plot and analyze training progress
- **`render_best_policy.py`** - Script to render the best trained policy

## Features

### Progress Tracking
- Real-time training progress display
- Episode counting and timing
- Average reward tracking

### Model Saving
- Automatic checkpoint saving every 10,000 steps
- Best model saving based on evaluation performance
- Final model saving upon completion

### Video Recording
- Periodic video recording of agent behavior
- Videos saved every 25 episodes (configurable)
- OpenCV-based video encoding

### Evaluation
- Separate evaluation environment
- Deterministic policy evaluation
- Performance metrics tracking

## Training Parameters

### PPO Configuration
- **Learning Rate**: 3e-4
- **Batch Size**: 256
- **N Steps**: 2048
- **N Epochs**: 15
- **Gamma**: 0.99
- **Entropy Coefficient**: 0.02
- **Clip Range**: 0.2

### Environment Configuration
- **SpaceInvaders**: 4 parallel environments with frame stacking (4 frames)
- **Custom Reward Wrapper**: Enhanced reward system for better learning

## Custom Reward System

The training uses a custom reward wrapper (`custom_reward_wrapper.py`) that provides:
- **Score bonus**: +2.0x for scoring points
- **Life loss penalty**: -100 for losing a life
- **Inaction penalty**: -0.1 after 5 consecutive NOOPs
- **Event-based rewards**: Focused on preventing reward hacking

## Output Structure

```
logs/
├── checkpoints/          # Model checkpoints
├── best_model/          # Best performing model
├── videos/              # Training videos
├── tensorboard_logs/    # TensorBoard logs
├── results/             # Evaluation results
└── final_model/         # Final trained model
```

## Troubleshooting

### Atari ROM Issues
The Atari environments are now working! Make sure to:

1. **Import ale_py**: All scripts include `import ale_py` to register ALE environments
2. **Check available environments**:
   ```bash
   python train_space_invaders.py --mode list
   ```

3. **If Atari ROMs are not working**:
   ```bash
   pip install autorom[accept-rom-license]
   pip install gymnasium[atari,accept-rom-license]
   ```

### Environment Issues
- Ensure the virtual environment is activated: `rl_env\Scripts\activate`
- Check that all dependencies are installed: `pip list`
- Test the environment first: `python train_space_invaders.py --mode test`

## Popular Atari Games for Training

1. **SpaceInvaders-v5** - Classic space shooter
2. **Breakout-v5** - Paddle and ball game
3. **Pong-v5** - Classic tennis game
4. **Qbert-v5** - Platform jumping game
5. **BeamRider-v5** - Space combat game
6. **Enduro-v5** - Racing game
7. **Seaquest-v5** - Underwater combat
8. **Asteroids-v5** - Space shooter
9. **Freeway-v5** - Crossing game
10. **Riverraid-v5** - Combat flight game

## Training Tips

1. **Monitor Progress**: Use TensorBoard to visualize training: `tensorboard --logdir ./logs/tensorboard_logs`
2. **Save Checkpoints**: Models are automatically saved, so you can resume training
3. **Video Review**: Check the videos folder to see how your agent improves over time
4. **Evaluation**: Regularly evaluate your model to track performance
5. **Custom Rewards**: The custom reward wrapper helps with better learning behavior

## Next Steps

1. **Test the Environment**: Run `python train_space_invaders.py --mode test`
2. **Start Training**: Run `python train_space_invaders.py --mode train`
3. **Monitor Progress**: Use TensorBoard to track training progress
4. **Analyze Results**: Use the analysis scripts to understand agent behavior
5. **Experiment**: Try different hyperparameters and environments 