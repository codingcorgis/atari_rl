# Reinforcement Learning Training Setup

This repository contains scripts for training PPO agents on various environments, including Atari games and classic RL environments.

## Setup

1. **Virtual Environment**: A Python virtual environment called `rl_env` has been created
2. **Dependencies**: All necessary packages have been installed including:
   - stable-baselines3
   - gymnasium[atari]
   - opencv-python
   - matplotlib
   - rl-zoo3

## Available Training Scripts

### 1. CartPole Training (`train_cartpole_simple.py`)
A working PPO training script for the CartPole environment.

**Usage:**
```bash
# Activate virtual environment
rl_env\Scripts\activate

# Test the environment
python train_cartpole_simple.py --mode test

# Train the agent
python train_cartpole_simple.py --mode train

# Evaluate a trained model
python train_cartpole_simple.py --mode evaluate --model-path ./logs/final_model
```

### 2. CartPole Training - Complete Version (`train_cartpole_final.py`)
A comprehensive PPO training script for CartPole with progress tracking and evaluation.

**Usage:**
```bash
# Activate virtual environment
rl_env\Scripts\activate

# Test the environment
python train_cartpole_final.py --mode test

# Train the agent
python train_cartpole_final.py --mode train

# Evaluate a trained model
python train_cartpole_final.py --mode evaluate --model-path ./logs/final_model

# List available environments
python train_cartpole_final.py --mode list
```

### 3. SpaceInvaders Training (`train_space_invaders.py`)
A comprehensive PPO training script for SpaceInvaders with progress tracking and video recording.

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

### 4. Demo Training (`demo_training.py`)
A minimal demo script for quick CartPole training verification.

**Usage:**
```bash
python demo_training.py
```

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
- **Batch Size**: 64
- **N Steps**: 128
- **N Epochs**: 4
- **Gamma**: 0.99
- **Entropy Coefficient**: 0.01
- **Clip Range**: 0.2

### Environment Configuration
- **CartPole**: 4 parallel environments
- **SpaceInvaders**: 4 parallel environments with frame stacking (4 frames)

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
- Test with CartPole first before trying Atari environments

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

1. **Start Simple**: Begin with CartPole to verify your setup works
2. **Monitor Progress**: Use TensorBoard to visualize training: `tensorboard --logdir ./logs/tensorboard_logs`
3. **Save Checkpoints**: Models are automatically saved, so you can resume training
4. **Video Review**: Check the videos folder to see how your agent improves over time
5. **Evaluation**: Regularly evaluate your model to track performance

## Next Steps

1. **Test CartPole Training**: Run the CartPole script to verify everything works
2. **Install Atari ROMs**: Follow the troubleshooting steps above
3. **Train SpaceInvaders**: Once ROMs are installed, train on SpaceInvaders
4. **Experiment**: Try different hyperparameters and environments
5. **Visualize**: Use TensorBoard to analyze training progress 