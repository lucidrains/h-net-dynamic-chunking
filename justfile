default:
    @just --list

# Train the base PPO expert policy
train-expert:
    uv run train_metacontroller.py --use_wandb=True

# Train the Discovery Module behavior cloning
train-discovery:
    uv run train_metacontroller.py --train_discovery=True --use_wandb=True --skip_ppo_eval=True

# Clear existing weights and train the Discovery Module from scratch
train-discovery-from-scratch:
    rm -f discovery.pt
    uv run train_metacontroller.py --train_discovery=True --use_wandb=True --skip_ppo_eval=True

# Evaluate the trained agent
evaluate:
    uv run train_metacontroller.py --evaluate=True
