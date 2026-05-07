default:
    @just --list

# Train the base PPO expert policy
train-expert:
    uv run train_metacontroller.py --use_wandb=True

# Train the Discovery Module behavior cloning (Continuous)
train-discovery-continuous:
    uv run train_metacontroller.py --train_discovery=True --use_wandb=True --skip_ppo_eval=True --discrete_high_actions=False --batch_size=32

# Train the Discovery Module behavior cloning (Discrete)
train-discovery-discrete:
    uv run train_metacontroller.py --train_discovery=True --use_wandb=True --skip_ppo_eval=True --discrete_high_actions=True --batch_size=32

# Clear existing weights and train the Discovery Module from scratch (Continuous)
train-discovery-continuous-from-scratch:
    rm -f discovery_continuous.pt
    uv run train_metacontroller.py --train_discovery=True --use_wandb=True --skip_ppo_eval=True --discrete_high_actions=False --batch_size=32

# Clear existing weights and train the Discovery Module from scratch (Discrete)
train-discovery-discrete-from-scratch:
    rm -f discovery_discrete.pt
    uv run train_metacontroller.py --train_discovery=True --use_wandb=True --skip_ppo_eval=True --discrete_high_actions=True --batch_size=32


# Evaluate the trained agent
evaluate:
    uv run train_metacontroller.py --evaluate=True

# Optimize the Discovery Module inner hierarchical network with Evolutionary Strategies
train-evo-strat:
    uv run train_metacontroller.py --train_evo_strat=True --use_wandb=True --cpu=True
