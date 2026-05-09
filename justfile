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

# Optimize the Discovery Module inner hierarchical network with Evolutionary Strategies on one machine
train-evo-strat pop="200":
    uv run train_metacontroller.py --train_evo_strat=True --use_wandb=True --cpu=True --evo_strat_population_size={{pop}}

# Optimize the Discovery Module inner hierarchical network with Evolutionary Strategies using Torchrun
train-evo-strat-distributed-cpu numprocs="10" pop="200":
    uv run torchrun --nproc_per_node={{numprocs}} train_metacontroller.py --train_evo_strat=True --use_wandb=True --cpu=True --evo_strat_population_size={{pop}}

# Optimize the Joint Metacontroller (higher/lower) with Evolutionary Strategies on one machine
train-evo-strat-joint-cpu pop="200" target="higher" discrete="True" lr="5e-4" noise="0.005":
    uv run train_metacontroller.py --train_evo_strat_joint=True --use_wandb=True --cpu=True --evo_strat_population_size={{pop}} --evo_strat_joint_target={{target}} --discrete_high_actions={{discrete}} --evo_strat_learning_rate={{lr}} --evo_strat_noise_scale={{noise}}

# Optimize the Joint Metacontroller (higher/lower) with Evolutionary Strategies using Torchrun
train-evo-strat-joint-distributed-cpu numprocs="10" pop="200" target="higher" discrete="True" lr="5e-4" noise="0.005":
    uv run torchrun --nproc_per_node={{numprocs}} train_metacontroller.py --train_evo_strat_joint=True --use_wandb=True --cpu=True --evo_strat_population_size={{pop}} --evo_strat_joint_target={{target}} --discrete_high_actions={{discrete}} --evo_strat_learning_rate={{lr}} --evo_strat_noise_scale={{noise}}

# Optimize the Joint Metacontroller with PPO (Joint PPO) on one machine
train-ppo-joint-cpu discrete="True":
    uv run train_metacontroller.py --train_ppo_joint=True --use_wandb=True --cpu=True --discrete_high_actions={{discrete}}
