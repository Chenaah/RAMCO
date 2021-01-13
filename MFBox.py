import os
import gym
from gym.wrappers import Monitor
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer
from tf2rl.algos.ppo import PPO
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim
from tf2rl.algos.td3 import TD3

import argparse
import assistive_gym
import tensorflow as tf

import wandb




def run(parser):

    args = parser.parse_args()

    if args.gpu < 0:
        tf.config.experimental.set_visible_devices([], 'GPU')
    else:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[args.gpu], 'GPU')
        tf.config.experimental.set_virtual_device_configuration(physical_devices[args.gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])

    if args.env == 200:
        envname = 'ScratchItchPR2X'
    elif args.env == 201:
        envname = 'DressingPR2X'
    elif args.env == 202:
        envname = 'BedBathingPR2X'
    
    logdir = f'MFBox_Assistive'
    if args.SAC:
        wandb.init(config=vars(args), project="Assistive Gym", name=f'SAC on {envname}')
    elif args.PPO:
        wandb.init(config=vars(args), project="Assistive Gym", name=f'PPO on {envname}')
    elif args.TD3:
        wandb.init(config=vars(args), project="Assistive Gym", name=f'TD3 on {envname}')
    elif args.DEBUG:
        logdir = f'DEBUG_Assistive'
        wandb.init(config=vars(args), project="Assistive Gym", name=f'DEBUG on {envname}')
    else:
        print('PLEASE INDICATE THE ALGORITHM !!')

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    parser.set_defaults(logdir=logdir)
    args = parser.parse_args()

    
    env = gym.make(f'{envname}-v0')
    #test_env = Monitor(env,logdir,force=True)
    test_env= gym.make(f'{envname}-v0')

    if args.SAC:

        policy = SAC(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            gpu=args.gpu,
            memory_capacity=args.memory_capacity,
            max_action=env.action_space.high[0],
            batch_size=args.batch_size,
            n_warmup=args.n_warmup,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha)
        trainer = Trainer(policy, env, args, test_env=test_env)
        

    elif args.PPO:
        policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=(64, 64),
        critic_units=(64, 64),
        n_epoch=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        hidden_activation_actor="tanh",
        hidden_activation_critic="tanh",
        discount=0.99,
        lam=0.95,
        entropy_coef=0.,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
        trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)

    elif args.TD3:
        policy = TD3(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
        trainer = Trainer(policy, env, args, test_env=test_env)

    elif args.DEBUG:

        policy = SAC(
            state_shape=env.observation_space.shape,
            action_dim=env.action_space.high.size,
            gpu=args.gpu,
            memory_capacity=args.memory_capacity,
            max_action=env.action_space.high[0],
            batch_size=args.batch_size,
            n_warmup=100,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha)
        parser.set_defaults(test_interval=200)
        args = parser.parse_args()

        trainer = Trainer(policy, env, args, test_env=None)
    
    trainer()

def main():
    parser = Trainer.get_argument()
    parser = OnPolicyTrainer.get_argument(parser)
    parser = SAC.get_argument(parser)
    parser = PPO.get_argument(parser)
    parser = TD3.get_argument(parser)

    parser.add_argument('--SAC', action='store_true')
    parser.add_argument('--PPO', action='store_true')
    parser.add_argument('--TD3', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--env', type=int, default=0)

    

    parser.set_defaults(batch_size=32) #100
    parser.set_defaults(n_warmup=10000) #10000
    parser.set_defaults(max_steps=2e6)
    parser.set_defaults(gpu=0)
    parser.set_defaults(test_interval=200*100)
    parser.set_defaults(test_episodes=3) 

    args = parser.parse_args()
    print(vars(args))
    run(parser)


main()