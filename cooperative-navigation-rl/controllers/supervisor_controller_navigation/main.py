import math
import random
import sys
import os
sys.path.append("/usr/local/webots/lib/controller/python")
sys.path.append("/usr/local/webots/lib/controller/python/controller")
sys.path.append('../')
os.environ['WEBOTS_HOME'] = '/usr/local/webots'
import numpy as np
import pandas as pd
from pathlib import Path
# import wandb
import socket
import setproctitle
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
exp_path = os.path.join(current_dir, "..", "..")
sys.path.append(exp_path)
save_path = os.path.join(current_dir, "..", "..", "results")
from config import parser
from my_runner import MyRunner as Runner


def main(args_):
    args, unknown = parser.parse_known_args(args_)

    # cuda and # threads
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(args.n_training_threads)
        if args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(args.n_training_threads)

    # setup file to output tensorboard, hyperparameters, and saved models
    run_dir = os.path.join(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0], "..",
        "results", args.env_name, args.algorithm_name
    )
    run_dir = Path(run_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))


    exist_run_nums = [int(str(folder.name).split('run')[1]) 
                      for folder in run_dir.iterdir() 
                      if str(folder.name).startswith('run')]
    if len(exist_run_nums) == 0:
        curr_run = 'run1'
    else:
        curr_run = 'run%i' % (max(exist_run_nums) + 1)
    
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(args.algorithm_name) + "-" + str(args.env_name) + "@yfw")

    # set seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    num_agents = args.num_agents

    assert args.env_name == "navigation"
    from supervisor_controller_navigation import Epuck2Supervisor
    env = Epuck2Supervisor(args)
    # elif args.env_name == "Predator_prey":
    #     from supervisor_controller_pp import Epuck2Supervisor
    #     env = Epuck2Supervisor(args)
    # elif args.env_name == "homing":
    #     from supervisor_controller_homing import Epuck2Supervisor
    #     env = Epuck2Supervisor(args)
    # elif args.env_name == "formation":
    #     from supervisor_controller_formation import Epuck2Supervisor
    #     env = Epuck2Supervisor(args)

    config = {"args": args,
              "env": env,
              "num_agents": num_agents,
              "device": device,
              "run_dir": run_dir}

    runner = Runner(config=config)
    
    progress_filename = os.path.join(run_dir,'config.csv')
    df = pd.DataFrame(list(args.__dict__.items()),columns=['Name', 'Value'])
    df.to_csv(progress_filename,index=False)
    
    columns = ['env_step', 'episode', 'train_step']
    for i in range(num_agents):
        columns.append(f'{i}_episode_r')
    columns.append('mean_episode_r')
    columns.append('action_left_mean')
    columns.append('action_right_mean')
    columns.append('action_abs_mean')
    columns.append('action_std')
    columns.append('mean_episode_cost')
    columns.append('safety_cost_rate')
    columns.append('projection_intervention_rate')
    columns.append('num_target')
    columns.append('num_collision')
    columns.append('finish_time')
    
    progress_filename = os.path.join(run_dir,'progress.csv')
    df = pd.DataFrame(columns=columns)
    df.to_csv(progress_filename,index=False)
    
    progress_filename = os.path.join(run_dir,'progress_eval.csv')
    df = pd.DataFrame(columns=columns)
    df.to_csv(progress_filename,index=False)
    
    # progress_filename_train = os.path.join(run_dir,'progress_train.csv')
    # df = pd.DataFrame(columns=['step','loss','Q_tot','grad_norm'])
    # df.to_csv(progress_filename_train,index=False)
    
    # progress_filename_train = os.path.join(run_dir,'progress_train.csv')
    # df = pd.DataFrame(columns=['step','advantage','clamp_ratio','actor_loss','critic_loss'])
    # df.to_csv(progress_filename_train,index=False)
    
    for i in range(num_agents):
        progress_filename_train = os.path.join(run_dir,f'progress_train_{i}.csv')
        df = pd.DataFrame(columns=[
            'step', 'total_loss', 'actor_loss', 'q1_loss', 'q2_loss',
            'alpha', 'cost_q1_loss', 'cost_q2_loss', 'lambda',
            'cost_estimate', 'cost_gap'
        ])
        df.to_csv(progress_filename_train,index=False)

    train_count = 0
    try:
        while train_count < args.max_episodes:
            train_count = runner.run()
            if train_count % args.eval_episodes == 0 and args.use_eval:
                print("=== {0:>8} is evaluating...".format('Main'))
                runner.eval()
        # from tqdm import tqdm
        # for _ in tqdm(range(50), desc="Evaluating"):
        #     runner.eval()
    except KeyboardInterrupt:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main'))
    
    env.close()

if __name__ == "__main__":
    main(sys.argv[1:])
