import argparse


parser = argparse.ArgumentParser()

# prepare parameters
parser.add_argument("--algorithm_name", type=str, default="SAC")
parser.add_argument("--policy_arch", type=str, default="quantum", choices=["classical", "quantum"],
                    help="Actor architecture to use")
parser.add_argument("--experiment_name", type=str, default="check")
parser.add_argument("--seed", type=int, default=1,
                    help="Random seed for numpy/torch")
parser.add_argument("--cuda", action='store_false', default=False)
parser.add_argument("--cuda_deterministic",
                    action='store_false', default=True)
parser.add_argument('--n_training_threads', type=int,
                    default=1, help="Number of torch threads for training")
parser.add_argument('--n_rollout_threads', type=int,  default=1,
                    help="Number of parallel envs for training rollout")
parser.add_argument('--num_env_steps', type=int,
                    default=2000000, help="Number of env steps to train for")
parser.add_argument('--use_wandb', action='store_false', default=True,
                    help="Whether to use weights&biases, if not, use tensorboardX instead")
parser.add_argument('--user_name', type=str, default="yfw")

# env parameters
parser.add_argument('--env_name', type=str, default="navigation")
parser.add_argument("--use_obs_instead_of_state", action='store_true',
                    default=False, help="Whether to use global state or concatenated obs")
parser.add_argument('--signal_strength', type=float, default=80)
parser.add_argument('--max_signal_strength', type=float, default=1600)
parser.add_argument('--noise_dis', type=float, default=0.02,
                    help="Standard deviation of distance noise")
parser.add_argument('--noise_angle', type=float, default=0.05,
                    help="Standard deviation of angle noise (rad)")
parser.add_argument('--update_obs_steps', type=int, default=40,
                    help="Number of steps to update global observations")

# replay buffer parameters
parser.add_argument('--buffer_length', type=int,
                    default=800, help="Max length for any episode")#单回合最大长度是800
parser.add_argument('--reward_normalize', action='store_true',
                    default=False, help="Whether to normalize rewards in replay buffer")

parser.add_argument('--lr', type=float, default=5e-4,
                    help="Learning rate for Adam")
parser.add_argument('--action_low', type=float, default=-1.0,
                    help="Lower bound of normalized continuous actions")
parser.add_argument('--action_high', type=float, default=1.0,
                    help="Upper bound of normalized continuous actions")

# SAC parameters
parser.add_argument('--sac_hidden_dim', type=int, default=128,
                    help="Hidden size used by SAC actor/critic networks")
parser.add_argument("--critic_arch", type=str, default="quantum", choices=["classical", "quantum"],
                    help="Critic architecture to use")
parser.add_argument('--quantum_num_qubits', type=int, default=6,
                    help="Number of qubits used by the quantum policy")
parser.add_argument('--quantum_reupload_layers', type=int, default=2,
                    help="Number of data re-uploading layers in the quantum policy")
parser.add_argument('--quantum_critic_num_qubits', type=int, default=8,
                    help="Number of qubits used by the quantum critic")
parser.add_argument('--quantum_critic_reupload_layers', type=int, default=2,
                    help="Number of data re-uploading layers in the quantum critic")
parser.add_argument('--sac_replay_size', type=int, default=200000,
                    help="Replay buffer capacity for SAC")
parser.add_argument('--sac_start_steps', type=int, default=1000,
                    help="Warmup steps sampled by random policy before SAC updates")
parser.add_argument('--sac_updates_per_step', type=int, default=1,
                    help="Number of SAC gradient updates per environment step")
parser.add_argument('--sac_tau', type=float, default=0.005,
                    help="Soft target update coefficient for SAC")
parser.add_argument('--sac_policy_lr', type=float, default=3e-4,
                    help="Learning rate of SAC actor")
parser.add_argument('--sac_q_lr', type=float, default=3e-4,
                    help="Learning rate of SAC Q networks")
parser.add_argument('--sac_cost_q_lr', type=float, default=3e-4,
                    help="Learning rate of the twin safety-cost Q networks")
parser.add_argument('--qng_actor_interval', type=int, default=2,
                    help="Apply the actor quantum natural-gradient update every N SAC updates")
parser.add_argument('--qng_critic_interval', type=int, default=4,
                    help="Apply critic quantum natural-gradient updates every N SAC updates")
parser.add_argument('--qng_actor_lr', type=float, default=None,
                    help="Quantum natural-gradient learning rate for the actor; defaults to sac_policy_lr")
parser.add_argument('--qng_critic_lr', type=float, default=None,
                    help="Quantum natural-gradient learning rate for critics; defaults to sac_q_lr")
parser.add_argument('--qng_metric_batch_size', type=int, default=1,
                    help="Maximum replay samples used to estimate each quantum Fisher matrix")
parser.add_argument('--qng_damping', type=float, default=1e-4,
                    help="Absolute diagonal damping used when solving the QNG system")
parser.add_argument('--qng_relative_damping', type=float, default=1e-2,
                    help="Damping multiplied by the mean Fisher diagonal")
parser.add_argument('--qng_max_step_norm', type=float, default=0.1,
                    help="Maximum norm of a QNG parameter step; <=0 disables clipping")
parser.add_argument('--sac_alpha_lr', type=float, default=3e-4,
                    help="Learning rate of SAC temperature")
parser.add_argument('--sac_alpha', type=float, default=0.2,
                    help="Initial entropy temperature for SAC")
parser.add_argument('--sac_auto_alpha', action='store_false', default=True,
                    help="Whether to auto-tune SAC entropy temperature")
parser.add_argument('--sac_target_entropy', type=float, default=None,
                    help="Target entropy for SAC alpha tuning. None -> -action_dim")
parser.add_argument('--sac_log_std_min', type=float, default=-20.0,
                    help="Minimum log std for SAC Gaussian actor")
parser.add_argument('--sac_log_std_max', type=float, default=0.2,
                    help="Maximum log std for SAC Gaussian actor (smaller -> less exploration noise)")
parser.add_argument('--sac_log_std_init', type=float, default=-1.5,
                    help="Initial log std bias for SAC actor (smaller -> lower initial action noise)")
parser.add_argument('--sac_reward_scale', type=float, default=1.0,
                    help="Scale factor applied to rewards before SAC target computation")
parser.add_argument('--action_smoothing_beta', type=float, default=0.6,
                    help="EMA smoothing factor for executed actions in [0, 1). Higher -> smoother actions")

# Constrained-SAC safety cost parameters.  Costs are per active agent and are
# normalized to [0, 1]; the normalized discounted critic return is therefore
# also approximately in [0, 1].
parser.add_argument('--safety_cost_collision_weight', type=float, default=1.0,
                    help='Weight of the binary proximity-sensor safety cost')
parser.add_argument('--safety_cost_wall_weight', type=float, default=0.5,
                    help='Weight of the dense wall-proximity safety cost')
parser.add_argument('--safety_cost_agent_weight', type=float, default=0.5,
                    help='Weight of the dense nearest-agent safety cost')
parser.add_argument('--safety_cost_max', type=float, default=1.0,
                    help='Maximum clipped per-step safety cost')
parser.add_argument('--safety_cost_wall_margin', type=float, default=None,
                    help='Wall distance at which dense wall cost starts; None uses manifold_wall_turn_margin')
parser.add_argument('--safety_cost_agent_margin', type=float, default=None,
                    help='Agent distance at which dense inter-agent cost starts; None uses manifold_agent_avoid_margin')
parser.add_argument('--safety_cost_limit', type=float, default=0.05,
                    help='Normalized discounted safety-cost budget')
parser.add_argument('--no_safety_cost_discount_normalized',
                    dest='safety_cost_discount_normalized', action='store_false', default=True,
                    help='Disable (1-gamma)*cost normalization and use an ordinary discounted cost return')
parser.add_argument('--safety_lambda_init', type=float, default=0.1,
                    help='Initial Lagrange multiplier')
parser.add_argument('--safety_lambda_max', type=float, default=20.0,
                    help='Upper bound for the Lagrange multiplier')
parser.add_argument('--safety_lambda_lr', type=float, default=1e-3,
                    help='Learning rate of the Lagrange multiplier')
parser.add_argument('--safety_cost_ema_beta', type=float, default=0.95,
                    help='EMA coefficient for cost-critic values used by lambda update')

# algo common parameters
parser.add_argument('--batch_size', type=int, default=32,
                    help="Number of buffer transitions to train on at once")
parser.add_argument('--gamma', type=float, default=0.99,
                    help="Discount factor for env")
parser.add_argument('--gae_lambda', type=float, default=0.95,
                    help="Lambda coefficient in GAE formula")
parser.add_argument('--grad_norm_init', type=float, default=100)




# exploration parameters
parser.add_argument('--exclude_steps', type=int, default=3,
                    help="Number of steps to exclude at the beginning of each episode")
parser.add_argument('--num_random_episodes', type=int, default=4,
                    help="Number of episodes to add to buffer with purely random actions")
parser.add_argument('--epsilon_start', type=float, default=1.0,
                    help="Starting value for epsilon, for eps-greedy exploration")
parser.add_argument('--epsilon_finish', type=float, default=0.05,
                    help="Ending value for epsilon, for eps-greedy exploration")
parser.add_argument('--max_episodes', type=int, default=8000)
parser.add_argument("--use_value_active_masks",
                    action='store_false', default=True)


# eval parameters
parser.add_argument('--use_eval', action='store_false',
                    default=True, help="Whether to conduct the evaluation")
parser.add_argument('--num_eval_episodes', type=int, default=1,
                    help="How many episodes to collect for each eval")
parser.add_argument('--eval_episodes', type=int,  default=10,
                    help="After how many episodes the policy should be evaled")
parser.add_argument('--eval_seed', type=int, default=10000,
                    help="Base seed used to make evaluation episodes comparable across checkpoints")
parser.add_argument('--train_reward_ema_beta', type=float, default=0.9,
                    help="EMA smoothing factor for training episode metrics logged to TensorBoard")

# pretained parameters
parser.add_argument("--model_dir", type=str, default=None)




# webots experiment parameters(pp)
parser.add_argument("--num_obs_targets", type=int, default=1)
parser.add_argument("--num_obs_agents", type=int, default=3)
parser.add_argument("--obs_view", type=float, default=0.6)
parser.add_argument("--rab_view", type=float, default=0.2)
parser.add_argument("--comm_view", type=float, default=0.8)
parser.add_argument("--catch_distance", type=float, default=0.20)
parser.add_argument("--collision_distance", type=float, default=0.1)
parser.add_argument("--collision_reward", type=float, default=-0.1)
parser.add_argument("--world_size_x", type=float, default=2.0,
                    help="Navigation arena width in meters")
parser.add_argument("--world_size_y", type=float, default=2.0,
                    help="Navigation arena height in meters")
parser.add_argument("--spawn_grid_size", type=float, default=0.15,
                    help="Approximate grid spacing used for reset position candidates")
parser.add_argument("--spawn_wall_margin", type=float, default=0.2,
                    help="Distance kept from physical walls when sampling reset positions")
parser.add_argument("--spawn_split_margin", type=float, default=0.2,
                    help="Gap kept around y=0 between lower agent spawn area and upper target spawn area")
parser.add_argument("--min_agent_distance", type=float, default=0.3,
                    help="Minimum reset distance between two agents")
parser.add_argument("--min_target_distance", type=float, default=0.45,
                    help="Minimum reset distance between two targets")
parser.add_argument("--min_agent_target_distance", type=float, default=0.35,
                    help="Minimum reset distance between any agent and target")
parser.add_argument("--spawn_sample_max_tries", type=int, default=500,
                    help="Maximum random attempts for constrained reset position sampling")
parser.add_argument('--reward_progress_weight', type=float, default=1.0,
                    help='Weight of non-repeatable best-distance progress reward term')
parser.add_argument('--reward_signed_progress_weight', type=float, default=0.5,
                    help='Weight of signed step-to-step nearest-target progress reward')
parser.add_argument('--reward_reach_weight', type=float, default=4.0,
                    help='Individual reward for the agent that captures a target')
parser.add_argument('--reward_team_reach_weight', type=float, default=1.0,
                    help='Shared team reward for each target captured')
parser.add_argument('--reward_finish_weight', type=float, default=6.0,
                    help='Shared bonus for finishing all targets, scaled by remaining time')
parser.add_argument('--reward_collision_weight', type=float, default=0.0001,
                    help='Weight of collision penalty term')
parser.add_argument('--reward_agent_proximity_weight', type=float, default=0.5,
                    help='Penalty weight for entering the soft inter-agent avoidance zone')
parser.add_argument('--reward_time_penalty', type=float, default=-0.01,
                    help='Per-step time penalty for alive agents')
parser.add_argument('--reward_progress_scale', type=float, default=0.05,
                    help='Distance delta scale used to normalize progress reward')
parser.add_argument("--timestep", type=int, default=240)#0.24s 时间步
parser.add_argument("--interval", type=int, default=10)#10
parser.add_argument("--num_target", type=int, default=6)
parser.add_argument("--agent_loc_dim", type=int, default=6)
parser.add_argument("--mutual_dim", type=int, default=4)
parser.add_argument('--num_agents', type=int, default=6, help="number of agents")
parser.add_argument('--capture_action_conditions', type=list, default=[1, 2],
                    help="number of agents that have to simultaneously execute catch action")
parser.add_argument('--capture_action', action='store_false', default=True,
                    help="whether capturing requires an extra action (True) or just capture_conditions (False)")
parser.add_argument('--use_constraint_manifold', action='store_false', default=True,
                    help='Enable ATACOM-style tangent-space action projection safety layer')
parser.add_argument('--manifold_kc', type=float, default=6.0,
                    help='Error-correction gain for manifold projection under discrete simulation')
parser.add_argument('--manifold_heading_gain', type=float, default=2.0,
                    help='Heading correction gain that aligns wheel steering with projected safe velocity')
parser.add_argument('--manifold_safety_margin', type=float, default=-1.0,
                    help='Safety margin for wall/agent constraints. Negative means using collision_distance')
parser.add_argument('--manifold_wall_turn_margin', type=float, default=0.2,
                    help='Soft wall margin that triggers short inward turning correction')
parser.add_argument('--manifold_agent_avoid_margin', type=float, default=0.2,
                    help='Soft inter-agent margin that triggers short avoidance correction')
parser.add_argument('--manifold_wheel_radius', type=float, default=0.0205,
                    help='Differential-drive wheel radius for wheel/velocity conversion')
parser.add_argument('--manifold_axle_length', type=float, default=0.053,
                    help='Differential-drive wheel axle length for wheel/velocity conversion')
