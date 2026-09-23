

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from atacom.navigation_manifold import ConstraintManifoldProjector
from .quantum_policy import QuantumQNetwork, QuantumVariationalPolicy


class ReplayBuffer:
    def __init__(self, obs_dim, state_dim, action_dim, capacity, num_agents):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.ptr = 0
        self.size = 0

        #智能体的观测、状态、动作、奖励、下一时刻的观测和状态
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)


        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.prev_action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.safety_cost = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)

        #终止标志
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        #智能体编号,用在约束流行投影的向量化处理
        self.agent_id = np.zeros((self.capacity, 1), dtype=np.int64)

        #当前时刻所有智能体的位置。当前时刻所有智能体的朝向。当前时刻所有智能体的存活状态。当前时刻竞技场中心位置。
        self.positions = np.zeros((self.capacity, self.num_agents, 2), dtype=np.float32)
        self.headings = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.alive_mask = np.zeros((self.capacity, self.num_agents), dtype=np.bool_)
        self.arena_center = np.zeros((self.capacity, 2), dtype=np.float32)

        #下一时刻所有智能体的位置。下一时刻所有智能体的朝向。下一时刻所有智能体的存活状态。下一时刻竞技场中心位置。
        self.next_positions = np.zeros((self.capacity, self.num_agents, 2), dtype=np.float32)
        self.next_headings = np.zeros((self.capacity, self.num_agents), dtype=np.float32)
        self.next_alive_mask = np.zeros((self.capacity, self.num_agents), dtype=np.bool_)
        self.next_arena_center = np.zeros((self.capacity, 2), dtype=np.float32)

    def __len__(self):
        return self.size

    def add_batch(
        self,
        obs,
        state,
        action,
        prev_action,
        reward,
        safety_cost,
        next_obs,
        next_state,
        done,
        projection_context,
        next_projection_context,
        agent_id,
    ):
        batch_size = obs.shape[0]
        idx = np.arange(self.ptr, self.ptr + batch_size) % self.capacity

        self.obs[idx] = obs
        self.state[idx] = state
        self.action[idx] = action
        self.prev_action[idx] = prev_action #上一时刻动作（用于平滑）
        self.reward[idx] = reward
        self.safety_cost[idx] = safety_cost
        self.next_obs[idx] = next_obs
        self.next_state[idx] = next_state
        self.done[idx] = done

        self.positions[idx] = projection_context["positions"]
        self.headings[idx] = projection_context["headings"]
        self.alive_mask[idx] = projection_context["alive_mask"]
        self.arena_center[idx] = projection_context["arena_centers"]
        self.next_positions[idx] = next_projection_context["positions"]
        self.next_headings[idx] = next_projection_context["headings"]
        self.next_alive_mask[idx] = next_projection_context["alive_mask"]
        self.next_arena_center[idx] = next_projection_context["arena_centers"]
        self.agent_id[idx] = agent_id.reshape(-1, 1)

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

#经验回放缓冲区采样函数,供神经网络训练时使用
    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": torch.as_tensor(self.obs[idx], dtype=torch.float32, device=device),
            "state": torch.as_tensor(self.state[idx], dtype=torch.float32, device=device),
            "action": torch.as_tensor(self.action[idx], dtype=torch.float32, device=device),
            "prev_action": torch.as_tensor(self.prev_action[idx], dtype=torch.float32, device=device),
            "reward": torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device),
            "safety_cost": torch.as_tensor(self.safety_cost[idx], dtype=torch.float32, device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "next_state": torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=device),
            "done": torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
            "positions": self.positions[idx],
            "headings": self.headings[idx],
            "alive_mask": self.alive_mask[idx],
            "arena_center": self.arena_center[idx],
            "next_positions": self.next_positions[idx],
            "next_headings": self.next_headings[idx],
            "next_alive_mask": self.next_alive_mask[idx],
            "next_arena_center": self.next_arena_center[idx],
            "agent_id": self.agent_id[idx].reshape(-1),
        }
        return batch


# class GaussianPolicy(nn.Module):
#     def __init__(self, obs_dim, action_dim, hidden_dim, log_std_min, log_std_max, log_std_init):
#         super().__init__()
#         self.log_std_min = log_std_min
#         self.log_std_max = log_std_max

#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
#         self.mean_layer = nn.Linear(hidden_dim, action_dim)
#         self.log_std_layer = nn.Linear(hidden_dim, action_dim)
#         nn.init.uniform_(self.log_std_layer.weight, -1e-3, 1e-3)
#         init_bias = float(np.clip(log_std_init, self.log_std_min, self.log_std_max))
#         nn.init.constant_(self.log_std_layer.bias, init_bias)

#     def forward(self, obs):
#         feat = self.net(obs)
#         mean = self.mean_layer(feat)
#         log_std = self.log_std_layer(feat)
#         log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
#         return mean, log_std

#     def sample(self, obs, deterministic=False):
#         mean, log_std = self.forward(obs)
#         std = torch.exp(log_std)

#         if deterministic:
#             pre_tanh = mean
#             action = torch.tanh(pre_tanh)
#             return action, None

#         normal = torch.distributions.Normal(mean, std)
#         pre_tanh = normal.rsample()
#         action = torch.tanh(pre_tanh)

#         # Tanh correction for reparameterized Gaussian policy.
#         log_prob = normal.log_prob(pre_tanh)
#         log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
#         log_prob = log_prob.sum(dim=-1, keepdim=True)
#         return action, log_prob


class QuantumPolicy(QuantumVariationalPolicy):
    pass


class QNetwork(nn.Module):
    # Critic uses all agent poses, all target positions, and this agent's 2-D action.
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)#Q值 每一个小车一个Q值


class Model(nn.Module):
    def __init__(
        self, obs_dim, state_dim, action_dim, device,
        args, reward_normalize=False, continuous=True,
        manifold_projector=None, world_size=None, max_wheel_speed=None,
    ):
        
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.args = args
        self.reward_normalize = reward_normalize
        self.continuous = continuous

        if not self.continuous:
            raise ValueError("SAC model only supports continuous action space")

        hidden_dim = args.sac_hidden_dim
        self.gamma = args.gamma
        self.tau = args.sac_tau
        self.batch_size = args.batch_size
        self.reward_scale = args.sac_reward_scale
        self.action_low = args.action_low
        self.action_high = args.action_high
        self.num_agents = int(args.num_agents)
        self.use_constraint_manifold = bool(args.use_constraint_manifold)
        self.action_smoothing_beta = float(np.clip(getattr(args, "action_smoothing_beta", 0.0), 0.0, 0.99))

        # QNG is applied to the quantum variational parameters only.  The
        # classical encoders/readout layers continue to use Adam, so the
        # hybrid networks are not updated twice on the same parameter.
        self.qng_actor_interval = max(int(getattr(args, "qng_actor_interval", 2)), 1)
        self.qng_critic_interval = max(int(getattr(args, "qng_critic_interval", 4)), 1)
        self.qng_metric_batch_size = max(
            int(getattr(args, "qng_metric_batch_size", 1)), 1
        )
        self.qng_damping = max(float(getattr(args, "qng_damping", 1e-4)), 0.0)
        self.qng_relative_damping = max(
            float(getattr(args, "qng_relative_damping", 1e-2)), 0.0
        )
        self.qng_max_step_norm = float(getattr(args, "qng_max_step_norm", 0.1))
        self.qng_actor_lr = float(
            getattr(args, "qng_actor_lr", None)
            if getattr(args, "qng_actor_lr", None) is not None
            else args.sac_policy_lr
        )
        qng_critic_lr = getattr(args, "qng_critic_lr", None)
        self.qng_critic_lr = float(
            args.sac_q_lr if qng_critic_lr is None else qng_critic_lr
        )
        self.qng_cost_critic_lr = float(
            args.sac_cost_q_lr if qng_critic_lr is None else qng_critic_lr
        )
        self.train_step_count = 0

        safety_margin = args.manifold_safety_margin
        if safety_margin < 0:
            safety_margin = args.collision_distance
        if manifold_projector is None:
            if world_size is None:
                world_size = [
                    float(getattr(args, "world_size_x", 2.0)),
                    float(getattr(args, "world_size_y", 2.0)),
                ]
            if max_wheel_speed is None:
                max_wheel_speed = 6.28
            manifold_projector = ConstraintManifoldProjector(
                world_size=world_size,
                safety_margin=safety_margin,
                dt=max(args.timestep / 1000.0, 1e-6),
                wheel_radius=args.manifold_wheel_radius,
                axle_length=args.manifold_axle_length,
                max_wheel_speed=max_wheel_speed,
                k_c=args.manifold_kc,
                heading_gain=args.manifold_heading_gain,
                wall_turn_margin=args.manifold_wall_turn_margin,
                agent_avoid_margin=args.manifold_agent_avoid_margin,
            )
        self.manifold_projector = manifold_projector

        self.policy_arch = getattr(args, "policy_arch", "quantum")

        if self.policy_arch == "quantum":
            self.policy = QuantumPolicy(
                obs_dim, action_dim, hidden_dim,
                args.sac_log_std_min, args.sac_log_std_max, args.sac_log_std_init,
                num_qubits=args.quantum_num_qubits,
                reupload_layers=args.quantum_reupload_layers,
            ).to(self.device)
        elif self.policy_arch == "classical":
            self.policy = GaussianPolicy(
                obs_dim, action_dim, hidden_dim,
                args.sac_log_std_min, args.sac_log_std_max, args.sac_log_std_init
            ).to(self.device)
        else:
            raise ValueError(f"Unknown policy_arch: {self.policy_arch}")
        

        self.critic_arch = getattr(args, "critic_arch", "quantum")

        def make_q_network():
            if self.critic_arch == "quantum":
                return QuantumQNetwork(
                    state_dim,
                    action_dim,
                    hidden_dim,
                    num_qubits=getattr(args, "quantum_critic_num_qubits", 8),
                    reupload_layers=getattr(args, "quantum_critic_reupload_layers", 2),
                )
            if self.critic_arch == "classical":
                return QNetwork(state_dim, action_dim, hidden_dim)
            raise ValueError(f"Unknown critic_arch: {self.critic_arch}")

        self.q1 = make_q_network().to(self.device)
        self.q2 = make_q_network().to(self.device)
        self.q1_target = make_q_network().to(self.device)
        self.q2_target = make_q_network().to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Dedicated twin cost critics.  They use the same quantum circuit
        # family as the reward critics but have independent parameters and
        # targets, so reward and safety estimation cannot contaminate each
        # other.
        self.cost_q1 = make_q_network().to(self.device)
        self.cost_q2 = make_q_network().to(self.device)
        self.cost_q1_target = make_q_network().to(self.device)
        self.cost_q2_target = make_q_network().to(self.device)
        self.cost_q1_target.load_state_dict(self.cost_q1.state_dict())
        self.cost_q2_target.load_state_dict(self.cost_q2.state_dict())

        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.cost_q1_target.parameters():
            p.requires_grad = False
        for p in self.cost_q2_target.parameters():
            p.requires_grad = False

        self.policy_opt = torch.optim.Adam(
            self._adam_parameters(self.policy), lr=args.sac_policy_lr
        )
        self.q1_opt = torch.optim.Adam(
            self._adam_parameters(self.q1), lr=args.sac_q_lr
        )
        self.q2_opt = torch.optim.Adam(
            self._adam_parameters(self.q2), lr=args.sac_q_lr
        )
        self.cost_q1_opt = torch.optim.Adam(
            self._adam_parameters(self.cost_q1), lr=args.sac_cost_q_lr
        )
        self.cost_q2_opt = torch.optim.Adam(
            self._adam_parameters(self.cost_q2), lr=args.sac_cost_q_lr
        )

        self.safety_cost_limit = float(args.safety_cost_limit)
        self.safety_cost_max = max(float(getattr(args, "safety_cost_max", 1.0)), 1e-6)
        self.safety_cost_discount_normalized = bool(
            getattr(args, "safety_cost_discount_normalized", True)
        )
        self.lambda_max = max(float(args.safety_lambda_max), 1e-6)
        lambda_init = np.clip(
            float(args.safety_lambda_init), 1e-6, self.lambda_max
        )
        self.log_lambda = nn.Parameter(
            torch.tensor(np.log(lambda_init), dtype=torch.float32, device=self.device)
        )
        self.lambda_opt = torch.optim.Adam([self.log_lambda], lr=args.safety_lambda_lr)
        self.cost_ema_beta = float(np.clip(args.safety_cost_ema_beta, 0.0, 0.9999))
        self.cost_ema = None

        self.auto_alpha = args.sac_auto_alpha
        self.target_entropy = (
            -float(action_dim)
            if args.sac_target_entropy is None
            else float(args.sac_target_entropy)
        )

        init_alpha = max(float(args.sac_alpha), 1e-6)
        if self.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32,
                requires_grad=True, device=self.device
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=args.sac_alpha_lr)
        else:
            self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device)
            self.alpha_opt = None

        self.replay_buffer = ReplayBuffer(
            obs_dim, state_dim, action_dim, args.sac_replay_size, self.num_agents
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def lagrange_multiplier(self):
        return torch.clamp(self.log_lambda.exp(), 0.0, self.lambda_max)

    def choose_action(self, obs, state=None, mask=None, phase="train"):
        del state, mask
        self.eval()

        deterministic = (phase == "eval")
        obs_flat = obs.reshape(-1, self.obs_dim)
        with torch.no_grad():
            action_flat, _ = self.policy.sample(obs_flat, deterministic=deterministic)
        action_flat = torch.clamp(action_flat, self.action_low, self.action_high)

        action = action_flat.reshape(*obs.shape[:-1], self.action_dim)
        return action.cpu().numpy()

    def _smooth_policy_action(self, raw_action, prev_action):
        raw_action = torch.clamp(raw_action, self.action_low, self.action_high)
        if self.action_smoothing_beta <= 0.0:
            return raw_action

        smoothed = (
            self.action_smoothing_beta * prev_action
            + (1.0 - self.action_smoothing_beta) * raw_action
        )
        return torch.clamp(smoothed, self.action_low, self.action_high)

    def _project_policy_action(self, action, batch, prefix="", straight_through=True):
        if prefix:
            positions = batch[f"{prefix}_positions"]
            headings = batch[f"{prefix}_headings"]
            alive_mask = batch[f"{prefix}_alive_mask"]
            arena_center = batch[f"{prefix}_arena_center"]
        else:
            positions = batch["positions"]
            headings = batch["headings"]
            alive_mask = batch["alive_mask"]
            arena_center = batch["arena_center"]

        if self.use_constraint_manifold:
            projected_np = self.manifold_projector.project_agent_batch(
                action.detach().cpu().numpy(),
                positions,
                headings,
                alive_mask,
                arena_center,
                batch["agent_id"],
            )
        else:
            projected_np = action.detach().cpu().numpy()

        projected = torch.as_tensor(projected_np, dtype=action.dtype, device=action.device)
        if straight_through:
            return action + (projected - action).detach()
        return projected

    def _executed_policy_action(self, raw_action, prev_action, batch, prefix="", straight_through=True):
        smoothed_action = self._smooth_policy_action(raw_action, prev_action)
        return self._project_policy_action(
            smoothed_action,
            batch,
            prefix=prefix,
            straight_through=straight_through,
        )

    def store_transitions(
        self,
        obs,
        state,
        action,
        prev_action,
        reward,
        safety_cost,
        next_obs,
        next_state,
        done,
        projection_context,
        next_projection_context,
        agent_id,
    ):
        obs = np.asarray(obs, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        prev_action = np.asarray(prev_action, dtype=np.float32)
        reward = np.asarray(reward, dtype=np.float32).reshape(-1, 1)
        safety_cost = np.asarray(safety_cost, dtype=np.float32).reshape(-1, 1)
        safety_cost = np.nan_to_num(safety_cost, nan=0.0, posinf=1.0, neginf=0.0)
        safety_cost = np.clip(safety_cost / self.safety_cost_max, 0.0, 1.0)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32).reshape(-1, 1)
        agent_id = np.asarray(agent_id, dtype=np.int64).reshape(-1)

        self.replay_buffer.add_batch(
            obs,
            state,
            action,
            prev_action,
            reward,
            safety_cost,
            next_obs,
            next_state,
            done,
            projection_context,
            next_projection_context,
            agent_id,
        )

    def can_update(self):
        return len(self.replay_buffer) >= self.batch_size

    def _soft_update(self, online_net, target_net):
        for p, p_target in zip(online_net.parameters(), target_net.parameters()):
            p_target.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_target.data)

    @staticmethod
    def _set_requires_grad(networks, enabled):
        for network in networks:
            for parameter in network.parameters():
                parameter.requires_grad_(enabled)

    @staticmethod
    def _adam_parameters(network):
        """Exclude the quantum variational tensor from the Adam optimizers."""
        return [
            parameter
            for name, parameter in network.named_parameters()
            if name != "var_params"
        ]

    @staticmethod
    def _clear_quantum_grad(network):
        parameter = getattr(network, "var_params", None)
        if parameter is not None:
            parameter.grad = None

    def _apply_qng_update(self, network, encoded_angles, learning_rate):
        """Apply one damped natural-gradient update to ``network.var_params``.

        The Fisher matrix is calculated from detached circuit inputs and
        parameters, so no graph from the SAC loss is retained.  Eigenvalue
        flooring makes the solve stable even when the circuit contains
        locally redundant rotations.
        """
        parameter = getattr(network, "var_params", None)
        if parameter is None or parameter.grad is None:
            return {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}

        gradient = parameter.grad.detach().reshape(-1)
        if not torch.isfinite(gradient).all():
            parameter.grad = None
            return {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}

        # ``quantum_fisher`` internally differentiates the state with respect
        # to a temporary parameter tensor, so it must not be wrapped in
        # ``torch.no_grad`` even though the returned metric is detached.
        fisher = network.quantum_fisher(
            encoded_angles.detach(),
            max_samples=self.qng_metric_batch_size,
        )
        fisher = 0.5 * (fisher + fisher.transpose(0, 1))
        gradient = gradient.to(device=fisher.device, dtype=fisher.dtype)
        if not torch.isfinite(fisher).all():
            parameter.grad = None
            return {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}

        parameter_count = gradient.numel()
        mean_diagonal = torch.trace(fisher) / float(parameter_count)
        damping = self.qng_damping + self.qng_relative_damping * mean_diagonal.abs()
        damping = damping.clamp_min(1e-8)

        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
            eigenvalues = eigenvalues.clamp_min(damping)
            direction = eigenvectors @ (
                eigenvectors.transpose(0, 1) @ gradient / eigenvalues
            )
            condition = float(
                (eigenvalues.max() / eigenvalues.min()).detach().cpu().item()
            )
        except RuntimeError:
            regularized = fisher + damping * torch.eye(
                parameter_count, device=fisher.device, dtype=fisher.dtype
            )
            direction = torch.linalg.solve(regularized, gradient)
            condition = 0.0

        update = float(learning_rate) * direction
        update_norm_tensor = torch.linalg.vector_norm(update)
        if (
            self.qng_max_step_norm > 0.0
            and update_norm_tensor > self.qng_max_step_norm
        ):
            update = update * (self.qng_max_step_norm / update_norm_tensor)
            update_norm_tensor = torch.linalg.vector_norm(update)
        update_norm = float(update_norm_tensor.detach().cpu().item())
        with torch.no_grad():
            parameter.add_(-update.to(device=parameter.device, dtype=parameter.dtype).view_as(parameter))
        parameter.grad = None
        return {
            "updated": 1.0,
            "step_norm": update_norm,
            "condition": condition,
        }

    def train_step(self):
        if not self.can_update():
            return None

        self.train_step_count += 1
        qng_actor_due = (
            self.policy_arch == "quantum"
            and self.train_step_count % self.qng_actor_interval == 0
        )
        qng_critic_due = (
            self.critic_arch == "quantum"
            and self.train_step_count % self.qng_critic_interval == 0
        )

        self.train()
        batch = self.replay_buffer.sample(self.batch_size, self.device)

        obs = batch["obs"]
        state = batch["state"]
        action = batch["action"]
        prev_action = batch["prev_action"]
        reward = batch["reward"]
        safety_cost = batch["safety_cost"]
        next_obs = batch["next_obs"]
        next_state = batch["next_state"]
        done = batch["done"]

        if self.reward_normalize:
            reward = (reward - reward.mean()) / (reward.std() + 1e-6)

        with torch.no_grad():
            next_raw_action, next_log_prob = self.policy.sample(next_obs, deterministic=False)
            next_action = self._executed_policy_action(
                next_raw_action,
                action,
                batch,
                prefix="next",
                straight_through=False,
            )
            target_q1 = self.q1_target(next_state, next_action)
            target_q2 = self.q2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_value = self.reward_scale * reward + (1.0 - done) * self.gamma * target_q

            target_cost_q1 = self.cost_q1_target(next_state, next_action)
            target_cost_q2 = self.cost_q2_target(next_state, next_action)
            target_cost_q = torch.clamp(
                torch.min(target_cost_q1, target_cost_q2), min=0.0
            )
            if self.safety_cost_discount_normalized:
                target_cost = (
                    (1.0 - self.gamma) * safety_cost
                    + (1.0 - done) * self.gamma * target_cost_q
                )
            else:
                target_cost = safety_cost + (1.0 - done) * self.gamma * target_cost_q

        q1_pred = self.q1(state, action)
        q2_pred = self.q2(state, action)
        q1_loss = F.mse_loss(q1_pred, target_value)
        q2_loss = F.mse_loss(q2_pred, target_value)

        self._clear_quantum_grad(self.q1)
        self.q1_opt.zero_grad()
        q1_loss.backward()
        q1_qng_info = {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}
        if qng_critic_due and hasattr(self.q1, "quantum_fisher"):
            q1_angles = self.q1._encode_input(
                torch.cat([state.detach(), action.detach()], dim=-1),
                action.detach(),
            )
            q1_qng_info = self._apply_qng_update(self.q1, q1_angles, self.qng_critic_lr)
        self.q1_opt.step()
        if not q1_qng_info["updated"]:
            self._clear_quantum_grad(self.q1)

        self._clear_quantum_grad(self.q2)
        self.q2_opt.zero_grad()
        q2_loss.backward()
        q2_qng_info = {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}
        if qng_critic_due and hasattr(self.q2, "quantum_fisher"):
            q2_angles = self.q2._encode_input(
                torch.cat([state.detach(), action.detach()], dim=-1),
                action.detach(),
            )
            q2_qng_info = self._apply_qng_update(self.q2, q2_angles, self.qng_critic_lr)
        self.q2_opt.step()
        if not q2_qng_info["updated"]:
            self._clear_quantum_grad(self.q2)

        cost_q1_pred = self.cost_q1(state, action)
        cost_q2_pred = self.cost_q2(state, action)
        cost_q1_loss = F.mse_loss(cost_q1_pred, target_cost)
        cost_q2_loss = F.mse_loss(cost_q2_pred, target_cost)

        self._clear_quantum_grad(self.cost_q1)
        self.cost_q1_opt.zero_grad()
        cost_q1_loss.backward()
        cost_q1_qng_info = {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}
        if qng_critic_due and hasattr(self.cost_q1, "quantum_fisher"):
            cost_q1_angles = self.cost_q1._encode_input(
                torch.cat([state.detach(), action.detach()], dim=-1),
                action.detach(),
            )
            cost_q1_qng_info = self._apply_qng_update(
                self.cost_q1, cost_q1_angles, self.qng_cost_critic_lr
            )
        self.cost_q1_opt.step()
        if not cost_q1_qng_info["updated"]:
            self._clear_quantum_grad(self.cost_q1)

        self._clear_quantum_grad(self.cost_q2)
        self.cost_q2_opt.zero_grad()
        cost_q2_loss.backward()
        cost_q2_qng_info = {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}
        if qng_critic_due and hasattr(self.cost_q2, "quantum_fisher"):
            cost_q2_angles = self.cost_q2._encode_input(
                torch.cat([state.detach(), action.detach()], dim=-1),
                action.detach(),
            )
            cost_q2_qng_info = self._apply_qng_update(
                self.cost_q2, cost_q2_angles, self.qng_cost_critic_lr
            )
        self.cost_q2_opt.step()
        if not cost_q2_qng_info["updated"]:
            self._clear_quantum_grad(self.cost_q2)

        new_raw_action, log_prob = self.policy.sample(obs, deterministic=False)
        new_action = self._executed_policy_action(
            new_raw_action,
            prev_action,
            batch,
            straight_through=True,
        )
        # Freeze critic parameters during the actor update while preserving
        # the gradient through their action input.
        self._set_requires_grad(
            (self.q1, self.q2, self.cost_q1, self.cost_q2), False
        )
        q1_new = self.q1(state, new_action)
        q2_new = self.q2(state, new_action)
        q_new = torch.min(q1_new, q2_new)
        cost_q1_new = self.cost_q1(state, new_action)
        cost_q2_new = self.cost_q2(state, new_action)
        cost_q_new = torch.clamp(torch.max(cost_q1_new, cost_q2_new), min=0.0)
        lagrange = self.lagrange_multiplier.detach()
        actor_loss = (
            self.alpha.detach() * log_prob
            - q_new
            + lagrange * cost_q_new
        ).mean()

        self._clear_quantum_grad(self.policy)
        self.policy_opt.zero_grad()
        actor_loss.backward()
        actor_qng_info = {"updated": 0.0, "step_norm": 0.0, "condition": 0.0}
        if qng_actor_due and hasattr(self.policy, "quantum_fisher"):
            with torch.no_grad():
                actor_angles = self.policy._encode_obs(obs.detach())
            actor_qng_info = self._apply_qng_update(
                self.policy, actor_angles, self.qng_actor_lr
            )
        self.policy_opt.step()
        if not actor_qng_info["updated"]:
            self._clear_quantum_grad(self.policy)
        self._set_requires_grad(
            (self.q1, self.q2, self.cost_q1, self.cost_q2), True
        )

        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros(1, device=self.device)

        # Dual ascent: minimizing -log(lambda) * (Jc - limit) increases
        # lambda when the estimated discounted danger exceeds the budget.
        cost_estimate = float(cost_q_new.detach().mean().cpu().item())
        if self.cost_ema is None:
            self.cost_ema = cost_estimate
        else:
            self.cost_ema = (
                self.cost_ema_beta * self.cost_ema
                + (1.0 - self.cost_ema_beta) * cost_estimate
            )
        cost_gap = self.cost_ema - self.safety_cost_limit
        lambda_gap = torch.as_tensor(cost_gap, dtype=torch.float32, device=self.device)
        lambda_loss = -(self.log_lambda * lambda_gap.detach())
        self.lambda_opt.zero_grad()
        lambda_loss.backward()
        self.lambda_opt.step()
        with torch.no_grad():
            self.log_lambda.clamp_(
                min=-20.0,
                max=float(np.log(self.lambda_max)),
            )

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        self._soft_update(self.cost_q1, self.cost_q1_target)
        self._soft_update(self.cost_q2, self.cost_q2_target)

        return {
            "total_loss": float(
                (actor_loss + q1_loss + q2_loss + cost_q1_loss + cost_q2_loss)
                .detach().cpu().item()
            ),
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "q1_loss": float(q1_loss.detach().cpu().item()),
            "q2_loss": float(q2_loss.detach().cpu().item()),
            "cost_q1_loss": float(cost_q1_loss.detach().cpu().item()),
            "cost_q2_loss": float(cost_q2_loss.detach().cpu().item()),
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "alpha": float(self.alpha.detach().cpu().item()),
            "lambda_loss": float(lambda_loss.detach().cpu().item()),
            "lambda": float(self.lagrange_multiplier.detach().cpu().item()),
            "cost_estimate": cost_estimate,
            "cost_gap": float(cost_gap),
            "train_step": self.train_step_count,
            "qng_actor_updated": actor_qng_info["updated"],
            "qng_actor_step_norm": actor_qng_info["step_norm"],
            "qng_critic_updated": float(
                q1_qng_info["updated"]
                + q2_qng_info["updated"]
                + cost_q1_qng_info["updated"]
                + cost_q2_qng_info["updated"]
            ),
            "qng_critic_step_norm": float(
                q1_qng_info["step_norm"]
                + q2_qng_info["step_norm"]
                + cost_q1_qng_info["step_norm"]
                + cost_q2_qng_info["step_norm"]
            ),
        }
