
#采样-训练-评估-记录
#1. 执行一次训练回合：run()
#2. 随机探索/策略动作切换、动作平滑。
#3. 环境交互采样：reset/step 收集 obs, action, reward, done。
#4. 经验入回放池：把 transition()约束流行变化之后的动作 写入
#5. SAC更新
#6. 评估与存档




import os
import random

import numpy as np
import pandas as pd
import torch

from base_runner import BaseRunner


class MyRunner(BaseRunner):
    def __init__(self, config):
        super(MyRunner, self).__init__(config)
        #平滑执行动作
        self.action_smoothing_beta = float(np.clip(getattr(self.args, "action_smoothing_beta", 0.0), 0.0, 0.99))
        #平滑训练日志指标
        self.train_reward_ema_beta = float(np.clip(getattr(self.args, "train_reward_ema_beta", 0.0), 0.0, 0.999))
        self.train_ema_infos = {}

    def run(self):#run() 一次就是一次训练 episode
        self.collect_rollout(phase="train")

        print(
            "Env-{}, Algo-{}, runs total num episodes-{}/{}.\n".format(
                self.env_name,
                self.algorithm_name,
                self.train_count,
                self.max_episodes,
            )
        )

        return self.train_count

    #根据当前观测 obs 生成动作
    def _sample_actions(self, obs, phase):
        #sac_start_steps 1000开始,1000/6 = 167步之后，开始使用策略网络生成动作，之前都是随机动作
        if phase == "train" and len(self.model.replay_buffer) < self.args.sac_start_steps:
            return np.random.uniform(
                self.args.action_low,
                self.args.action_high,
                size=(self.num_envs, self.num_agents, self.env.num_actions),
            ).astype(np.float32)

        return self.model.choose_action(
            torch.tensor(obs, dtype=torch.float32, device=self.device),
            phase=phase,
        )

    def _maybe_update_model(self):
        if len(self.model.replay_buffer) < max(self.batch_size, self.args.sac_start_steps):
            return

        for _ in range(self.args.sac_updates_per_step):
            info = self.model.train_step()#参数更新
            if info is None:
                break
            self.total_train_steps += 1
            self.log_train_update(info)

    #时间序列（当前动作和上一步动作）,平滑动作，避免动作过于剧烈
    def _smooth_actions(self, actions, prev_actions):
        if self.action_smoothing_beta <= 0.0:
            return actions
        beta = self.action_smoothing_beta
        smoothed = beta * prev_actions + (1.0 - beta) * actions
        return np.clip(smoothed, self.args.action_low, self.args.action_high)


    def _flatten_projection_context(self, context):
        env_ids = np.repeat(np.arange(self.num_envs), self.num_agents)
        return {
            "positions": context["positions"][env_ids],
            "headings": context["headings"][env_ids],
            "alive_mask": context["alive_mask"][env_ids],
            "arena_centers": context["arena_centers"][env_ids],
        }

    def _filter_projection_context(self, context, mask):
        return {key: value[mask] for key, value in context.items()}

    def _flat_agent_ids(self):
        return np.tile(np.arange(self.num_agents), self.num_envs)

    def _summarize_episode(
        self,
        episode_rewards,
        episode_actions,
        episode_costs,
        episode_projection_costs,
        episode_traj_masks,
    ):
        rewards = np.array(episode_rewards, dtype=np.float32).squeeze(-1)
        if rewards.ndim == 2:
            rewards = rewards.reshape(rewards.shape[0], self.num_envs, self.num_agents)

        agent_rewards = []
        for agent_id in range(self.num_agents):
            env_rewards = []
            for env_id in range(self.num_envs):
                traj_len = int(np.clip(episode_traj_masks[env_id, agent_id], 1, rewards.shape[0]))
                env_rewards.append(float(rewards[:traj_len, env_id, agent_id].sum()))
            agent_rewards.append(float(np.mean(env_rewards)))

        mean_reward = float(np.mean(agent_rewards))
        rewards_infos = {f"{i}_episode_r": agent_rewards[i] for i in range(self.num_agents)}
        rewards_infos["mean_episode_r"] = mean_reward

        actions = np.array(episode_actions, dtype=np.float32)
        if actions.ndim == 3:
            actions = actions.reshape(actions.shape[0], self.num_envs, self.num_agents, self.env.num_actions)

        action_infos = {
            "action_left_mean": float(actions[..., 0].mean()),
            "action_right_mean": float(actions[..., 1].mean()),
            "action_abs_mean": float(np.abs(actions).mean()),
            "action_std": float(actions.std()),
        }

        costs = np.asarray(episode_costs, dtype=np.float32)
        projection_costs = np.asarray(episode_projection_costs, dtype=np.float32)
        if costs.ndim == 2:
            costs = costs.reshape(costs.shape[0], self.num_envs, self.num_agents)
        if projection_costs.ndim == 2:
            projection_costs = projection_costs.reshape(
                projection_costs.shape[0], self.num_envs, self.num_agents
            )
        cost_sum_by_agent = []
        projection_sum_by_agent = []
        active_step_count = 0
        for agent_id in range(self.num_agents):
            env_costs = []
            env_projection_costs = []
            for env_id in range(self.num_envs):
                traj_len = int(np.clip(episode_traj_masks[env_id, agent_id], 1, costs.shape[0]))
                env_costs.append(float(costs[:traj_len, env_id, agent_id].sum()))
                env_projection_costs.append(
                    float(projection_costs[:traj_len, env_id, agent_id].sum())
                )
                active_step_count += traj_len
            cost_sum_by_agent.append(float(np.mean(env_costs)))
            projection_sum_by_agent.append(float(np.mean(env_projection_costs)))

        safety_infos = {
            "mean_episode_cost": float(np.mean(cost_sum_by_agent)),
            "safety_cost_rate": float(costs.sum() / max(active_step_count, 1)),
            "projection_intervention_rate": float(
                projection_costs.sum() / max(active_step_count, 1)
            ),
        }

        episode_num_target = float((~self.env.alive_target_buf).sum() / self.num_envs)
        episode_num_collision = int(self.env.num_collision.sum())
        episode_finish_time = float(self.env.steps.sum() / self.num_envs)

        episode_infos = {
            "num_target": np.array(episode_num_target),
            "num_collision": np.array(episode_num_collision, dtype=np.int32),
            "finish_time": np.array(episode_finish_time),
        }
        episode_infos = {
            **rewards_infos,
            **action_infos,
            **safety_infos,
            **episode_infos,
        }
        return episode_infos

    def collect_rollout(self, phase, log_result=True):
        obs, state = self.env.reset()
        stopped = np.zeros((self.num_envs, self.num_agents), dtype=np.bool_)
        episode_traj_masks = np.ones((self.num_envs, self.num_agents), dtype=np.int32) * self.buffer_length

        local_step = 0
        episode_rewards = []
        episode_actions = []
        episode_costs = []
        episode_projection_costs = []
        prev_actions = np.zeros((self.num_envs, self.num_agents, self.env.num_actions), dtype=np.float32)

        while local_step < self.buffer_length:
            local_step += 1
            active_agent_mask = (~stopped).reshape(-1)
            obs_before_step = obs.copy()
            state_before_step = state.copy()
            prev_actions_before_step = prev_actions.copy()
            
            #这里是产生原始的动作,接着做平滑处理得到动作
            raw_actions = self._sample_actions(obs, phase)
            actions = self._smooth_actions(raw_actions, prev_actions)

            obs_next, state_next, rewards, dones, info = self.env.step(actions, phase)
            executed_actions = info["safe_action"]
            prev_actions = executed_actions.copy()
            if phase == "train":
                self.total_env_steps += self.num_envs

            episode_rewards.append(rewards.copy())
            episode_actions.append(executed_actions.copy())
            episode_costs.append(info["safety_cost"].copy())
            episode_projection_costs.append(info["safety_cost_projection"].copy())

            if phase == "train":
                projection_context = self._flatten_projection_context(info["projection_context"])
                next_projection_context = self._flatten_projection_context(info["next_projection_context"])
                self.model.store_transitions(
                    obs_before_step.reshape(-1, self.env.num_observations)[active_agent_mask],
                    state_before_step.reshape(-1, self.env.num_states)[active_agent_mask],
                    executed_actions.reshape(-1, self.env.num_actions)[active_agent_mask],
                    prev_actions_before_step.reshape(-1, self.env.num_actions)[active_agent_mask],
                    rewards.reshape(-1, 1)[active_agent_mask],
                    info["safety_cost"].reshape(-1, 1)[active_agent_mask],
                    obs_next.reshape(-1, self.env.num_observations)[active_agent_mask],
                    state_next.reshape(-1, self.env.num_states)[active_agent_mask],
                    dones.reshape(-1, 1).astype(np.float32)[active_agent_mask],
                    self._filter_projection_context(projection_context, active_agent_mask),
                    self._filter_projection_context(next_projection_context, active_agent_mask),
                    self._flat_agent_ids()[active_agent_mask],
                )
                self._maybe_update_model()

            for env_id in range(self.num_envs):
                for agent_id in range(self.num_agents):
                    if dones[env_id, agent_id, 0] and not stopped[env_id, agent_id]:
                        episode_traj_masks[env_id, agent_id] = local_step
                        stopped[env_id, agent_id] = True

            obs, state = obs_next, state_next

            if dones.all():
                break

        episode_infos = self._summarize_episode(
            episode_rewards,
            episode_actions,
            episode_costs,
            episode_projection_costs,
            episode_traj_masks,
        )

        if phase == "train":
            self.train_count += 1
            if log_result:
                self.log_env(episode_infos, self.train_count)
        elif phase == "eval":
            if log_result:
                self.eval_count += 1
                self.log_env(episode_infos, self.eval_count, suffix="eval_")
        return episode_infos

    @torch.no_grad()
    def eval(self):
        torch.save(self.model.state_dict(), self.save_dir + f"/version_{self.train_count}.pt")
        print(f"\033[0;31maccess the latest model - version {self.train_count}\033[0m")

        eval_infos = []
        rng_state = np.random.get_state()
        random_state = random.getstate()
        for eval_idx in range(self.args.num_eval_episodes):
            np.random.seed(self.args.eval_seed + eval_idx)
            random.seed(self.args.eval_seed + eval_idx)
            eval_infos.append(self.collect_rollout(phase="eval", log_result=False))
        np.random.set_state(rng_state)
        random.setstate(random_state)

        mean_eval_info = {}
        for key in eval_infos[0]:
            values = [float(info[key]) for info in eval_infos]
            mean_eval_info[key] = np.array(float(np.mean(values)))
        self.eval_count += 1
        self.log_env(mean_eval_info, self.eval_count, suffix="eval_")

    def log_env(self, env_info, count=None, suffix=""):
        episode_step = self.train_count if suffix == "eval_" else count
        data_env = [self.total_env_steps, episode_step, self.total_train_steps]
        for k, v in env_info.items():
            data_env.append(v)
            suffix_k = k if suffix is None else suffix + k
            print(suffix_k + " is " + str(v))
            if "episode_r" in suffix_k:
                tag = f"{suffix}episode_r/{suffix_k}"
            elif "action" in suffix_k:
                tag = f"{suffix}action_stats/{suffix_k}"
            else:
                tag = suffix_k
            self.tb_writer.add_scalar(tag, v, self.total_env_steps)
            self.tb_writer.add_scalar(f"by_episode/{tag}", v, episode_step)
            value_array = np.asarray(v)
            if suffix == "" and self.train_reward_ema_beta > 0.0 and value_array.size == 1:
                value = float(value_array.reshape(-1)[0])
                prev = self.train_ema_infos.get(k, value)
                ema = self.train_reward_ema_beta * prev + (1.0 - self.train_reward_ema_beta) * value
                self.train_ema_infos[k] = ema
                self.tb_writer.add_scalar(f"train_ema/{tag}", ema, self.total_env_steps)
        self.tb_writer.flush()

        if suffix == "eval_":
            progress_filename = os.path.join(self.run_dir, "progress_eval.csv")
        else:
            progress_filename = os.path.join(self.run_dir, "progress.csv")

        pd.DataFrame([data_env]).to_csv(progress_filename, mode="a", header=False, index=False)

    def log_train_update(self, info):
        print(
            "\033[33m"
            + "total_loss:{:.6f}, actor_loss:{:.6f}, q1_loss:{:.6f}, q2_loss:{:.6f}, "
              "cost_q1_loss:{:.6f}, cost_q2_loss:{:.6f}, alpha:{:.6f}, lambda:{:.6f}, "
              "Jc:{:.6f}".format(
                info["total_loss"],
                info["actor_loss"],
                info["q1_loss"],
                info["q2_loss"],
                info["cost_q1_loss"],
                info["cost_q2_loss"],
                info["alpha"],
                info["lambda"],
                info["cost_estimate"],
            )
            + "\033[0m"
        )

        for i in range(self.num_agents):
            self.tb_train_writers[i].add_scalar("loss/total", info["total_loss"], self.total_train_steps)
            self.tb_train_writers[i].add_scalar("loss/actor", info["actor_loss"], self.total_train_steps)
            self.tb_train_writers[i].add_scalar("loss/q1", info["q1_loss"], self.total_train_steps)
            self.tb_train_writers[i].add_scalar("loss/q2", info["q2_loss"], self.total_train_steps)
            self.tb_train_writers[i].add_scalar(
                "loss/cost_q1", info["cost_q1_loss"], self.total_train_steps
            )
            self.tb_train_writers[i].add_scalar(
                "loss/cost_q2", info["cost_q2_loss"], self.total_train_steps
            )
            self.tb_train_writers[i].add_scalar("alpha/value", info["alpha"], self.total_train_steps)
            self.tb_train_writers[i].add_scalar("lambda/value", info["lambda"], self.total_train_steps)
            self.tb_train_writers[i].add_scalar(
                "constraint/cost_estimate", info["cost_estimate"], self.total_train_steps
            )
            self.tb_train_writers[i].add_scalar(
                "constraint/cost_gap", info["cost_gap"], self.total_train_steps
            )
            self.tb_train_writers[i].flush()

            info_record = [
                self.total_train_steps,
                info["total_loss"],
                info["actor_loss"],
                info["q1_loss"],
                info["q2_loss"],
                info["alpha"],
                info["cost_q1_loss"],
                info["cost_q2_loss"],
                info["lambda"],
                info["cost_estimate"],
                info["cost_gap"],
            ]
            pd.DataFrame([info_record]).to_csv(
                os.path.join(self.log_dir, f"../progress_train_{i}.csv"),
                mode="a",
                header=False,
                index=False,
            )
