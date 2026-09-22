import random
import importlib

import numpy as np
from gym.spaces import Discrete, Box
import utilities
import math
import os
import sys

sys.path.append("/usr/local/webots/lib/controller/python")
os.environ["WEBOTS_HOME"] = "/usr/local/webots"

try:
    CSVSupervisorEnv = importlib.import_module(
        "deepbots.supervisor.controllers.csv_supervisor_env"
    ).CSVSupervisorEnv
except ModuleNotFoundError:
    _SupervisorCSV = importlib.import_module(
        "deepbots.supervisor.controllers.supervisor_emitter_receiver"
    ).SupervisorCSV

    class CSVSupervisorEnv(_SupervisorCSV):
        def __init__(self, timestep=None, emitter_name="emitter", receiver_name="receiver"):
            super().__init__(emitter_name=emitter_name, receiver_name=receiver_name, time_step=timestep)

        def initialize_comms(self, emitter_name, receiver_name):
            emitter = self.supervisor.getDevice(emitter_name)
            receiver = self.supervisor.getDevice(receiver_name)

            if emitter is None:
                emitter = self.supervisor.getEmitter(emitter_name)
            if receiver is None:
                receiver = self.supervisor.getReceiver(receiver_name)

            if receiver is not None:
                receiver.enable(self.timestep)
            self.emitter = emitter
            self.receiver = receiver
            return emitter, receiver

        def __getattr__(self, name):
            if hasattr(self.supervisor, name):
                return getattr(self.supervisor, name)
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

from scipy.spatial.transform import Rotation as R
from scipy import interpolate
# from sophuspy import SO3
# import gtsam
import time

from atacom.navigation_manifold import ConstraintManifoldProjector

class Epuck2Supervisor(CSVSupervisorEnv):
    def _wb_supervisor(self):
        return getattr(self, "supervisor", self)

    def _wb_supervisor_step(self, timestep):
        return self._wb_supervisor().step(timestep)

    def __init__(self,all_args=None):
        super().__init__(timestep=all_args.timestep // all_args.interval)
        self.args = all_args
        self.num_agents = self.args.num_agents
        self.num_obs_targets = self.args.num_obs_targets
        self.num_obs_agents = self.args.num_obs_agents
        self.obs_view = self.args.obs_view
        self.comm_view = self.args.comm_view
        self.rab_view = self.args.rab_view
        self.catch_distance = self.args.catch_distance
        self.buffer_length = self.args.buffer_length
        self.start_flag = True
        self.start_time = 0
        self.collision_distance = self.args.collision_distance
        self.collision_reward = self.args.collision_reward
        self.agent_avoid_margin = self.args.manifold_agent_avoid_margin
        self.capture_action_conditions = self.args.capture_action_conditions[0]
        self.reward_progress_weight = self.args.reward_progress_weight
        self.reward_reach_weight = self.args.reward_reach_weight
        self.reward_finish_weight = self.args.reward_finish_weight
        self.reward_collision_weight = self.args.reward_collision_weight
        self.reward_time_penalty = self.args.reward_time_penalty
        self.reward_progress_scale = max(self.args.reward_progress_scale, 1e-6)

        # Constrained-SAC safety-cost parameters.  The three components are
        # normalized to [0, 1] before being combined into one per-agent cost.
        self.safety_cost_collision_weight = float(
            getattr(self.args, "safety_cost_collision_weight", 1.0)
        )
        self.safety_cost_wall_weight = float(
            getattr(self.args, "safety_cost_wall_weight", 0.5)
        )
        self.safety_cost_agent_weight = float(
            getattr(self.args, "safety_cost_agent_weight", 0.5)
        )
        self.safety_cost_max = max(
            float(getattr(self.args, "safety_cost_max", 1.0)), 1e-6
        )
        configured_wall_margin = getattr(self.args, "safety_cost_wall_margin", None)
        if configured_wall_margin is None:
            configured_wall_margin = self.args.manifold_wall_turn_margin
        configured_agent_margin = getattr(self.args, "safety_cost_agent_margin", None)
        if configured_agent_margin is None:
            configured_agent_margin = self.args.manifold_agent_avoid_margin
        self.safety_cost_wall_margin = max(
            float(configured_wall_margin), self.collision_distance + 1e-6
        )
        self.safety_cost_agent_margin = max(
            float(configured_agent_margin), self.collision_distance + 1e-6
        )

        self.agent_pose_dim = 3
        self.target_hint_start = self.agent_pose_dim
        self.target_hint_dim = 3

        self.agent_loc_dim = self.agent_pose_dim + self.target_hint_dim
        configured_agent_loc_dim = getattr(self.args, "agent_loc_dim", self.agent_loc_dim)
        if configured_agent_loc_dim != self.agent_loc_dim:
            raise ValueError(
                f"agent_loc_dim is fixed to {self.agent_loc_dim} for the reduced actor "
                f"observation layout, got {configured_agent_loc_dim}"
            )

        self.mutual_dim = self.args.mutual_dim
        self.agent_mutual_dim = self.mutual_dim
        self.num_target = self.args.num_target
        self.num_observations = (
            self.agent_loc_dim
            + self.agent_mutual_dim * self.num_obs_agents
            + self.mutual_dim * self.num_obs_targets
        )
        self.num_actions = 2  # Continuous action: [left_wheel_ratio, right_wheel_ratio]
        
        self.world_size = [float(self.args.world_size_x), float(self.args.world_size_y)]
        if self.world_size[0] <= 0.0 or self.world_size[1] <= 0.0:
            raise ValueError(f"world_size must be positive, got {self.world_size}")

        self.num_states_agent = 3  # 2 for position, 1 for angle
        self.num_states_target = 2  # 2 for position
        self.num_states = (
            self.num_states_agent * self.num_agents
            + self.num_states_target * self.num_target
        )
        # Centralized critic state: all agent poses + all target positions.
        # self.num_observations * self.num_agents
        self.num_envs = self.args.n_rollout_threads
        self.timestep = self.args.timestep
        self.interval = self.args.interval

        # self.emitter, self.emitter_target, self.receiver = self.initialize_comms(
        #     "emitter", "emitter_target", "receiver")

        self.emitter_target = self.getDevice("emitter_target")

        self.arena_pos = self.add_arenas()
        # import pdb;pdb.set_trace()
        collision_wall_pos = self.arena_pos[:self.num_envs, np.newaxis, :].repeat(self.num_agents, axis=1)
        self.collision_wall_cond = np.zeros((self.num_envs, self.num_agents, 4), dtype=np.float32)
        self.collision_wall_cond[..., 0] = collision_wall_pos[..., 0] - self.world_size[0]/2 + self.collision_distance
        self.collision_wall_cond[..., 1] = collision_wall_pos[..., 0] + self.world_size[0]/2 - self.collision_distance
        self.collision_wall_cond[..., 2] = collision_wall_pos[..., 1] - self.world_size[1]/2 + self.collision_distance
        self.collision_wall_cond[..., 3] = collision_wall_pos[..., 1] + self.world_size[1]/2 - self.collision_distance

        self.target_pos = self.add_agents(self.arena_pos)
        self.target_markers = []

        self.action_buffer = np.zeros((self.num_envs, self.num_agents, 2), dtype=np.int32)
        # action_buffer[:, :, 0] 存储上一次的动作
        # action_buffer[:, :, 1] 存储连续相同动作的计数

        # self.target_pos = self.env_target_pos()
        print("========== supervisor info ==========")
        print(f"num_agents: {self.num_agents}")
        print(f"num_targets: {self.num_target}")


        self.robot = []
        for i in range(1,self.num_envs+1):
            robot_env = []
            for j in range(1,self.num_agents+1):
                robot_env.append(self.getFromDef(f"epuck{i}-{j}"))
            self.robot.append(robot_env)

        self.target = []
        for i in range(1, self.num_envs+1):
            target_env = []
            for j in range(1,self.num_target+1):
                target_env.append(self.getFromDef(f"target{i}-{j}"))
            self.target.append(target_env)

        self.signal_strength = self.args.signal_strength
        self.max_signal_strength = self.args.max_signal_strength
        self.ps_sensor_mm = {'min': 50, 'max': self.max_signal_strength}
        self.angle_mm = {'min': -np.pi, 'max': np.pi}
        self.dis_mm = {'min': 0, 'max': float(np.hypot(self.world_size[0], self.world_size[1]))}
        self.pos_x_mm = {'min': -self.world_size[0] / 2.0, 'max': self.world_size[0] / 2.0}
        self.pos_y_mm = {'min': -self.world_size[1] / 2.0, 'max': self.world_size[1] / 2.0}
        self.max_speed = 6.28
        self.steps = np.zeros((self.num_envs), dtype=np.float32)

        self.radius_epuck = 0.035
        self.use_constraint_manifold = self.args.use_constraint_manifold
        safety_margin = self.args.manifold_safety_margin
        if safety_margin < 0:
            safety_margin = self.collision_distance
        dt = max(self.timestep / 1000.0, 1e-6)
        self.manifold_projector = ConstraintManifoldProjector(
            world_size=self.world_size,
            safety_margin=safety_margin,
            dt=dt,
            wheel_radius=self.args.manifold_wheel_radius,
            axle_length=self.args.manifold_axle_length,
            max_wheel_speed=self.max_speed,
            k_c=self.args.manifold_kc,
            heading_gain=self.args.manifold_heading_gain,
            wall_turn_margin=self.args.manifold_wall_turn_margin,
            agent_avoid_margin=self.args.manifold_agent_avoid_margin,
        )

        self._wb_supervisor_step(self.timestep // self.interval)
        
        self.cleanup()

    # def env_target_pos(self):
    #     target_pos = np.array([
    #         [-0.75, 0.70],
    #         [-0.25, 0.70],
    #         [0.25, 0.70],
    #         [0.75, 0.70],
    #         [-0.50, 0.30],
    #         [0.50, 0.30],
    #     ])

    #     env_tar_pos = target_pos[np.newaxis,:].repeat(self.num_envs,0) +  self.arena_pos[:self.num_envs,:-1][:,np.newaxis,:]

    #     return env_tar_pos

    def _axis_candidates(self, low, high, spacing):
        if high < low:
            raise ValueError(f"invalid spawn bounds: low={low}, high={high}")
        spacing = max(float(spacing), 1e-6)
        if np.isclose(low, high):
            return np.asarray([low], dtype=np.float32)

        count = max(int(math.floor((high - low) / spacing)) + 1, 2)
        return np.linspace(low, high, count, dtype=np.float32)

    def _region_candidates(self, arena_pos, x_bounds, y_bounds, spacing):
        xs = self._axis_candidates(x_bounds[0], x_bounds[1], spacing)
        ys = self._axis_candidates(y_bounds[0], y_bounds[1], spacing)
        candidates = np.asarray([[x, y] for y in ys for x in xs], dtype=np.float32)
        candidates[:, 0] += float(arena_pos[0])
        candidates[:, 1] += float(arena_pos[1])
        return candidates

    def _sample_spaced_points(
        self,
        candidates,
        count,
        min_distance,
        avoid_points=None,
        avoid_distance=0.0,
        max_tries=500,
    ):
        candidates = np.asarray(candidates, dtype=np.float32).reshape(-1, 2)
        count = int(count)
        if count <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if candidates.shape[0] < count:
            raise ValueError(f"not enough spawn candidates: need {count}, have {candidates.shape[0]}")

        min_distance = max(float(min_distance), 0.0)
        avoid_distance = max(float(avoid_distance), 0.0)
        max_tries = max(int(max_tries), 1)
        if avoid_points is None:
            avoid_points = np.zeros((0, 2), dtype=np.float32)
        avoid_points = np.asarray(avoid_points, dtype=np.float32).reshape(-1, 2)

        best_selected = []
        for _ in range(max_tries):
            selected = []
            for idx in np.random.permutation(candidates.shape[0]):
                point = candidates[idx]
                if avoid_points.size > 0:
                    avoid_dis = np.linalg.norm(avoid_points - point, axis=1)
                    if np.any(avoid_dis < avoid_distance):
                        continue
                if selected:
                    selected_dis = np.linalg.norm(np.asarray(selected, dtype=np.float32) - point, axis=1)
                    if np.any(selected_dis < min_distance):
                        continue

                selected.append(point)
                if len(selected) == count:
                    return np.asarray(selected, dtype=np.float32)

            if len(selected) > len(best_selected):
                best_selected = selected

        raise RuntimeError(
            "failed to sample reset positions with distance constraints: "
            f"need={count}, best={len(best_selected)}, candidates={candidates.shape[0]}, "
            f"min_distance={min_distance}, avoid_distance={avoid_distance}"
        )

    def random_pos(self, arena_pos):
        arena_pos = np.asarray(arena_pos, dtype=np.float32)
        grid_size = max(float(self.args.spawn_grid_size), 1e-6)
        wall_margin = max(float(self.args.spawn_wall_margin), float(self.collision_distance))
        split_margin = max(float(self.args.spawn_split_margin), 0.0)

        half_w = self.world_size[0] / 2.0
        half_h = self.world_size[1] / 2.0
        x_bounds = (-half_w + wall_margin, half_w - wall_margin)
        target_y_bounds = (split_margin, half_h - wall_margin)
        agent_y_bounds = (-half_h + wall_margin, -split_margin)

        if x_bounds[0] > x_bounds[1]:
            raise ValueError(f"spawn_wall_margin={wall_margin} leaves no x spawn space")
        if target_y_bounds[0] > target_y_bounds[1]:
            raise ValueError(f"spawn margins leave no target spawn space: {target_y_bounds}")
        if agent_y_bounds[0] > agent_y_bounds[1]:
            raise ValueError(f"spawn margins leave no agent spawn space: {agent_y_bounds}")

        agent_candidates = self._region_candidates(arena_pos, x_bounds, agent_y_bounds, grid_size)
        target_candidates = self._region_candidates(arena_pos, x_bounds, target_y_bounds, grid_size)
        max_tries = self.args.spawn_sample_max_tries

        agent_pos = self._sample_spaced_points(
            agent_candidates,
            self.num_agents,
            self.args.min_agent_distance,
            max_tries=max_tries,
        )
        target_pos = self._sample_spaced_points(
            target_candidates,
            self.num_target,
            self.args.min_target_distance,
            avoid_points=agent_pos,
            avoid_distance=self.args.min_agent_target_distance,
            max_tries=max_tries,
        )

        return agent_pos, target_pos

    def add_agents(self, arena_pos):
        target_env_pos = []
        for i in range(1, self.num_envs+1):
            agent_pos, target_pos = self.random_pos(arena_pos[i-1])
            target_env_pos.append(target_pos)
            for j in range(1, self.num_agents + 1):
                self.importRobot(i, j, agent_pos[j - 1, 0], agent_pos[j - 1, 1], 0.05, random.uniform(-np.pi, np.pi))
        return np.array(target_env_pos)
    
    def add_arenas(self):
        column = int(math.ceil(math.sqrt(self.num_envs)))
        wall_len = 0.1
        wall_high = 0.1
        x_arena = self.world_size[0] * column + (column-1)*wall_len
        y_arena = self.world_size[1] * column + (column-1)*wall_len
        self.importArena(x_arena ,y_arena)
        x_begin = -x_arena / 2
        y_begin = -y_arena / 2
        x_tmp = 0
        y_tmp = 0
        for i in range(column-1):
            if i == 0:
                x_tmp = x_begin + self.world_size[0] + wall_len/2
            else:
                x_tmp += (self.world_size[0] + wall_len)
            self.importWall(x_tmp,0,wall_high/2,wall_len,y_arena,wall_high)

        for i in range(column-1):
            if i == 0:
                y_tmp = y_begin + self.world_size[1] + wall_len/2
            else:
                y_tmp += (self.world_size[1] + wall_len)


            for j in range(column):
                if j == 0:
                    x_tmp = x_begin + self.world_size[0]/2
                else:
                    x_tmp += (self.world_size[0] + wall_len)
                self.importWall(x_tmp,y_tmp,wall_high/2,self.world_size[0],wall_len,wall_high)

            # self.importWall(0, y_tmp, wall_high / 2, x_arena, wall_len, wall_high)

        arena_pos = np.zeros((column, column, 3), dtype=np.float32)
        x = 0
        y = 0
        for i in range(column):
            if i==0:
                x = x_begin+self.world_size[0]/2
            else:
                x += (self.world_size[0]+wall_len)
            arena_pos[:,i,0] = x

        for i in range(column):
            if i==0:
                y = y_begin+self.world_size[1]/2
            else:
                y += (self.world_size[1]+wall_len)
            arena_pos[i,:,1] = y
        arena_pos_re = arena_pos.reshape(-1,3)
        # for i in range(self.num_envs):
        #     self.importArena(i+1,arena_pos_re[i,0],arena_pos_re[i,1],arena_pos_re[i,2],self.world_size[0],self.world_size[1])
        return arena_pos_re

    def importArena(self, len_x, len_y):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        RectangleArena{
                            translation 0 0 0
                            name "rectangle arena"
                            floorSize %f %f        
                        }
                        """ % (len_x, len_y)
        chFd.importMFNodeFromString(-1, line_String)
    
    def importWall(self,x, y, z, len_x, len_y, len_z):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        Solid {
                          translation %f %f %f
                          children [
                            Shape {
                              appearance Appearance {
                                material Material {
                                  diffuseColor 0.8 0.8 0.8
                                }
                              }
                              geometry Box {
                                size %f %f %f
                              }
                            }
                          ]
                          boundingObject Box {
                            size %f %f %f
                          }
                          physics Physics {
                            density 1000  # 使用默认密度
                          }
                        }
                        """ % (x, y, z, len_x, len_y, len_z, len_x, len_y, len_z)
        # line_String = """
        #                 Transform {
        #                   translation %f %f %f
        #                   children [
        #                     Shape {
        #                       appearance Appearance {
        #                         material Material {
        #                           diffuseColor 0.8 0.8 0.8
        #                         }
        #                       }
        #                       geometry Box {
        #                         size %f %f %f
        #                       }
        #                     }
        #                   ]
        #                 }
        #                         """ % (x, y, z, len_x, len_y, len_z)
        chFd.importMFNodeFromString(-1, line_String)

    def importRobot(self, arena_id, id, x, y, z, ro):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        DEF epuck%d-%d E-puck{
                            translation %f %f %f
                            rotation 0 0 1 %f
                            name "e-puck%d-%d"
                            controller "epuck_controller"
                            supervisor FALSE
                            version "2"
                            emitter_channel %d
                            receiver_channel %d
                            receiver_rab_channel %d           
                        }
                        """ % (arena_id, id, x, y, z, ro, arena_id, id, (arena_id-1)*(self.num_agents+3)+1, (arena_id-1)*(self.num_agents+3)+2, (arena_id-1)*(self.num_agents+3)+id+3)
        chFd.importMFNodeFromString(-1, line_String)

    def importTarget(self, arena_id, id, x, y, z, ro):
        root = self.getRoot()
        chFd = root.getField("children")
        line_String = """
                        DEF target%d-%d E-puck{
                            translation %f %f %f
                            rotation 0 0 1 %f
                            name "target%d-%d"
                            controller "target_controller"
                            supervisor FALSE
                            version "1"
                            emitter_channel 3
                            receiver_channel %d
                        }
                        """ % (arena_id, id, x, y, z, ro, arena_id, id, (arena_id-1)*(self.num_agents+3)+3)


        chFd.importMFNodeFromString(-1, line_String)

    def initialize_emitter(self, id):
        emitter = self.getDevice('emitter0'+str(id))
        return emitter

    def Interpolate(self, Range, Values):
        Points, Values = zip(*Values)
        f = interpolate.interp1d(Points, Values, fill_value="extrapolate")
        return f(Range*100)

    def handle_emitter(self, action, agent_alive, target_alive, phase):
        assert self.num_envs == 1
        for i in range(self.num_envs):
            action_flat = np.asarray(action[i], dtype=np.float32).reshape(-1).tolist()
            message = ",".join(f"{v:.6f}" for v in action_flat)
            message += "," + (",".join(map(str, agent_alive[i])))
            message += "," + str(phase)
            self.emitter.send(message.encode("utf-8"))

            target_message = (",".join(map(str, target_alive[i])))
            self.emitter_target.send(target_message.encode("utf-8"))

    def handle_receiver(self):
        if self.receiver is None:
            return
        message = np.zeros((self.num_envs, self.num_agents, 10), dtype=np.float32)

        for i in range(self.num_envs):
            for j in range(self.num_agents):
                if self.receiver.getQueueLength() > 0:
                    try:
                        packet = self.receiver.getString()
                    except AttributeError:
                        packet = self.receiver.getData()

                    if isinstance(packet, (bytes, bytearray)):
                        packet = packet.decode("utf-8", errors="ignore")
                    else:
                        packet = str(packet)

                    self.receiver.nextPacket()

                    string_message = packet.split(",")
                    if len(string_message) < 2:
                        continue

                    robot_token = string_message[0].strip()
                    if not robot_token.startswith("a"):
                        continue

                    try:
                        idx = int(robot_token[1:]) - 1
                    except ValueError:
                        continue

                    if idx < 0 or idx >= self.num_agents:
                        continue

                    try:
                        payload = np.asarray(string_message[1:], dtype=np.float32)
                    except ValueError:
                        continue

                    length = min(payload.shape[0], message.shape[-1])
                    if length > 0:
                        message[i, idx, :length] = payload[:length]

        self.message[:] = message[...,-8:]  # only the ps values

    def draw_target_markers(self):
        for env_idx in range(self.num_envs):
            for target_idx in range(self.num_target):
                # 获取目标位置
                target_x, target_y = self.target_pos[env_idx, target_idx]
                target_z = 0.01  # 设置标记的高度（略高于地面）

                self.import_marker(
                    name=f'target_marker_{env_idx}_{target_idx}',
                    x=target_x,
                    y=target_y,
                    z=target_z,
                    color=(1, 0, 0)
                )
    
    def import_marker(self, name, x, y, z, color=(1, 0, 0)):
        root = self.getRoot()
        chFd = root.getField("children")
        line_string = f"""
            DEF {name} Solid {{
                translation {x} {y} {z}
                children [
                    Shape {{
                        appearance Appearance {{
                            material Material {{
                                diffuseColor {color[0]} {color[1]} {color[2]}
                            }}
                        }}
                        geometry Box {{
                            size 0.01 0.01 0.01
                        }}
                    }}
                ]
            }}
        """
        chFd.importMFNodeFromString(-1, line_string)

    def get_action_mask(self, last_actions):
        del last_actions
        return np.ones((self.num_envs, self.num_agents, self.num_actions), dtype=np.float32)

    def _get_agent_headings(self):
        headings = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for env_idx in range(self.num_envs):
            for agent_idx in range(self.num_agents):
                rotation = self.robot[env_idx][agent_idx].getField("rotation").getSFRotation()
                headings[env_idx, agent_idx] = rotation[3] if rotation[2] > 0 else -rotation[3]
        return headings

    def _refresh_agent_positions(self):
        for env_idx in range(self.num_envs):
            for agent_idx in range(self.num_agents):
                self.agent_pos[env_idx, agent_idx] = self.robot[env_idx][agent_idx].getField("translation").getSFVec3f()

    def _get_projection_context(self):
        self._refresh_agent_positions()
        return {
            "positions": self.agent_pos[..., :2].copy(),
            "headings": self._get_agent_headings().copy(),
            "alive_mask": self.alive_agent_buf.copy(),
            "arena_centers": self.arena_pos[:self.num_envs, :2].copy(),
        }

    def _project_actions_to_constraint_manifold(self, action, projection_context=None):
        action = np.asarray(action, dtype=np.float32).reshape(self.num_envs, self.num_agents, self.num_actions)
        action = np.clip(action, -1.0, 1.0)
        if not self.use_constraint_manifold:
            return action

        if projection_context is None:
            projection_context = self._get_projection_context()
        return self.manifold_projector.project(
            action,
            projection_context["positions"],
            projection_context["headings"],
            projection_context["alive_mask"],
            projection_context["arena_centers"],
        )

    def step(self,action, phase):
        action = np.asarray(action, dtype=np.float32).reshape(self.num_envs, self.num_agents, self.num_actions)
        projection_context = self._get_projection_context()

        #得到的原始动作增加约束流行为安全动作，保证在仿真中不会发生碰撞等.
        safe_action = self._project_actions_to_constraint_manifold(action, projection_context)

        self.handle_emitter(
            safe_action,
            self.alive_agent_buf.astype(int).reshape(self.num_envs, -1).tolist(), 
            self.alive_target_buf.astype(int).reshape(self.num_envs, -1).tolist(),
            phase
        )
        import time; time.sleep(1e-6)
        if self._wb_supervisor_step(self.timestep) == -1:
            exit()
        self.handle_receiver()
        obs_next = self.get_observations()
        state_next = self.get_state()

        # Capture the post-step geometry before get_reward() updates the alive
        # masks for captured agents.  The cost belongs to this transition and
        # must use the alive mask from its beginning.
        next_cost_context = self._get_projection_context()
        safety_cost = self.get_safety_cost(
            projection_context,
            next_cost_context,
            action,
            safe_action,
        )
        rewards = self.get_reward(safe_action)
        dones = self.is_done()
        next_projection_context = self._get_projection_context()
        info = self.get_info() or {}
        info.update({
            "raw_action": action.copy(),
            "safe_action": safe_action.copy(),
            "projection_delta": safe_action - action,
            "projection_context": projection_context,
            "next_projection_context": next_projection_context,
            "safety_cost": safety_cost["total"].copy(),
            "safety_cost_collision": safety_cost["collision"].copy(),
            "safety_cost_wall": safety_cost["wall"].copy(),
            "safety_cost_agent": safety_cost["agent"].copy(),
            "safety_cost_projection": safety_cost["projection"].copy(),
        })
        return (
            obs_next,
            state_next,
            rewards,
            dones,
            info,
        )

    def _collision_hits(self):
        """Return the same sensor collision event used by the task reward."""
        exist_obstacle = np.zeros(
            (self.num_envs, self.num_agents), dtype=np.bool_
        )
        for sensor_idx in range(8):
            exist_obstacle |= self.message[..., sensor_idx] > self.max_signal_strength
        return np.where(self.alive_agent_buf, exist_obstacle, False)

    def get_safety_cost(self, projection_context, next_context, raw_action, safe_action):
        """Compute bounded per-agent safety costs for one transition.

        Wall and inter-agent costs are dense proximity costs.  Collision cost is
        the binary proximity-sensor event already used by the environment
        reward.  The optional projection term is returned for diagnostics and
        is deliberately excluded from the constrained cost itself.
        """
        alive = np.asarray(projection_context["alive_mask"], dtype=np.bool_)
        next_positions = np.asarray(next_context["positions"], dtype=np.float32)[..., :2]
        centers = np.asarray(next_context["arena_centers"], dtype=np.float32)[..., :2]

        half_w = self.world_size[0] * 0.5
        half_h = self.world_size[1] * 0.5
        lower_x = centers[..., 0] - half_w
        upper_x = centers[..., 0] + half_w
        lower_y = centers[..., 1] - half_h
        upper_y = centers[..., 1] + half_h
        wall_distance = np.minimum.reduce((
            next_positions[..., 0] - lower_x[:, None],
            upper_x[:, None] - next_positions[..., 0],
            next_positions[..., 1] - lower_y[:, None],
            upper_y[:, None] - next_positions[..., 1],
        ))
        hard_wall = self.collision_distance
        wall_denominator = max(self.safety_cost_wall_margin - hard_wall, 1e-6)
        wall_cost = np.clip(
            (self.safety_cost_wall_margin - wall_distance) / wall_denominator,
            0.0,
            1.0,
        )

        agent_distance = np.full(
            (self.num_envs, self.num_agents), np.inf, dtype=np.float32
        )
        for env_idx in range(self.num_envs):
            for agent_idx in range(self.num_agents):
                other_mask = alive[env_idx].copy()
                other_mask[agent_idx] = False
                other_positions = next_positions[env_idx, other_mask]
                if other_positions.size:
                    agent_distance[env_idx, agent_idx] = np.linalg.norm(
                        other_positions - next_positions[env_idx, agent_idx], axis=-1
                    ).min()
        hard_agent = max(
            float(getattr(self.manifold_projector, "safety_margin", hard_wall)),
            1e-6,
        )
        agent_denominator = max(self.safety_cost_agent_margin - hard_agent, 1e-6)
        agent_cost = np.clip(
            (self.safety_cost_agent_margin - agent_distance) / agent_denominator,
            0.0,
            1.0,
        )
        agent_cost[~np.isfinite(agent_distance)] = 0.0

        collision_cost = self._collision_hits().astype(np.float32)
        collision_cost = np.where(alive, collision_cost, 0.0)
        total = np.clip(
            self.safety_cost_collision_weight * collision_cost
            + self.safety_cost_wall_weight * wall_cost
            + self.safety_cost_agent_weight * agent_cost,
            0.0,
            self.safety_cost_max,
        )
        total = np.where(alive, total, 0.0).astype(np.float32)

        projection_delta = np.linalg.norm(
            np.asarray(safe_action, dtype=np.float32)
            - np.asarray(raw_action, dtype=np.float32),
            axis=-1,
        )
        projection_cost = np.clip(projection_delta / 2.0, 0.0, 1.0)
        projection_cost = np.where(alive, projection_cost, 0.0).astype(np.float32)

        return {
            "total": total,
            "collision": collision_cost,
            "wall": np.where(alive, wall_cost, 0.0).astype(np.float32),
            "agent": np.where(alive, agent_cost, 0.0).astype(np.float32),
            "projection": projection_cost,
        }
    
    def cleanup(self):
        """Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.states_buf = np.zeros((self.num_envs, self.num_agents, self.num_states), dtype=np.float32)
        self.alive_target_buf = np.ones((self.num_envs, self.num_target), dtype=np.bool_)
        self.alive_agent_buf = np.ones((self.num_envs, self.num_agents), dtype=np.bool_)
        self.agent_dis = np.zeros((self.num_envs, self.num_agents, self.num_agents), dtype=np.float32)
        self.distance = np.zeros((self.num_envs, self.num_agents, self.num_target), dtype=np.float32)
        self.min_target_dis = np.full((self.num_envs, self.num_agents, self.num_target), np.inf, dtype=np.float32)
        self.agent_pos = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32)
        self.message = np.zeros((self.num_envs, self.num_agents, 8), dtype=np.float32)  # only ps values
        self.early_stop = np.zeros((self.num_envs, 1),dtype=np.bool_)
        self.num_collision = np.zeros((self.num_envs, 1), dtype=np.int32)
        self.agent_angle = np.zeros((self.num_envs, self.num_agents, 9), dtype=np.float32)
        self.agent_vec = np.zeros((self.num_envs, self.num_agents, 3), dtype=np.float32)
        self.obs_target_sequence = np.zeros((self.num_envs, self.num_agents, self.num_target, 2), dtype=np.float32)
        self.action_buffer = np.zeros((self.num_envs, self.num_agents, 2), dtype=np.int32)

    def reset_positions(self):
        root = self.getRoot()
        chFd = root.getField("children")
        for i in range(self.num_envs):
            agent_pos, target_pos = self.random_pos(self.arena_pos[i])
            for j in range(self.num_agents):
                epuck_default_pos = self.robot[i][j].getField('translation').getSFVec3f()
                epuck_default_pos[:2] = agent_pos[j][:2]
                robot_default_rotation = [0,0,1,0]
                robot_default_rotation[3] = random.uniform(-np.pi, np.pi)
                self.robot[i][j].getField('translation').setSFVec3f(epuck_default_pos)
                self.robot[i][j].getField('rotation').setSFRotation(robot_default_rotation)

            self.target_pos[i] = target_pos
            for k in range(self.num_target):
                marker_node = self.getFromDef(f'target_marker_{i}_{k}')
                if marker_node is not None:
                    marker_node.remove()
        
        self.draw_target_markers()

    def get_state(self):
        return self.states_buf.copy()

    def get_availactions(self):
        return None

    def get_observations(self):
        for n in range(self.num_envs):
            for i in range(self.num_agents):
                self.agent_pos[n, i] = self.robot[n][i].getField('translation').getSFVec3f()

        observation = np.zeros((self.num_envs, self.num_agents, self.num_observations),dtype=np.float32)
        agent_obs_start = self.agent_loc_dim
        target_obs_start = self.agent_loc_dim + self.agent_mutual_dim * self.num_obs_agents
        for agent_slot in range(self.num_obs_agents):
            observation[..., agent_obs_start + self.agent_mutual_dim * agent_slot + 3] = -1.0
        for target_slot in range(self.num_obs_targets):
            observation[..., target_obs_start + self.mutual_dim * target_slot + 3] = -1.0

        for n in range(self.num_envs):
            epuck_pos = self.agent_pos[n,:,:2].copy()
            epuck_angle = np.zeros((self.num_agents, 1), dtype=np.float32)
            for i in range(self.num_agents):
                robot_rotation = self.robot[n][i].getField('rotation').getSFRotation()
                epuck_angle[i] = robot_rotation[3] if robot_rotation[2] > 0 else -robot_rotation[3]


            target_pos = np.zeros((self.num_target, 2), dtype=np.float32)
            target_pos[:] = self.target_pos[n]

            dis_to_epuck = utilities.get_distance_from_target(self.agent_pos[n,:,:2], self.agent_pos[n,:,:2])
            dis_to_target = utilities.get_distance_from_target(epuck_pos, target_pos)
            angle_to_epuck = utilities.get_angle_from_target(epuck_pos, epuck_pos, epuck_angle)
            angle_to_target = utilities.get_angle_from_target(epuck_pos, target_pos, epuck_angle)
            
            dis_to_epuck += np.random.normal(0, self.args.noise_dis, dis_to_epuck.shape)
            dis_to_target += np.random.normal(0, self.args.noise_dis, dis_to_target.shape)
            angle_to_epuck += np.random.normal(0, self.args.noise_angle, angle_to_epuck.shape)
            angle_to_target += np.random.normal(0, self.args.noise_angle, angle_to_target.shape)

            # 避免加上噪声后出现负距离
            dis_to_epuck = np.maximum(dis_to_epuck, 0)
            dis_to_target = np.maximum(dis_to_target, 0)
            # 角度归一化到[-pi, pi]
            angle_to_epuck = (angle_to_epuck + np.pi) % (2 * np.pi) - np.pi
            angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi

            dx = dis_to_target * np.cos(epuck_angle + angle_to_target)
            dy = dis_to_target * np.sin(epuck_angle + angle_to_target)
            epuck_pos_expand = epuck_pos[:, None, :]
            target_pos_measure = np.stack([
                epuck_pos_expand[:, :, 0] + dx,
                epuck_pos_expand[:, :, 1] + dy
            ], axis=-1)  # num_agent x num_target x 2
            
            self.agent_dis[n] = dis_to_epuck
            self.distance[n] = dis_to_target
            
            epuck_pos_temp = np.zeros_like(epuck_pos)
            target_pos_temp = np.zeros_like(target_pos)
            epuck_pos_temp[..., 0] = epuck_pos[..., 0] - self.arena_pos[n, 0]
            epuck_pos_temp[..., 1] = epuck_pos[..., 1] - self.arena_pos[n, 1]
            target_pos_temp[..., 0] = target_pos[..., 0] - self.arena_pos[n, 0]
            target_pos_temp[..., 1] = target_pos[..., 1] - self.arena_pos[n, 1]

            target_hint = np.zeros((self.num_agents, self.target_hint_dim), dtype=np.float32)
            alive_target_mask = self.alive_target_buf[n].astype(bool)
            if alive_target_mask.any():

                alive_target_dis = np.where(alive_target_mask[np.newaxis, :], dis_to_target, np.inf)
                nearest_target_idx = np.argmin(alive_target_dis, axis=1)
                
                nearest_target_diff = target_pos_temp[nearest_target_idx] - epuck_pos_temp
                nearest_target_dis = np.linalg.norm(nearest_target_diff, axis=1)
                target_hint[:, 0] = utilities.normalize_to_range(
                    nearest_target_diff[:, 0],
                    -self.world_size[0],
                    self.world_size[0],
                    -1,
                    1,
                    clip=True,
                )
                target_hint[:, 1] = utilities.normalize_to_range(
                    nearest_target_diff[:, 1],
                    -self.world_size[1],
                    self.world_size[1],
                    -1,
                    1,
                    clip=True,
                )
                target_hint[:, 2] = utilities.normalize_to_range(
                    nearest_target_dis,
                    self.dis_mm['min'],
                    self.dis_mm['max'],
                    0,
                    1,
                    clip=True,
                )

            epuck_angle_raw = epuck_angle.copy()
            epuck_pos[..., 0] = utilities.normalize_to_range(epuck_pos[..., 0] - self.arena_pos[n, 0], self.pos_x_mm['min'], self.pos_x_mm['max'], -1, 1,clip=True)
            epuck_pos[..., 1] = utilities.normalize_to_range(epuck_pos[..., 1] - self.arena_pos[n, 1], self.pos_y_mm['min'], self.pos_y_mm['max'], -1, 1,clip=True)
            epuck_angle = utilities.normalize_to_range(epuck_angle, self.angle_mm['min'], self.angle_mm['max'], -1, 1,clip=True)
            target_pos[..., 0] = utilities.normalize_to_range(target_pos[..., 0] - self.arena_pos[n, 0],self.pos_x_mm['min'], self.pos_x_mm['max'], -1, 1,clip=True)
            target_pos[..., 1] = utilities.normalize_to_range(target_pos[..., 1] - self.arena_pos[n, 1],self.pos_y_mm['min'], self.pos_y_mm['max'], -1, 1,clip=True)

            angle_to_epuck = utilities.normalize_to_range(angle_to_epuck, self.angle_mm['min'],self.angle_mm['max'], -1, 1,clip=True)
            angle_to_target = utilities.normalize_to_range(angle_to_target, self.angle_mm['min'], self.angle_mm['max'], -1,1,clip=True)

            observation[n, :, :2] = epuck_pos
            observation[n, :, 2] = epuck_angle[...,0]
            observation[
                n,
                :,
                self.target_hint_start: self.target_hint_start + self.target_hint_dim,
            ] = target_hint

            for i in range(self.num_agents):
                other_agent_pos_diff = np.concatenate((epuck_pos_temp[:i],epuck_pos_temp[i+1:]),axis=0) - epuck_pos_temp[i]

                other_agent_pos_diff[:, 0] = utilities.normalize_to_range(other_agent_pos_diff[:, 0], -self.obs_view ,self.obs_view, -1, 1, clip=True)
                other_agent_pos_diff[:, 1] = utilities.normalize_to_range(other_agent_pos_diff[:, 1], -self.obs_view ,self.obs_view, -1, 1, clip=True)

                other_agent_dis = np.concatenate((dis_to_epuck[i][:i],dis_to_epuck[i][i+1:]),axis=-1)
                other_agent_angle = np.concatenate((angle_to_epuck[i][:i], angle_to_epuck[i][i + 1:]), axis=-1)
                other_message = np.concatenate((
                    other_agent_pos_diff, other_agent_angle[:,np.newaxis],
                    utilities.normalize_to_range(other_agent_dis, self.dis_mm['min'],self.dis_mm['max'], 0, 1,clip=True)[:,np.newaxis],
                ),axis=-1)


                obs_agent = other_agent_dis < self.obs_view
                near_agent = np.zeros_like(obs_agent)
                near_agent_indices = np.argsort(other_agent_dis, axis=0)[:self.num_obs_agents]
                near_agent[near_agent_indices] = 1
                is_obs_agent = obs_agent & near_agent 

                num_observable_agent = is_obs_agent.sum(-1)

                if num_observable_agent > 0:
                    observation[
                        n, i,
                        agent_obs_start: agent_obs_start + self.agent_mutual_dim * num_observable_agent
                    ] = other_message[is_obs_agent].reshape(1,-1)

                target_pos_diff = target_pos_temp - epuck_pos_temp[i]
                target_pos_diff[:, 0] = utilities.normalize_to_range(target_pos_diff[:, 0], -self.obs_view, self.obs_view, -1, 1, clip=True)
                target_pos_diff[:, 1] = utilities.normalize_to_range(target_pos_diff[:, 1], -self.obs_view, self.obs_view, -1, 1, clip=True)

                target_dis = dis_to_target[i]
                t2a_angle = angle_to_target[i]
                target_message = np.concatenate((target_pos_diff, t2a_angle[:, np.newaxis],utilities.normalize_to_range(target_dis, self.dis_mm['min'],self.dis_mm['max'], 0, 1,clip=True)[:, np.newaxis]),axis=-1)

                in_sequence = (self.obs_target_sequence[n, i] != 0.0).any(-1)
                in_sequence = in_sequence & self.alive_target_buf[n]
                obs_target = (target_dis < self.obs_view) & self.alive_target_buf[n]
                # obs_target = self.alive_target_buf[n]
                is_obs_target = np.zeros_like(obs_target)
                observable_target_indices = np.where(obs_target)[0]
                if observable_target_indices.size > 0:
                    nearest_visible_indices = observable_target_indices[
                        np.argsort(target_dis[observable_target_indices])[:self.num_obs_targets]
                    ]
                    is_obs_target[nearest_visible_indices] = True
                in_sequence_not_in_obs = in_sequence & (~is_obs_target)
                num_observable_target = is_obs_target.sum(-1)
                if num_observable_target > 0:
                    observation[
                        n, i, 
                        target_obs_start: 
                        target_obs_start + self.mutual_dim * num_observable_target
                    ] = target_message[is_obs_target].reshape(1,-1)
                    
                    # update target position sequence
                    meas = target_pos_measure[i].copy()
                    meas[~is_obs_target] = 0.0
                    seq = self.obs_target_sequence[n, i]

                    has_prev, has_meas = (seq != 0.0).any(-1), (meas != 0.0).any(-1)
                    first_idx = (~has_prev) & has_meas
                    update_idx = has_prev & has_meas

                    seq[first_idx] = meas[first_idx]
                    seq[update_idx] = 0.8 * seq[update_idx] + 0.2 * meas[update_idx]
                if num_observable_target < self.num_obs_targets and in_sequence_not_in_obs.sum(-1) > 0:

                    seq_targets = self.obs_target_sequence[n, i, in_sequence_not_in_obs]
                    seq_target_pos_diff = seq_targets - epuck_pos_temp[i]
                    seq_target_dis = utilities.get_distance_from_target(epuck_pos_temp[i][np.newaxis, :], seq_targets)
                    seq_target_angle = utilities.get_angle_from_target(epuck_pos_temp[i][np.newaxis, :], seq_targets, epuck_angle_raw[i])
                    seq_target_angle = (seq_target_angle + np.pi) % (2 * np.pi) - np.pi
                    seq_target_dis, seq_target_angle = seq_target_dis.squeeze(0), seq_target_angle.squeeze(0)
                    
                    seq_target_pos_diff[:, 0] = utilities.normalize_to_range(seq_target_pos_diff[:, 0], -self.obs_view, self.obs_view, -1, 1, clip=True)
                    seq_target_pos_diff[:, 1] = utilities.normalize_to_range(seq_target_pos_diff[:, 1], -self.obs_view, self.obs_view, -1, 1, clip=True)

                    # 找到距离最近的remainning_num个目标
                    remaining_num = self.num_obs_targets - num_observable_target
                    remaining_num = min(remaining_num, in_sequence_not_in_obs.sum(-1))
                    seq_target_dis_sorted_indices = np.argsort(seq_target_dis.squeeze(), axis=0)[:remaining_num]
                    seq_target_message = np.concatenate((
                        seq_target_pos_diff, seq_target_angle[:, np.newaxis],
                        utilities.normalize_to_range(seq_target_dis, self.dis_mm['min'],self.dis_mm['max'], 0, 1,clip=True)[:, np.newaxis]
                    ),axis=-1)
                    observation[
                        n, i,
                        target_obs_start + self.mutual_dim * num_observable_target: 
                        target_obs_start + self.mutual_dim * (num_observable_target + remaining_num)
                    ] = seq_target_message[seq_target_dis_sorted_indices].reshape(1,-1)

            if (self.steps[n] + 1) % self.args.update_obs_steps == 0:
                self.update_obs_target_sequence(n)

            target_all_message = np.where(self.alive_target_buf[n, :, None], target_pos, 0.0).reshape(-1)
            for i in range(self.num_agents):
                current_agent_info = np.concatenate((
                    epuck_pos[i: i+1], epuck_angle[i: i+1]
                ), axis=-1)
                other_agents_pos = np.concatenate((epuck_pos[:i], epuck_pos[i+1:]), axis=0)  # (num_agents-1, 2)
                other_agents_angle = np.concatenate((epuck_angle[:i], epuck_angle[i+1:]), axis=0)  # (num_agents-1, 1)
                other_agents_info = np.concatenate((other_agents_pos, other_agents_angle), axis=-1)
                agent_all_message = np.concatenate((current_agent_info, other_agents_info), axis=0)
                ordered_alive_agents = np.concatenate((
                    self.alive_agent_buf[n, i:i+1],
                    np.concatenate((self.alive_agent_buf[n, :i], self.alive_agent_buf[n, i+1:]), axis=0),
                ), axis=0)
                agent_all_message_mask = np.where(ordered_alive_agents[:, None], agent_all_message, 0.0).reshape(-1)
                state = np.concatenate((
                    agent_all_message_mask,
                    target_all_message,
                ), axis=-1)
                self.states_buf[n, i] = state

        return observation

    def get_reward(self, action):
        #读取所有目标的位置和所有机器人的位置
        distance = np.zeros((self.num_envs, self.num_agents, self.num_target), dtype=np.float32)
        for n in range(self.num_envs):
            target_pos = np.zeros((self.num_target, 2), dtype=np.float32)
            target_pos[:] = self.target_pos[n]
            epuck_pos = self.agent_pos[n,:,:2].copy()
            #计算每个机器人与每个目标之间的欧几里得距离，存入 distance 数组
            distance[n] = utilities.get_distance_from_target(epuck_pos, target_pos)

        #初始化奖励数组
        rew_all = np.zeros((self.num_envs, self.num_agents, 1), dtype=np.float32)
        if (self.steps==0.0).all() | self.early_stop.all():
            self.min_target_dis[:] = np.where(self.alive_target_buf[:, np.newaxis, :], distance, np.inf)
            return rew_all

        # Snapshot alive masks before capture updates so shaping and penalties are consistent.
        alive_agents_before = self.alive_agent_buf.copy()
        alive_targets_before = self.alive_target_buf.copy()
        alive_agent_expand = alive_agents_before[..., np.newaxis]
        alive_target_expand = np.repeat(self.alive_target_buf[:, np.newaxis, :], self.num_agents, axis=1)
        valid_pair = alive_agent_expand & alive_target_expand

        large_dis = 10.0
        cur_dis = np.where(valid_pair, distance, large_dis)
        best_dis = np.where(valid_pair, self.min_target_dis, large_dis)
        target_progress = np.maximum(best_dis - cur_dis, 0.0)
        progress_reward = target_progress.max(axis=2) / self.reward_progress_scale
        progress_reward = np.clip(progress_reward, 0.0, 1.0)
        has_alive_target = alive_targets_before.any(axis=1, keepdims=True)
        progress_reward = np.where(
            alive_agents_before & has_alive_target,
            progress_reward,
            0.0,
        )
        
        collision_hits = self._collision_hits()
        collision_hits = np.where(alive_agents_before, collision_hits, False)
        self.num_collision += np.sum(collision_hits, axis=-1, keepdims=True).astype(np.int32)
        collision_penalty = collision_hits.astype(np.float32)

        reach_reward = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        finish_reward = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        for env_idx in range(self.num_envs):
            for target_idx in range(self.num_target):
                if not self.alive_target_buf[env_idx, target_idx]:
                    continue

                alive_agent_idx = np.where(self.alive_agent_buf[env_idx])[0]
                if alive_agent_idx.size == 0:
                    continue

                target_dis = distance[env_idx, alive_agent_idx, target_idx]
                capture_agents = int(np.sum(target_dis < self.catch_distance))
                if capture_agents < self.capture_action_conditions:
                    continue

                sorted_idx = np.argsort(target_dis)[:self.capture_action_conditions]
                capture_idx = alive_agent_idx[sorted_idx]
                reach_reward[env_idx, capture_idx] += 1.0
                self.alive_agent_buf[env_idx, capture_idx] = False
                self.alive_target_buf[env_idx, target_idx] = False

            if alive_targets_before[env_idx].any() and not self.alive_target_buf[env_idx].any():
                remaining_ratio = max(
                    float(self.buffer_length - (self.steps[env_idx] + 1.0)) / float(self.buffer_length),
                    0.0,
                )
                finish_reward[env_idx, :] = 1.0 + remaining_ratio

        reward = (
            self.reward_progress_weight * progress_reward
            + self.reward_reach_weight * reach_reward
            + self.reward_finish_weight * finish_reward
            - self.reward_collision_weight * collision_penalty
        )


        reward += np.where(alive_agents_before, self.reward_time_penalty, 0.0)

        rew_all[:] = reward[..., np.newaxis]

        self.min_target_dis[:] = np.where(valid_pair, np.minimum(self.min_target_dis, distance), self.min_target_dis)
        self.min_target_dis[:] = np.where(self.alive_target_buf[:, np.newaxis, :], self.min_target_dis, np.inf)

        return rew_all

    def is_done(self):
        next_steps = np.minimum(self.steps + 1, self.buffer_length)
        cond = (~self.alive_target_buf.any(-1)) | (next_steps >= self.buffer_length) | self.early_stop[:,0]
        self.steps[:] = next_steps
        self.alive_agent_buf[:] = np.where(cond[:, np.newaxis].repeat(self.num_agents, 1), 0, self.alive_agent_buf)
        #print(self.early_stop)
        #print(cond)

        return (~self.alive_agent_buf)[..., np.newaxis]

    def update_obs_target_sequence(self, env_idx):
        seq = self.obs_target_sequence[env_idx].copy()  # num_agent, num_target, 2
        for t in range(self.num_target):
            if not self.alive_target_buf[env_idx, t]:
                continue

            obs_mask = np.any(seq[:, t, :] != 0.0, axis=-1)  # (num_agents,)
            obs_agents = np.where(obs_mask)[0]  # return the indices of agents that have observations

            if obs_agents.size < 1:
                continue
            avg_pos = seq[obs_agents, t, :].mean(axis=0)  # shape = (2,)
            seq[:, t, :] = avg_pos
        self.obs_target_sequence[env_idx] = seq
    
    def reset(self):
        if self.start_flag:
            for _ in range(self.args.exclude_steps):
                self._wb_supervisor_step(self.timestep)
                self.handle_receiver()
            self.start_flag = False
        self.steps[:] = 0
        self.alive_target_buf = np.ones((self.num_envs, self.num_target), dtype=np.bool_)
        self.alive_agent_buf = np.ones((self.num_envs, self.num_agents), dtype=np.bool_)
        self.distance = np.zeros((self.num_envs, self.num_agents, self.num_target), dtype=np.float32)
        self.min_target_dis = np.full((self.num_envs, self.num_agents, self.num_target), np.inf, dtype=np.float32)
        self.early_stop = np.zeros((self.num_envs, 1), dtype=np.bool_)
        self.num_collision = np.zeros((self.num_envs, 1), dtype=np.int32)
        self.obs_target_sequence = np.zeros((self.num_envs, self.num_agents, self.num_target, 2), dtype=np.float32)
        
        self.simulationResetPhysics()
        self.reset_positions()
        self.start_time = self._wb_supervisor().getTime()
        self._wb_supervisor_step(self.timestep)
        self.handle_receiver()

        obs = self.get_observations()

        return obs, self.states_buf.copy()

    def get_info(self):
        return None
