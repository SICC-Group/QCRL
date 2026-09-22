import numpy as np


class ConstraintManifoldProjector:
    """ATACOM-style safety projector for differential-drive wheel-ratio actions."""

    def __init__(
        self,
        world_size,
        safety_margin,
        dt,
        wheel_radius=0.0205,
        axle_length=0.053,
        max_wheel_speed=6.28,
        k_c=6.0,
        heading_gain=2.0,
        wall_turn_margin=0.2,
        agent_avoid_margin=0.2,
    ):
        self.world_size = np.asarray(world_size, dtype=np.float32)
        self.safety_margin = float(safety_margin)
        self.wall_turn_margin = float(max(wall_turn_margin, self.safety_margin))
        self.agent_avoid_margin = float(max(agent_avoid_margin, self.safety_margin))
        self.dt = float(max(dt, 1e-6))
        self.wheel_radius = float(wheel_radius)#小车轮子的半径(m)
        self.axle_length = float(axle_length)#小车轮子之间的距离(m)
        self.max_wheel_speed = float(max_wheel_speed)#小车轮子的最大转速(rad/s)
        self.k_c = float(k_c) #约束违反的惩罚增益
        self.heading_gain = float(heading_gain) #朝向误差的增益

        # 计算小车的最大线速度和角速度，以便在投影过程中进行速度限制
        self.v_max = self.wheel_radius * self.max_wheel_speed
        self.omega_max = 2.0 * self.wheel_radius * self.max_wheel_speed / self.axle_length

    def _build_constraints(self, q, others, arena_center):
        x, y = float(q[0]), float(q[1])
        cx, cy = float(arena_center[0]), float(arena_center[1])#场地的中心点位置坐标

        half_w, half_h = self.world_size[0] / 2.0, self.world_size[1] / 2.0

        # 约束1-4: 场地边界约束，确保小车在安全边界内
        x_min = cx - half_w + self.safety_margin
        x_max = cx + half_w - self.safety_margin
        y_min = cy - half_h + self.safety_margin
        y_max = cy + half_h - self.safety_margin

        # g(q) <= 0
        g_list = [
            x_min - x,
            x - x_max,
            y_min - y,
            y - y_max,
        ]
        j_list = [
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.0, -1.0], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]

        #机器人间避碰约束
        min_dist_sq = self.safety_margin ** 2
        for other in others:
            diff = q - other
            dist_sq = float(np.dot(diff, diff))
            g_list.append(min_dist_sq - dist_sq)
            j_list.append((-2.0 * diff).astype(np.float32))

        g = np.asarray(g_list, dtype=np.float32)#转换为NumPy数组
        j = np.asarray(j_list, dtype=np.float32)
        return g, j #g 是一组不等式约束 j是约束的雅可比矩阵

    def _nearest_soft_wall(self, q, arena_center):
        x, y = float(q[0]), float(q[1])
        cx, cy = float(arena_center[0]), float(arena_center[1])
        half_w, half_h = self.world_size[0] / 2.0, self.world_size[1] / 2.0

        bounds = (
            (x - (cx - half_w), np.array([1.0, 0.0], dtype=np.float32)),
            ((cx + half_w) - x, np.array([-1.0, 0.0], dtype=np.float32)),
            (y - (cy - half_h), np.array([0.0, 1.0], dtype=np.float32)),
            ((cy + half_h) - y, np.array([0.0, -1.0], dtype=np.float32)),
        )
        distance, inward = min(bounds, key=lambda item: item[0])
        return float(distance), inward

    def _apply_soft_wall_turn(self, v_raw, omega_raw, q, theta, arena_center):
        wall_distance, inward = self._nearest_soft_wall(q, arena_center)
        if wall_distance >= self.wall_turn_margin:
            return v_raw, omega_raw

        heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        outward_drive = -float(np.dot(heading, inward)) * max(float(v_raw), 0.0)
        if outward_drive <= 1e-6:
            return v_raw, omega_raw

        turn_strength = np.clip(
            (self.wall_turn_margin - wall_distance) / max(self.wall_turn_margin - self.safety_margin, 1e-6),
            0.0,
            1.0,
        )
        target_heading = float(np.arctan2(inward[1], inward[0]))
        heading_error = (target_heading - theta + np.pi) % (2.0 * np.pi) - np.pi

        v_safe = min(float(v_raw), self.v_max * (1.0 - turn_strength))
        omega_safe = float(omega_raw) + turn_strength * self.heading_gain * heading_error
        omega_safe = np.clip(omega_safe, -self.omega_max, self.omega_max)
        return v_safe, omega_safe

    def _apply_soft_agent_avoidance(self, v_raw, omega_raw, q, theta, others):
        if others.size == 0:
            return v_raw, omega_raw

        diff = q[np.newaxis, :] - others
        distances = np.linalg.norm(diff, axis=1)
        close_idx = int(np.argmin(distances))
        agent_distance = float(distances[close_idx])
        if agent_distance >= self.agent_avoid_margin:
            return v_raw, omega_raw

        heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        if agent_distance > 1e-6:
            away = (diff[close_idx] / agent_distance).astype(np.float32)
        else:
            away = -heading

        toward_other = -away
        approach_drive = float(np.dot(heading, toward_other)) * max(float(v_raw), 0.0)
        if approach_drive <= 1e-6:
            return v_raw, omega_raw

        avoid_strength = np.clip(
            (self.agent_avoid_margin - agent_distance)
            / max(self.agent_avoid_margin - self.safety_margin, 1e-6),
            0.0,
            1.0,
        )
        target_heading = float(np.arctan2(away[1], away[0]))
        heading_error = (target_heading - theta + np.pi) % (2.0 * np.pi) - np.pi

        v_safe = min(float(v_raw), self.v_max * (1.0 - avoid_strength))
        omega_safe = float(omega_raw) + avoid_strength * self.heading_gain * heading_error
        omega_safe = np.clip(omega_safe, -self.omega_max, self.omega_max)
        return v_safe, omega_safe

    #将单个智能体的原始归一化动作投影到安全动作空间
    def _project_one(self, action, q, theta, others, arena_center):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        wl_raw, wr_raw = action * self.max_wheel_speed
        v_raw = self.wheel_radius * 0.5 * (wl_raw + wr_raw)
        omega_raw = self.wheel_radius * (wr_raw - wl_raw) / self.axle_length
        v_raw, omega_raw = self._apply_soft_agent_avoidance(v_raw, omega_raw, q, theta, others)
        v_raw, omega_raw = self._apply_soft_wall_turn(v_raw, omega_raw, q, theta, arena_center)

        heading = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
        dq_raw = v_raw * heading

        g, j = self._build_constraints(q, others, arena_center)
        m = g.shape[0]

        # Slack equality: c(q, s) = g(q) + 0.5 * s^2 = 0
        s = np.sqrt(np.maximum(-2.0 * g, 0.0)).astype(np.float32)
        c = g + 0.5 * s ** 2

        j_c = np.zeros((m, 2 + m), dtype=np.float32)
        j_c[:, :2] = j
        j_c[:, 2:] = np.diag(s)

        # Build a tangent velocity directly.
        # For inactive inequalities (s_i > 0), choose slack rates so c_dot = 0
        # while preserving task-space motion dq as much as possible.
        tangent = np.zeros(2 + m, dtype=np.float32)
        tangent[:2] = dq_raw
        j_dq = j @ dq_raw
        valid_s = s > 1e-6
        tangent[2:][valid_s] = -j_dq[valid_s] / s[valid_s]

        # Discrete-time correction: only correct predicted inequality violations.
        # This avoids over-constraining interior motions (which caused in-place spinning).
        q_next = q + tangent[:2] * self.dt
        g_next, _ = self._build_constraints(q_next, others, arena_center)
        violation = np.maximum(np.maximum(g, g_next), 0.0)
        correction_q = np.zeros(2, dtype=np.float32)
        if np.any(violation > 0.0):
            correction_q = -j.T @ (self.k_c * violation)
            corr_norm = np.linalg.norm(correction_q)
            if corr_norm > self.v_max:
                correction_q *= self.v_max / max(corr_norm, 1e-9)

        safe_aug = tangent.copy()
        safe_aug[:2] += correction_q
        dq_safe = safe_aug[:2]

        forward_comp = float(np.dot(dq_safe, heading))
        lateral_comp = float(heading[0] * dq_safe[1] - heading[1] * dq_safe[0])
        heading_error = np.arctan2(lateral_comp, max(abs(forward_comp), 1e-6))

        omega_safe = np.clip(omega_raw + self.heading_gain * heading_error, -self.omega_max, self.omega_max)
        v_safe = np.clip(forward_comp, -self.v_max, self.v_max)

        wl = (2.0 * v_safe - omega_safe * self.axle_length) / (2.0 * self.wheel_radius)
        wr = (2.0 * v_safe + omega_safe * self.axle_length) / (2.0 * self.wheel_radius)

        return np.clip(np.array([wl, wr], dtype=np.float32) / self.max_wheel_speed, -1.0, 1.0)

    def project(self, actions, positions, headings, alive_mask, arena_centers):
        actions = np.asarray(actions, dtype=np.float32)
        projected = np.clip(actions.copy(), -1.0, 1.0)

        num_envs, num_agents, _ = projected.shape
        for env_idx in range(num_envs):
            env_center = arena_centers[env_idx]
            env_alive = alive_mask[env_idx].astype(bool)
            env_pos = positions[env_idx]
            env_headings = headings[env_idx]

            for agent_idx in range(num_agents):
                if not env_alive[agent_idx]:
                    projected[env_idx, agent_idx] = 0.0
                    continue
                others_idx = np.arange(num_agents)
                others_idx = others_idx[others_idx != agent_idx]
                others = env_pos[others_idx] if others_idx.size > 0 else np.zeros((0, 2), dtype=np.float32)

                projected[env_idx, agent_idx] = self._project_one(
                    projected[env_idx, agent_idx],
                    env_pos[agent_idx],
                    env_headings[agent_idx],
                    others,
                    env_center,
                )

        return projected

    def project_agent_batch(self, actions, positions, headings, alive_mask, arena_centers, agent_ids):
        actions = np.asarray(actions, dtype=np.float32)
        projected = np.clip(actions.copy(), -1.0, 1.0)
        positions = np.asarray(positions, dtype=np.float32)
        headings = np.asarray(headings, dtype=np.float32)
        alive_mask = np.asarray(alive_mask, dtype=np.bool_)
        arena_centers = np.asarray(arena_centers, dtype=np.float32)
        agent_ids = np.asarray(agent_ids, dtype=np.int64).reshape(-1)

        if projected.ndim != 2:
            raise ValueError(f"actions must have shape (batch, action_dim), got {projected.shape}")
        if positions.ndim != 3:
            raise ValueError(f"positions must have shape (batch, num_agents, 2), got {positions.shape}")

        batch_size = projected.shape[0]
        for batch_idx in range(batch_size):
            agent_idx = int(agent_ids[batch_idx])
            env_alive = alive_mask[batch_idx].astype(bool)
            env_pos = positions[batch_idx]

            if agent_idx < 0 or agent_idx >= env_pos.shape[0]:
                raise ValueError(f"agent_id {agent_idx} is outside num_agents={env_pos.shape[0]}")

            if not env_alive[agent_idx]:
                projected[batch_idx] = 0.0
                continue

            others_idx = np.arange(env_pos.shape[0])
            others_idx = others_idx[others_idx != agent_idx]
            others = env_pos[others_idx] if others_idx.size > 0 else np.zeros((0, 2), dtype=np.float32)

            projected[batch_idx] = self._project_one(
                projected[batch_idx],
                env_pos[agent_idx],
                headings[batch_idx, agent_idx],
                others,
                arena_centers[batch_idx],
            )

        return projected
