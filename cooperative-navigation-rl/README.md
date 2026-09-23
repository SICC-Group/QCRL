### Webots Worlds | webots世界
运行`./worlds/generate_wbt.py`生成世界文件`generated_world.wbt`，注意修改`num_envs`和`num_agents`


### RL info | 强化学习信息
#### observation | 观测

#### individual state | 全局状态
`num_agent` * (2`pos` + 1`rotation`) + `num_target` * (2`pos`)


#### action | 动作
每个智能体动作为二维连续向量：`[left_wheel_ratio, right_wheel_ratio]`。

- 取值范围：`[-1, 1]`
- 执行方式：左右轮速度分别为 `max_speed * left_wheel_ratio` 与 `max_speed * right_wheel_ratio`
- 默认启用 **约束流形安全层**：先将动作映射为平面速度，再基于不等式约束 `g(q)<=0` 的 slack 等式化、雅可比与切空间投影进行纠偏，最终发送安全动作到轮子控制器。

#### constraint manifold safety | 约束流形安全层
- 主要约束：边界安全距离、智能体间最小间距（用于降低碰撞）。
- 离散时间纠偏：包含一步预测误差纠偏项，适配 Webots 离散步进。
- 关键参数（`config.py`）：
  - `--use_constraint_manifold`（默认启用）
  - `--manifold_kc`
  - `--manifold_heading_gain`
  - `--manifold_safety_margin`
  - `--manifold_wheel_radius`
  - `--manifold_axle_length`

#### reward | 奖励
- 距离奖励，系数`-1`
    - 计算每个智能体到最近目标的距离，并相对于初始最近距离，归一化到`[0, 1]`之间，距离越远惩罚越大（-1倍）
    - 如果当前距离小于历史距离，给予`+1`奖励
    - 存活目标掩码，只对存活的智能体计算奖励
- 碰撞惩罚，系数`-10`
    - 判断红外传感器值是否大于阈值，大于则记为存在障碍物
- 捕获奖励，系数`50`
    - 捕获后，智能体死亡

### Log info | 日志记录
train: `reward`; eval: `num_collision` & `num_target`
