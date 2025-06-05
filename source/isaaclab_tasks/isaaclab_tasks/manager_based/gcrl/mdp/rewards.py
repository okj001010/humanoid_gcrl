# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class progress_to_target(ManagerTermBase):
    """Dense reward for making progress towards the target based on the spatial distance to the target."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self.step_counter = torch.zeros(env.num_envs, dtype=torch.int32, device=env.device)
        # distance history
        self.distance = torch.zeros(env.num_envs, device=env.device)
        self.prev_distance = torch.zeros_like(self.distance)

    def reset(self, env_ids: torch.Tensor):
        """Reset the reward term for specified environments."""
        self.step_counter[env_ids] = 0
        # reset distance tracking for clean start
        self.distance[env_ids] = 0.0
        self.prev_distance[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        
        # store previous distance
        self.prev_distance[:] = self.distance[:]
        
        # calculate current distance to the goal
        goal_pos = env.command_manager.get_command(command_name)
        self.distance[:] = torch.norm(goal_pos - asset.data.root_pos_w[:, :2], p=2, dim=-1)
        
        # calculate progress reward (only calculate after first step)
        progress_reward = torch.where(
            self.step_counter > 0,
            self.prev_distance - self.distance,
            torch.zeros_like(self.distance)
        )
        
        self.step_counter += 1
        return progress_reward

class goal_reached_with_low_energy(ManagerTermBase):
    """Sparse reward for reaching the goal with low energy consumption."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        # total energy consumption
        self.energy_consumption = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor):
        """Reset the reward term for specified environments."""
        self.energy_consumption[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float = 0.1,
        energy_scale: float = 0.01,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        
        # calculate distance to the goal
        goal_pos = env.command_manager.get_command(command_name)
        to_goal_pos = torch.norm(goal_pos - asset.data.root_pos_w[:, :2], p=2, dim=-1)
        
        # calculate energy consumption
        power = torch.sum(torch.abs(asset.data.joint_vel * asset.data.applied_torque), dim=1)
        self.energy_consumption[:] += power * env.step_dt
        
        # check if goal is reached
        goal_reached = (to_goal_pos < threshold).float()
        
        # return scaled reward based on energy efficiency
        energy_reward = torch.exp(-self.energy_consumption * energy_scale)

        return goal_reached * energy_reward