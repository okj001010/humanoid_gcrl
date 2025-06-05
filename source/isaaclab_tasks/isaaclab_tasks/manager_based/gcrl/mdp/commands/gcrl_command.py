# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import RootXYPosCommandCfg


class RootXYPosCommand(CommandTerm):
    """Command generator for generating root xy position command uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space.

    The position command are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.
    """

    cfg: RootXYPosCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: RootXYPosCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        
        # intial robot state
        self.init_root_xy_pos_w = torch.zeros(self.num_envs, 2, device=self.device)

        # create buffers
        # -- commands: (x, y) in world frame -> transformation to base frame
        self.goal_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.goal_command_w = torch.zeros(self.num_envs, 2, device=self.device)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "RootXYPosCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired root xy position command. Shape is (num_envs, 2) in (x, y) format.
        """
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        
        diff_pos = torch.zeros(self.num_envs, 3, device=self.device)
        diff_pos[:, :2] = self.goal_command_w - root_pos_w[:, :2]
        
        self.goal_command_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(root_quat_w), diff_pos)[:, :2]
        return self.goal_command_b
    
    """
    Operations
    """
    
    def _average_metric_value_over_environments(self, metric_value, env_ids):
        return torch.mean(metric_value[env_ids]).item()
    
    def reset(self, env_ids: Sequence[int]) -> dict[str, float | torch.Tensor]:
        # save initial root xy position in world frame
        self.init_root_xy_pos_w[env_ids] = self.robot.data.root_state_w[env_ids, :2]
        
        # logs after a reset
        extras = {}
        extras["position_error"] = self._average_metric_value_over_environments(
            self.metrics["position_error"], env_ids
        )
        
        return extras

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the relative position error between the goal and current root xy position
        current_root_pos_w = self.robot.data.root_pos_w[:, :2]
        pos_error = self.goal_command_w - current_root_pos_w
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- root xy position
        r = torch.empty(len(env_ids), device=self.device)
        self.goal_command_w[env_ids, :] = self.init_root_xy_pos_w[env_ids, :]
        self.goal_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.root_x)
        self.goal_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.root_y)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_root_pos_visualizer"):
                # -- goal root xy pos
                self.goal_root_pos_visualizer = VisualizationMarkers(self.cfg.goal_root_pos_visualizer_cfg)
                # -- current root xy pos
                self.current_root_pos_visualizer = VisualizationMarkers(self.cfg.current_root_pos_visualizer_cfg)
            # set their visibility to true
            self.goal_root_pos_visualizer.set_visibility(True)
            self.current_root_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_root_pos_visualizer"):
                self.goal_root_pos_visualizer.set_visibility(False)
                self.current_root_pos_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal root xy pos
        vis_goal_root_pos_w = torch.zeros_like(self.robot.data.root_pos_w)
        vis_goal_root_pos_w[:, :2] = self.goal_command_w[:, :2]
        vis_goal_root_pos_w[:, 2] = 0.1
        self.goal_root_pos_visualizer.visualize(vis_goal_root_pos_w)
        
        # -- current root xy pos with arrow starting from current position
        arrow_start_pos, diff_arrow_quat, diff_arrow_scale = self._resolve_xy_pos_diff_to_arrow_from_start(
            self.robot.data.root_pos_w, self.goal_command_w
        )
        self.current_root_pos_visualizer.visualize(arrow_start_pos, diff_arrow_quat, diff_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_pos_diff_to_arrow_from_start(
        self, current_pos_w: torch.Tensor, goal_pos_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts the position difference to arrow starting from current position pointing to goal."""
        # Calculate position difference in world frame
        pos_diff_w = goal_pos_w[:, :2] - current_pos_w[:, :2]
        
        # obtain default scale of the marker
        default_scale = self.current_root_pos_visualizer.cfg.markers["arrow"].scale
        
        # Calculate arrow length based on distance to goal
        arrow_length = torch.linalg.norm(pos_diff_w, dim=1)
        
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(current_pos_w.shape[0], 1)
        arrow_scale[:, 0] = arrow_length
        
        # arrow-direction
        heading_angle = torch.atan2(pos_diff_w[:, 1], pos_diff_w[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # Calculate arrow start position
        direction_unit = pos_diff_w / (torch.linalg.norm(pos_diff_w, dim=1, keepdim=True) + 1e-8)
        arrow_offset = direction_unit * (arrow_length.unsqueeze(1) * default_scale[0] / 2.0)
        
        arrow_start_pos = torch.zeros_like(current_pos_w)
        arrow_start_pos[:, :2] = current_pos_w[:, :2] + arrow_offset
        arrow_start_pos[:, 2] = 0.1

        return arrow_start_pos, arrow_quat, arrow_scale
