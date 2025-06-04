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
        self.goal_command_w[env_ids, 0] = r.uniform_(*self.cfg.ranges.root_x)
        self.goal_command_w[env_ids, 1] = r.uniform_(*self.cfg.ranges.root_y)

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
        self.goal_root_pos_visualizer.visualize(self.vis_goal_root_pos_w)
        # -- current root xy pos
        vis_current_root_pos_w = torch.zeros_like(self.robot.data.root_pos_w)
        vis_current_root_pos_w[:, :2] = self.robot.data.root_pos_w[:, :2]
        diff_arrow_scale, diff_arrow_quat = self._resolve_xy_pos_diff_to_arrow(self.goal_command_b)
        self.current_root_pos_visualizer.visualize(vis_current_root_pos_w, diff_arrow_quat, diff_arrow_scale)
        
    """
    Internal helpers.
    """

    def _resolve_xy_pos_diff_to_arrow(self, xy_pos_diff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY root position command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.current_root_pos_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_pos_diff.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_pos_diff, dim=1) * 3.0
        
        # arrow-direction
        heading_angle = torch.atan2(xy_pos_diff[:, 1], xy_pos_diff[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
