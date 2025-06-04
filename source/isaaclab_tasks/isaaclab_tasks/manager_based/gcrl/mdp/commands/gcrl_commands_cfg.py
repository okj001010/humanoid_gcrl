# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .gcrl_command import RootXYPosCommand


@configclass
class RootXYPosCommandCfg(CommandTermCfg):
    """Configuration for the uniform root xy position command generator."""

    class_type: type = RootXYPosCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the root xy position commands."""

        root_x: tuple[float, float] = MISSING
        """Range for the root-x position command (in m)."""

        root_y: tuple[float, float] = MISSING
        """Range for the root-y position command (in m)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the root xy position commands."""

    goal_root_pos_visualizer_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/root_pos_goal",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            )
        }
    )
    """The configuration for the goal root position visualization marker. Defaults to sphere with RED color."""

    current_root_pos_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/root_pos_current")
    """The configuration for the current root position visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_root_pos_visualizer_cfg.markers["sphere"].scale = (0.5, 0.5, 0.5)
    current_root_pos_visualizer_cfg.markers["arrow"].scale = (0.25, 0.25, 0.25)