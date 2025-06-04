# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pelvis_bad_ori(
    env: ManagerBasedRLEnv,
    limit_euler_angle: list[float] = [0.5, 1.5],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's orientation is out of predefined range

    Args:
        limit_euler_angle: euler angle threshold [roll, pitch]. Episode
            will be terminated if the abs of the root euler angle will
            exceed this threshold
    """
    asset: Articulation = env.scene[asset_cfg.name]
    euler = math_utils.wrap_to_pi(
        torch.stack(math_utils.euler_xyz_from_quat(asset.data.root_quat_w), dim=-1)
    )
    out_of_limit = torch.logical_or(
        torch.abs(euler[..., 0]) > limit_euler_angle[0],
        torch.abs(euler[..., 1]) > limit_euler_angle[1],
    )

    return out_of_limit