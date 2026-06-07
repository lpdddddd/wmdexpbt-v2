"""
DexPBT Kuka-Allegro single-arm environment for Isaac Lab.

Port of IsaacGymEnvs AllegroKukaBase + DexPBT mixin to DirectRLEnv.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.views import XformPrimView
from isaaclab.utils.math import euler_xyz_from_quat, sample_uniform, saturate

from .dexpbt_utils import (
    quat_conjugate,
    quat_mul,
    quat_rotate,
    scale,
    tolerance_successes_objective,
    unscale,
)
from .pbt.pbt_curriculum_p1 import apply_env_tolerance_curriculum

if TYPE_CHECKING:
    from .env_cfg import DexPBTKukaAllegroEnvCfg


def _filter_thin_plates(scales: list[int]) -> bool:
    scales = sorted(scales)
    return scales[0] * 3 <= scales[1]


def _filter_non_elongated(scales: list[int]) -> bool:
    scales = sorted(scales)
    return scales[2] <= scales[0] * 3 or scales[2] <= scales[1] * 3


def _append_scale_candidates(
    candidates: list[list[float]],
    scales: list[int],
    min_volume: float,
    max_volume: float,
    filter_funcs: list,
) -> None:
    for x_scale in scales:
        for y_scale in scales:
            for z_scale in scales:
                volume = x_scale * y_scale * z_scale / (100 * 100 * 100)
                if volume > max_volume or volume < min_volume:
                    continue
                curr_scales = [x_scale, y_scale, z_scale]
                sorted_scales = sorted(curr_scales)
                if any(filter_func(sorted_scales) for filter_func in filter_funcs):
                    continue
                candidates.append([x_scale / 100.0, y_scale / 100.0, z_scale / 100.0])


class DexPBTKukaAllegroEnv(DirectRLEnv):
    """
    Base DexPBT environment for Kuka+Allegro single-arm manipulation.

    Subclasses (DexPBTRegraspingEnv, DexPBTThrowEnv, DexPBTReorientationEnv) override:
      - _object_keypoint_offsets()
      - _reset_target(env_ids)
      - _extra_reset_rules(terminated)
    """

    cfg: DexPBTKukaAllegroEnvCfg

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: DexPBTKukaAllegroEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- body index resolution ---
        self.palm_body_idx = self.robot.body_names.index(self.cfg.palm_body_name)
        self.fingertip_body_indices = [
            self.robot.body_names.index(n) for n in self.cfg.fingertip_body_names
        ]
        self.num_fingertips = len(self.fingertip_body_indices)

        # --- joint index resolution ---
        self.arm_joint_indices = [self.robot.joint_names.index(n) for n in self.cfg.arm_joint_names]
        self.hand_joint_indices = [self.robot.joint_names.index(n) for n in self.cfg.hand_joint_names]
        self.all_joint_indices = self.arm_joint_indices + self.hand_joint_indices
        self.num_arm_dofs = len(self.arm_joint_indices)
        self.num_hand_dofs = len(self.hand_joint_indices)
        self.num_total_dofs = self.num_arm_dofs + self.num_hand_dofs

        # --- joint limits ---
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = joint_pos_limits[0, :, 0]  # (num_dofs,)
        self.dof_upper = joint_pos_limits[0, :, 1]

        # --- action target buffers ---
        self.prev_targets = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
        self.cur_targets = torch.zeros_like(self.prev_targets)
        # initialize targets to default joint positions
        self.prev_targets[:] = self.robot.data.default_joint_pos
        self.cur_targets[:] = self.robot.data.default_joint_pos

        # --- fingertip / palm offsets ---
        self.fingertip_offsets_t = torch.tensor(
            self.cfg.fingertip_offsets, device=self.device, dtype=torch.float
        ).unsqueeze(0).expand(self.num_envs, -1, -1)  # (N, 4, 3)
        self.palm_center_offset_t = torch.tensor(
            self.cfg.palm_center_offset, device=self.device, dtype=torch.float
        ).unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)

        # --- keypoints ---
        raw_offsets = self._object_keypoint_offsets()
        self.num_keypoints = len(raw_offsets)
        self.base_object_keypoint_offsets = torch.tensor(raw_offsets, device=self.device, dtype=torch.float)

        # --- object scales / geometry ---
        self.object_scales = self._sample_object_scales()
        self.object_keypoint_offsets = torch.zeros(self.num_envs, self.num_keypoints, 3, device=self.device)
        self._refresh_object_keypoint_offsets()
        self.object_xform_view = XformPrimView(self.cfg.object_cfg.prim_path, device=self.device, validate_xform_ops=False)
        self.object_xform_view.set_scales(self.object_scales)

        # --- target sampling volume ---
        self.target_volume_origin = torch.tensor([0.0, 0.05, 0.8], device=self.device)
        self.target_volume_extent = torch.tensor(
            [[-0.4, 0.4], [-0.05, 0.3], [-0.12, 0.25]], device=self.device
        )

        # --- goal state ---
        self.goal_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.goal_rot[:, 0] = 1.0  # wxyz identity
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self.bucket_pos = torch.zeros(self.num_envs, 3, device=self.device) if hasattr(self, "bucket") else None
        self.table_top_z = 0.255
        self.object_rest_heights = self.table_top_z + 0.5 * self.cfg.object_base_size * self.object_scales[:, 2]

        # --- episode tracking ---
        self.successes = torch.zeros(self.num_envs, device=self.device)
        self.prev_episode_successes = torch.zeros(self.num_envs, device=self.device)
        self.true_objective = torch.zeros(self.num_envs, device=self.device)
        self.near_goal_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.lifted_object = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # closest distances (initialized to -1 => first-frame init)
        self.closest_fingertip_dist = torch.full(
            (self.num_envs, self.num_fingertips), -1.0, device=self.device
        )
        self.closest_keypoint_max_dist = torch.full((self.num_envs,), -1.0, device=self.device)
        self.furthest_hand_dist = torch.full((self.num_envs,), -1.0, device=self.device)

        # DexPBT attempt-level tracking
        self.attempt_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.closest_hand_obj_d = torch.full((self.num_envs,), -1.0, device=self.device)
        self.closest_d_hat = torch.full((self.num_envs,), -1.0, device=self.device)
        self.prev_is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # tolerance curriculum state
        self.success_tolerance = self.cfg.success_tolerance
        self.initial_tolerance = self.cfg.success_tolerance
        self.target_tolerance = self.cfg.target_success_tolerance
        self.tolerance_curriculum_interval = self.cfg.tolerance_curriculum_interval
        self.last_curriculum_update = 0
        self.frame_since_restart = 0
        # Patch H (2026-05-31): post-restart curriculum freeze. 0 means
        # no freeze (legacy behaviour). PBT restarts set this via
        # --curriculum_freeze_frames_after_restart_resume.
        self.curriculum_freeze_frames_after_restart = 0
        # Patch P1 (2026-06-01): mastery curriculum via observer control JSON.
        self.curriculum_mode = "legacy"
        self.curriculum_control_path = None
        self.curriculum_shrink_factor = 0.95

        # dt for action integration
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # intermediate value cache
        self.palm_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.palm_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.palm_center_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_rot = torch.zeros(self.num_envs, 4, device=self.device)

        # cached values for _get_dones / _get_rewards coupling
        self._cached_reward = torch.zeros(self.num_envs, device=self.device)
        self._latest_is_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # reward logging accumulators
        self.rewards_episode = {}
        for k in ["reward_reach", "reward_pick", "reward_pick_bonus",
                   "reward_target", "reward_success_bonus", "reward_vel_penalty", "reward_total"]:
            self.rewards_episode[k] = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------------ #
    # Scene
    # ------------------------------------------------------------------ #

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        if self.cfg.bucket_cfg is not None:
            self.bucket = RigidObject(self.cfg.bucket_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # spawn a table (static cuboid)
        table_cfg = sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        table_cfg.func(
            "/World/envs/env_.*/table",
            table_cfg,
            translation=(-0.55, 0.0, 0.235),
        )

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        if self.cfg.bucket_cfg is not None:
            self.scene.rigid_objects["bucket"] = self.bucket

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # Abstract hooks for subclasses
    # ------------------------------------------------------------------ #

    def _object_keypoint_offsets(self) -> list[list[float]]:
        """Return keypoint offsets in unit cube coordinates. Override in subclass."""
        return [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]

    def _reset_target(self, env_ids: Tensor) -> None:
        """Reset goal position / orientation for given envs. Override in subclass."""
        pass

    def _extra_reset_rules(self, terminated: Tensor) -> Tensor:
        """Add task-specific termination rules. Override in subclass."""
        return terminated

    # ------------------------------------------------------------------ #
    # Intermediate computations (called before obs / reward)
    # ------------------------------------------------------------------ #

    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None):
        """Derive fingertip, palm, object, keypoint quantities from raw sim data."""
        origins = self.scene.env_origins  # (N, 3)

        # palm
        self.palm_pos[:] = self.robot.data.body_pos_w[:, self.palm_body_idx] - origins
        self.palm_rot[:] = self.robot.data.body_quat_w[:, self.palm_body_idx]
        self.palm_center_pos[:] = self.palm_pos + quat_rotate(self.palm_rot, self.palm_center_offset_t)

        # palm velocities (lin + ang) in world frame
        palm_vel = self.robot.data.body_vel_w[:, self.palm_body_idx]  # (N,6)
        self.palm_linvel = palm_vel[:, 0:3]
        self.palm_angvel = palm_vel[:, 3:6]

        # fingertips
        ft_pos_w = self.robot.data.body_pos_w[:, self.fingertip_body_indices]  # (N,4,3)
        ft_rot_w = self.robot.data.body_quat_w[:, self.fingertip_body_indices]  # (N,4,4)
        self.fingertip_pos = ft_pos_w - origins.unsqueeze(1)
        self.fingertip_rot = ft_rot_w

        # fingertip contact points (with offset)
        self.fingertip_pos_offset = torch.zeros_like(self.fingertip_pos)
        for i in range(self.num_fingertips):
            self.fingertip_pos_offset[:, i] = self.fingertip_pos[:, i] + quat_rotate(
                self.fingertip_rot[:, i], self.fingertip_offsets_t[:, i]
            )

        # object
        self.object_pos[:] = self.object.data.root_pos_w - origins
        self.object_rot[:] = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w
        if self.bucket_pos is not None:
            self.bucket_pos[:] = self.bucket.data.root_pos_w - origins

        # fingertip distances to object
        obj_repeat = self.object_pos.unsqueeze(1).expand_as(self.fingertip_pos_offset)
        self.fingertip_pos_rel_object = self.fingertip_pos_offset - obj_repeat
        self.curr_fingertip_distances = torch.norm(self.fingertip_pos_rel_object, dim=-1)  # (N,4)

        # update closest fingertip distance (init on first frame when == -1)
        self.closest_fingertip_dist = torch.where(
            self.closest_fingertip_dist < 0.0, self.curr_fingertip_distances, self.closest_fingertip_dist
        )
        self.furthest_hand_dist = torch.where(
            self.furthest_hand_dist < 0.0, self.curr_fingertip_distances[:, 0], self.furthest_hand_dist
        )

        # fingertip relative to palm center
        palm_repeat = self.palm_center_pos.unsqueeze(1).expand_as(self.fingertip_pos_offset)
        self.fingertip_pos_rel_palm = self.fingertip_pos_offset - palm_repeat  # (N,4,3)

        # keypoints
        self.obj_keypoint_pos = torch.zeros(self.num_envs, self.num_keypoints, 3, device=self.device)
        self.goal_keypoint_pos = torch.zeros_like(self.obj_keypoint_pos)
        for i in range(self.num_keypoints):
            self.obj_keypoint_pos[:, i] = self.object_pos + quat_rotate(
                self.object_rot, self.object_keypoint_offsets[:, i]
            )
            self.goal_keypoint_pos[:, i] = self.goal_pos + quat_rotate(
                self.goal_rot, self.object_keypoint_offsets[:, i]
            )

        self.keypoints_rel_goal = self.obj_keypoint_pos - self.goal_keypoint_pos
        palm_kp_repeat = self.palm_center_pos.unsqueeze(1).expand(self.num_envs, self.num_keypoints, 3)
        self.keypoints_rel_palm = self.obj_keypoint_pos - palm_kp_repeat
        self.keypoint_distances_l2 = torch.norm(self.keypoints_rel_goal, dim=-1)
        self.keypoints_max_dist = self.keypoint_distances_l2.max(dim=-1).values

        self.closest_keypoint_max_dist = torch.where(
            self.closest_keypoint_max_dist < 0.0, self.keypoints_max_dist, self.closest_keypoint_max_dist
        )

        # DOF state
        self.dof_pos = self.robot.data.joint_pos  # (N, num_joints)
        self.dof_vel = self.robot.data.joint_vel

    def _sample_object_scales(self) -> Tensor:
        """Sample per-env object dimensions following the original DexPBT cuboid families."""
        if not self.cfg.randomize_object_dimensions:
            return torch.ones(self.num_envs, 3, device=self.device)

        candidates: list[list[float]] = [[1.0, 1.0, 1.0]]
        if self.cfg.with_small_cuboids:
            _append_scale_candidates(
                candidates,
                [100, 50, 66, 75, 90, 110, 125, 150, 175, 200, 250, 300],
                1.0,
                2.5,
                [],
            )
        if self.cfg.with_big_cuboids:
            _append_scale_candidates(
                candidates,
                [100, 125, 150, 200, 250, 300, 350],
                2.5,
                15.0,
                [_filter_thin_plates],
            )
        if self.cfg.with_sticks:
            _append_scale_candidates(
                candidates,
                [100, 50, 75, 200, 300, 400, 500, 600],
                2.5,
                6.0,
                [_filter_thin_plates, _filter_non_elongated],
            )
        candidate_tensor = torch.tensor(candidates, device=self.device, dtype=torch.float)
        sample_ids = torch.randint(0, candidate_tensor.shape[0], (self.num_envs,), device=self.device)
        return candidate_tensor[sample_ids]

    def _refresh_object_keypoint_offsets(self) -> None:
        kp_scale = self.cfg.keypoint_scale * self.cfg.object_base_size / 2.0
        self.object_keypoint_offsets[:] = self.base_object_keypoint_offsets.unsqueeze(0) * (
            self.object_scales.unsqueeze(1) * kp_scale
        )

    # ------------------------------------------------------------------ #
    # DirectRLEnv interface
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.frame_since_restart += 1

        # handle goal resets from previous step (before timer increment,
        # so reset envs go 0→1 matching the original post_physics_step order)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._do_reset_target(goal_env_ids)

        self.attempt_timer += 1

    def _apply_action(self) -> None:
        hand_ids = self.hand_joint_indices
        arm_ids = self.arm_joint_indices

        # --- Finger DOFs (7..22 in action): absolute position scaled + EMA ---
        hand_actions = self.actions[:, self.num_arm_dofs:]  # (N, 16)
        raw = scale(
            hand_actions,
            self.dof_lower[hand_ids].unsqueeze(0),
            self.dof_upper[hand_ids].unsqueeze(0),
        )
        ema = (
            self.cfg.act_moving_average * raw
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, hand_ids]
        )
        self.cur_targets[:, hand_ids] = ema.clamp(
            self.dof_lower[hand_ids].unsqueeze(0),
            self.dof_upper[hand_ids].unsqueeze(0),
        )

        # --- Arm DOFs (0..6 in action): delta-velocity integration ---
        arm_actions = self.actions[:, :self.num_arm_dofs]  # (N, 7)
        arm_targets = self.prev_targets[:, arm_ids] + self.cfg.dof_speed_scale * self.dt * arm_actions
        self.cur_targets[:, arm_ids] = arm_targets.clamp(
            self.dof_lower[arm_ids].unsqueeze(0),
            self.dof_upper[arm_ids].unsqueeze(0),
        )

        self.prev_targets[:] = self.cur_targets
        self.robot.set_joint_position_target(self.cur_targets)

    def _get_dones(self) -> tuple[Tensor, Tensor]:
        self._compute_intermediate_values()

        # tolerance curriculum runs before reward (matching original post_physics_step order)
        self._extra_curriculum()

        # compute reward first (determines is_success, which affects progress reset)
        self._cached_reward = self._compute_dexpbt_reward()

        # termination: object fallen
        terminated = (self.object_pos[:, 2] < 0.1).float()
        # termination: attempt timeout
        terminated = torch.where(
            self.attempt_timer >= self.cfg.attempt_timeout_steps,
            torch.ones_like(terminated),
            terminated,
        )
        # task-specific rules (e.g. hand too far for reorientation)
        terminated = self._extra_reset_rules(terminated)

        # zero episode_length_buf on success (extends episode for chaining successes,
        # matching original _compute_resets: progress_buf = where(is_success, 0, progress_buf))
        if self.cfg.max_consecutive_successes > 0:
            self.episode_length_buf = torch.where(
                self._latest_is_success,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )

        # time-out (uses potentially zeroed episode_length_buf)
        time_out = (self.episode_length_buf >= self.max_episode_length - 1).float()

        # max consecutive successes
        if self.cfg.max_consecutive_successes > 0:
            max_succ = (self.successes >= self.cfg.max_consecutive_successes).float()
            time_out = torch.max(time_out, max_succ)

        # Debug-only diagnostics for episode termination behavior.
        if "log" not in self.extras:
            self.extras["log"] = {}
        done = torch.max(terminated, time_out)
        self.extras["log"]["debug_terminated_rate"] = terminated.mean()
        self.extras["log"]["debug_timeout_rate"] = time_out.mean()
        self.extras["log"]["debug_done_rate"] = done.mean()

        return terminated, time_out

    def _get_rewards(self) -> Tensor:
        return self._cached_reward

    def _get_observations(self) -> dict:
        obs = self._compute_full_state_obs()
        if self.cfg.clamp_abs_observations > 0:
            obs.clamp_(-self.cfg.clamp_abs_observations, self.cfg.clamp_abs_observations)
        # Debug-only observation stats to spot dead/NaN inputs quickly.
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["debug_obs_mean"] = obs.mean()
        self.extras["log"]["debug_obs_std"] = obs.std()
        self.extras["log"]["debug_obs_min"] = obs.min()
        self.extras["log"]["debug_obs_max"] = obs.max()
        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids_t = torch.tensor(env_ids, device=self.device, dtype=torch.long) if not isinstance(env_ids, Tensor) else env_ids

        super()._reset_idx(env_ids)

        # reset goal
        self._do_reset_target(env_ids_t)

        # reset object pose with noise
        obj_default = self.object.data.default_root_state[env_ids_t].clone()
        rand_pos = sample_uniform(-1.0, 1.0, (len(env_ids_t), 3), device=self.device)
        obj_default[:, 0] += self.cfg.reset_position_noise_x * rand_pos[:, 0]
        obj_default[:, 1] += self.cfg.reset_position_noise_y * rand_pos[:, 1]
        obj_default[:, 2] = self.object_rest_heights[env_ids_t] + self.cfg.reset_position_noise_z * rand_pos[:, 2]
        # random orientation (uniform quaternion, wxyz)
        obj_default[:, 3:7] = self._random_quat(len(env_ids_t))
        obj_default[:, 7:] = 0.0
        # write in world frame
        obj_default[:, 0:3] += self.scene.env_origins[env_ids_t]
        self.object.write_root_pose_to_sim(obj_default[:, :7], env_ids=env_ids_t)
        self.object.write_root_velocity_to_sim(obj_default[:, 7:13], env_ids=env_ids_t)

        # reset robot DOFs with noise
        default_dof = self.robot.data.default_joint_pos[env_ids_t].clone()
        delta_max = self.dof_upper.unsqueeze(0) - default_dof
        delta_min = self.dof_lower.unsqueeze(0) - default_dof

        rand_dof = sample_uniform(0.0, 1.0, (len(env_ids_t), self.robot.num_joints), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * rand_dof

        noise_coeff = torch.zeros(self.robot.num_joints, device=self.device)
        for ji in self.arm_joint_indices:
            noise_coeff[ji] = self.cfg.reset_dof_pos_random_interval_arm
        for ji in self.hand_joint_indices:
            noise_coeff[ji] = self.cfg.reset_dof_pos_random_interval_fingers

        dof_pos = default_dof + noise_coeff * rand_delta
        rand_vel = sample_uniform(-1.0, 1.0, dof_pos.shape, device=self.device)
        dof_vel = self.cfg.reset_dof_vel_random_interval * rand_vel

        self.prev_targets[env_ids_t] = dof_pos
        self.cur_targets[env_ids_t] = dof_pos

        self.robot.set_joint_position_target(dof_pos, env_ids=env_ids_t)
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids_t)

        # reset tracking buffers
        self.prev_episode_successes[env_ids_t] = self.successes[env_ids_t]
        self.successes[env_ids_t] = 0
        self.true_objective[env_ids_t] = 0
        self.lifted_object[env_ids_t] = False
        self.closest_keypoint_max_dist[env_ids_t] = -1
        self.closest_fingertip_dist[env_ids_t] = -1
        self.furthest_hand_dist[env_ids_t] = -1
        self.near_goal_steps[env_ids_t] = 0
        self.attempt_timer[env_ids_t] = 0
        self.closest_hand_obj_d[env_ids_t] = -1.0
        self.closest_d_hat[env_ids_t] = -1.0
        self.prev_is_success[env_ids_t] = False
        self.reset_goal_buf[env_ids_t] = False
        for k in self.rewards_episode:
            self.rewards_episode[k][env_ids_t] = 0.0

        self._compute_intermediate_values()

    def _do_reset_target(self, env_ids: Tensor) -> None:
        """Reset target pose + clear attempt-level buffers."""
        self._reset_target(env_ids)
        self.reset_goal_buf[env_ids] = False
        self.near_goal_steps[env_ids] = 0
        self.closest_keypoint_max_dist[env_ids] = -1
        self.attempt_timer[env_ids] = 0
        self.closest_hand_obj_d[env_ids] = -1.0
        self.closest_d_hat[env_ids] = -1.0
        self.prev_is_success[env_ids] = False

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def _compute_full_state_obs(self) -> Tensor:
        """Build the full_state observation vector matching IsaacGymEnvs layout."""
        N = self.num_envs
        if self.cfg.paper_observation_mode:
            return self._compute_paper_obs()

        parts: list[Tensor] = []

        # 1) unscaled DOF positions
        parts.append(unscale(self.dof_pos, self.dof_lower.unsqueeze(0), self.dof_upper.unsqueeze(0)))
        # 2) DOF velocities
        parts.append(self.dof_vel)
        # 3) palm center position
        parts.append(self.palm_center_pos)
        # 4) palm rot + linvel + angvel  (10)
        parts.append(self.palm_rot)          # 4
        parts.append(self.palm_linvel)       # 3
        parts.append(self.palm_angvel)       # 3
        # 5) object rot + linvel + angvel  (10)
        parts.append(self.object_rot)        # 4
        parts.append(self.object_linvel)     # 3
        parts.append(self.object_angvel)     # 3
        # 6) fingertip pos relative to palm
        parts.append(self.fingertip_pos_rel_palm.reshape(N, -1))
        # 7) keypoints relative to palm
        parts.append(self.keypoints_rel_palm.reshape(N, -1))
        # 8) keypoints relative to goal
        parts.append(self.keypoints_rel_goal.reshape(N, -1))
        # 9) object scales
        parts.append(self.object_scales)
        # 10) closest keypoint max dist
        parts.append(self.closest_keypoint_max_dist.unsqueeze(-1))
        # 11) closest fingertip distances
        parts.append(self.closest_fingertip_dist)
        # 12) lifted object flag
        parts.append(self.lifted_object.float().unsqueeze(-1))
        # 13) log progress, log successes
        parts.append(torch.log(self.episode_length_buf.float() / 10.0 + 1.0).unsqueeze(-1))
        parts.append(torch.log(self.successes + 1.0).unsqueeze(-1))
        # 14) reward observation (current step reward * 0.01, matching original post_physics_step)
        parts.append(self._cached_reward.unsqueeze(-1) * 0.01)

        # 15) DexPBT extra obs: d_closest, d_hat
        if self.cfg.dexpbt_extra_observation_dims == 2:
            hd = self.curr_fingertip_distances.min(dim=-1).values
            dh = self._d_hat_current()
            d_closest_show = torch.where(
                self.closest_hand_obj_d < 0, hd, self.closest_hand_obj_d
            ).unsqueeze(-1)
            d_hat_show = torch.where(
                self.closest_d_hat < 0, dh, self.closest_d_hat
            ).unsqueeze(-1)
            parts.append(d_closest_show)
            parts.append(d_hat_show)

        return torch.cat(parts, dim=-1)

    def _compute_paper_obs(self) -> Tensor:
        """Observation layout following Table I in the paper (single-arm, 110D for reorientation)."""
        N = self.num_envs
        parts: list[Tensor] = []

        palm_euler = torch.stack(euler_xyz_from_quat(self.palm_rot), dim=-1)

        parts.append(unscale(self.dof_pos, self.dof_lower.unsqueeze(0), self.dof_upper.unsqueeze(0)))
        parts.append(self.dof_vel)
        parts.append(self.palm_center_pos)
        parts.append(palm_euler)
        parts.append(self.palm_linvel)
        parts.append(self.palm_angvel)
        parts.append(self.fingertip_pos_rel_palm.reshape(N, -1))
        parts.append(self.keypoints_rel_palm.reshape(N, -1))
        parts.append(self.keypoints_rel_goal.reshape(N, -1))
        parts.append(self.object_rot)
        parts.append(self.object_linvel)
        parts.append(self.object_angvel)
        parts.append(self.object_scales * self.cfg.object_base_size)
        parts.append(self.lifted_object.float().unsqueeze(-1))
        d_closest_show = torch.where(
            self.closest_hand_obj_d < 0, self.curr_fingertip_distances.min(dim=-1).values, self.closest_hand_obj_d
        ).unsqueeze(-1)
        d_hat_show = torch.where(
            self.closest_d_hat < 0, self._d_hat_current(), self.closest_d_hat
        ).unsqueeze(-1)
        parts.append(d_closest_show)
        parts.append(d_hat_show)
        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------ #
    # DexPBT reward
    # ------------------------------------------------------------------ #

    def _d_hat_current(self) -> Tensor:
        """Current task error."""
        if self.cfg.subtask in ("regrasping", "throw"):
            return torch.norm(self.object_pos - self.goal_pos, dim=-1)
        return self.keypoints_max_dist

    def _compute_dexpbt_reward(self) -> Tensor:
        """Staged DexPBT reward: reach -> pick -> target -> success."""
        hand_object_d = self.curr_fingertip_distances.min(dim=-1).values
        d_hat = self._d_hat_current()

        # lifting
        object_init_z = self.object_rest_heights
        z_lift = 0.05 + self.object_pos[:, 2] - object_init_z
        picked_prev = self.lifted_object.clone()
        lifted_now = (z_lift > self.cfg.lifting_bonus_threshold) | self.lifted_object
        just_picked = lifted_now & ~picked_prev

        # r_reach: before picked, reward for getting closer
        r_reach = (~picked_prev).float() * self.cfg.alpha_reach * torch.clamp(
            self.closest_hand_obj_d - hand_object_d, 0, 100
        )
        self.closest_hand_obj_d = torch.where(
            self.closest_hand_obj_d < 0, hand_object_d,
            torch.minimum(self.closest_hand_obj_d, hand_object_d)
        )

        # r_pick: height reward + bonus
        h_clip = torch.clamp(z_lift, 0, 0.5)
        r_pick = (~picked_prev).float() * self.cfg.alpha_pick * h_clip
        r_pick_bonus = self.cfg.r_picked * just_picked.float()

        # r_target: after picked, reward for reducing task error
        r_target = picked_prev.float() * self.cfg.alpha_targ * torch.clamp(
            self.closest_d_hat - d_hat, 0, 100
        )
        self.closest_d_hat = torch.where(
            self.closest_d_hat < 0, d_hat, torch.minimum(self.closest_d_hat, d_hat)
        )

        # success detection
        if self.cfg.subtask == "regrasping":
            near_goal = d_hat <= self.success_tolerance
        else:
            thresh = (
                self.success_tolerance * self.cfg.keypoint_scale
                if self.cfg.success_tolerance_use_keypoint_scale
                else self.success_tolerance
            )
            near_goal = d_hat <= thresh

        self.near_goal_steps += near_goal.long()
        is_success = self.near_goal_steps >= self.cfg.success_steps
        self._latest_is_success = is_success
        self.reset_goal_buf[:] = is_success
        self.successes += is_success.float()

        # success reward
        if self.cfg.success_reward_mode == "legacy":
            bonus_rew = near_goal.float() * (self.cfg.r_success / max(self.cfg.success_steps, 1))
            r_success_bonus = torch.zeros_like(bonus_rew)
        else:
            bonus_rew = torch.zeros(self.num_envs, device=self.device)
            edge_success = is_success & ~self.prev_is_success
            r_success_bonus = self.cfg.r_success * edge_success.float()

        # action velocity penalty (paper Eq.1 uses: +r_reach+r_pick+r_targ-r_vel)
        dof_vel = self.robot.data.joint_vel
        kuka_pen = self.cfg.kuka_actions_penalty_scale * torch.sum(
            torch.abs(dof_vel[:, self.arm_joint_indices]), dim=-1
        )
        allegro_pen = self.cfg.allegro_actions_penalty_scale * torch.sum(
            torch.abs(dof_vel[:, self.hand_joint_indices]), dim=-1
        )
        r_vel_penalty = kuka_pen + allegro_pen

        reward = r_reach + r_pick + r_pick_bonus + r_target + bonus_rew + r_success_bonus - r_vel_penalty

        # update state
        self.prev_is_success = is_success.clone()
        self.lifted_object = lifted_now

        # extras for logging
        self.true_objective = self._true_objective()
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["successes"] = self.prev_episode_successes.mean()
        self.extras["log"]["success_tolerance"] = self.success_tolerance
        self.extras["log"]["true_objective_mean"] = self.true_objective.mean()
        # Debug-only task-state metrics for failure localization.
        self.extras["log"]["debug_lifted_object_rate"] = lifted_now.float().mean()
        self.extras["log"]["debug_near_goal_rate"] = near_goal.float().mean()
        self.extras["log"]["debug_is_success_rate"] = is_success.float().mean()
        self.extras["log"]["debug_d_hat_mean"] = d_hat.mean()
        self.extras["log"]["debug_d_hat_min"] = d_hat.min()
        self.extras["log"]["debug_d_hat_closest_mean"] = self.closest_d_hat.mean()
        self.extras["log"]["debug_hand_object_d_mean"] = hand_object_d.mean()
        self.extras["log"]["debug_hand_object_d_min"] = hand_object_d.min()
        self.extras["log"]["debug_object_z_mean"] = self.object_pos[:, 2].mean()
        self.extras["log"]["debug_object_z_min"] = self.object_pos[:, 2].min()
        self.extras["log"]["debug_drop_rate"] = (self.object_pos[:, 2] < 0.1).float().mean()
        self.extras["log"]["debug_action_mean"] = self.actions.mean()
        self.extras["log"]["debug_action_std"] = self.actions.std()
        self.extras["log"]["debug_action_min"] = self.actions.min()
        self.extras["log"]["debug_action_max"] = self.actions.max()
        # Store a detached COPY so the value is stable after _reset_idx() zeroes
        # self.true_objective[done_env_ids] during episode termination. rl_games
        # process_infos() reads infos["true_objective"][done_idx] to compute r_meta_agg;
        # without .clone() the reference is already zero by the time rl_games reads it.
        self.extras["true_objective"] = self.true_objective.clone().detach()

        # WM data channel — feeds world model training via infos["wm"]
        # All quantities computed by _get_rewards() via _compute_intermediate_values()
        self.extras["wm"] = {
            "object_pos": self.object_pos.clone().detach(),
            "object_linvel": self.object_linvel.clone().detach(),
            "object_angvel": self.object_angvel.clone().detach(),
            "d_hat": self.keypoints_max_dist.unsqueeze(-1).clone().detach(),
            "failure": (self.object_pos[:, 2] < 0.1).float().unsqueeze(-1).clone().detach(),
        }

        # also store dexpbt extras
        self.extras["log"]["dexpbt_reward_total"] = reward.mean()
        self.extras["log"]["dexpbt_reward_reach"] = r_reach.mean()
        self.extras["log"]["dexpbt_reward_target"] = r_target.mean()
        self.extras["log"]["dexpbt_reward_success"] = r_success_bonus.mean()

        # accumulate episode rewards
        for name, t in [("reward_reach", r_reach), ("reward_pick", r_pick),
                        ("reward_pick_bonus", r_pick_bonus), ("reward_target", r_target),
                        ("reward_success_bonus", r_success_bonus), ("reward_vel_penalty", r_vel_penalty),
                        ("reward_total", reward)]:
            self.rewards_episode[name] += t

        return reward

    def _extra_curriculum(self):
        """Tolerance curriculum -- override in subclass if needed.

        Patch H (2026-05-31): also forwards `curriculum_freeze_frames_after_
        restart` so a freshly-restarted worker holds its inherited
        tolerance for a configurable warm-up window.
        """
        self.success_tolerance, self.last_curriculum_update = apply_env_tolerance_curriculum(
            curriculum_mode=str(getattr(self, "curriculum_mode", "legacy") or "legacy"),
            curriculum_control_path=getattr(self, "curriculum_control_path", None),
            curriculum_shrink_factor=float(
                getattr(self, "curriculum_shrink_factor", 0.95) or 0.95
            ),
            last_curriculum_update=self.last_curriculum_update,
            frames_since_restart=self.frame_since_restart,
            curriculum_interval=self.tolerance_curriculum_interval,
            prev_episode_successes=self.prev_episode_successes,
            success_tolerance=self.success_tolerance,
            initial_tolerance=self.initial_tolerance,
            target_tolerance=self.target_tolerance,
            curriculum_freeze_frames_after_restart=int(
                getattr(self, "curriculum_freeze_frames_after_restart", 0) or 0
            ),
            post_shrink_consolidation_env_frames=int(
                getattr(self, "post_shrink_consolidation_env_frames", 100_000_000)
                or 100_000_000
            ),
            num_envs=int(getattr(self, "num_envs", 1) or 1),
            pbt_single_shrink_per_window=bool(
                getattr(self, "pbt_single_shrink_per_window", True)
            ),
        )

    def _true_objective(self) -> Tensor:
        return tolerance_successes_objective(
            self.success_tolerance, self.initial_tolerance, self.target_tolerance, self.successes
        )

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _random_quat(self, n: int) -> Tensor:
        """Sample uniform random quaternion (wxyz convention used by Isaac Lab)."""
        uvw = sample_uniform(0.0, 1.0, (n, 3), device=self.device)
        w = torch.sqrt(1.0 - uvw[:, 0]) * torch.sin(2 * math.pi * uvw[:, 1])
        x = torch.sqrt(1.0 - uvw[:, 0]) * torch.cos(2 * math.pi * uvw[:, 1])
        y = torch.sqrt(uvw[:, 0]) * torch.sin(2 * math.pi * uvw[:, 2])
        z = torch.sqrt(uvw[:, 0]) * torch.cos(2 * math.pi * uvw[:, 2])
        return torch.stack([w, x, y, z], dim=-1)

    def _sample_target_pos(self, n: int) -> Tensor:
        """Sample positions within the target volume."""
        t_min = self.target_volume_origin + self.target_volume_extent[:, 0]
        t_max = self.target_volume_origin + self.target_volume_extent[:, 1]
        rand = sample_uniform(0.0, 1.0, (n, 3), device=self.device)
        return t_min + rand * (t_max - t_min)


# ====================================================================== #
# Regrasping
# ====================================================================== #


class DexPBTRegraspingEnv(DexPBTKukaAllegroEnv):
    """Regrasping: move object center to target position."""

    def _object_keypoint_offsets(self):
        return [[0, 0, 0]]

    def _reset_target(self, env_ids: Tensor) -> None:
        self.goal_pos[env_ids] = self._sample_target_pos(len(env_ids))
        self.goal_rot[env_ids, 0] = 1.0
        self.goal_rot[env_ids, 1:] = 0.0

        # also reset object to table (regrasping resets object on target change)
        obj_default = self.object.data.default_root_state[env_ids].clone()
        rand_pos = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        obj_default[:, 0] += self.cfg.reset_position_noise_x * rand_pos[:, 0]
        obj_default[:, 1] += self.cfg.reset_position_noise_y * rand_pos[:, 1]
        obj_default[:, 2] = self.object_rest_heights[env_ids] + self.cfg.reset_position_noise_z * rand_pos[:, 2]
        obj_default[:, 3:7] = self._random_quat(len(env_ids))
        obj_default[:, 7:] = 0.0
        obj_default[:, 0:3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(obj_default[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(obj_default[:, 7:13], env_ids=env_ids)

        self.lifted_object[env_ids] = False
        self.closest_fingertip_dist[env_ids] = -1
        self.furthest_hand_dist[env_ids] = -1

        # update goal marker
        goal_world = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_world, self.goal_rot)


# ====================================================================== #
# Reorientation
# ====================================================================== #


class DexPBTReorientationEnv(DexPBTKukaAllegroEnv):
    """Reorientation: match object orientation to target orientation."""

    def _object_keypoint_offsets(self):
        return [[1, 1, 1], [1, 1, -1], [-1, -1, 1], [-1, -1, -1]]

    def _reset_target(self, env_ids: Tensor) -> None:
        self.goal_pos[env_ids] = self._sample_target_pos(len(env_ids))
        self.goal_rot[env_ids] = self._random_quat(len(env_ids))

        # update goal marker
        goal_world = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_world, self.goal_rot)

    def _extra_reset_rules(self, terminated: Tensor) -> Tensor:
        # hand too far from object -> reset
        too_far = (self.curr_fingertip_distances.max(dim=-1).values > 1.5).float()
        return torch.max(terminated, too_far)


# ====================================================================== #
# Throw
# ====================================================================== #


class DexPBTThrowEnv(DexPBTKukaAllegroEnv):
    """Grasp-and-throw: toss the object into a randomized bucket."""

    def _object_keypoint_offsets(self):
        return [[0, 0, 0]]

    def _reset_target(self, env_ids: Tensor) -> None:
        left_right_random = sample_uniform(-1.0, 1.0, (len(env_ids), 1), device=self.device)
        x_pos = torch.where(
            left_right_random > 0,
            0.5 * torch.ones_like(left_right_random),
            -0.5 * torch.ones_like(left_right_random),
        )
        x_pos += torch.sign(left_right_random) * sample_uniform(0.0, 0.4, (len(env_ids), 1), device=self.device)
        y_pos = sample_uniform(-1.0, 0.7, (len(env_ids), 1), device=self.device)
        z_pos = sample_uniform(0.0, 1.0, (len(env_ids), 1), device=self.device)

        bucket_default = self.bucket.data.default_root_state[env_ids].clone()
        bucket_default[:, 0:1] = x_pos + self.scene.env_origins[env_ids, 0:1]
        bucket_default[:, 1:2] = y_pos + self.scene.env_origins[env_ids, 1:2]
        bucket_default[:, 2:3] = z_pos + self.scene.env_origins[env_ids, 2:3]
        bucket_default[:, 3] = 1.0
        bucket_default[:, 4:7] = 0.0
        bucket_default[:, 7:13] = 0.0
        self.bucket.write_root_pose_to_sim(bucket_default[:, :7], env_ids=env_ids)
        self.bucket.write_root_velocity_to_sim(bucket_default[:, 7:13], env_ids=env_ids)

        self.goal_pos[env_ids, 0:1] = x_pos
        self.goal_pos[env_ids, 1:2] = y_pos
        self.goal_pos[env_ids, 2:3] = z_pos + 0.05
        self.goal_rot[env_ids, 0] = 1.0
        self.goal_rot[env_ids, 1:] = 0.0

        obj_default = self.object.data.default_root_state[env_ids].clone()
        rand_pos = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        obj_default[:, 0] += self.cfg.reset_position_noise_x * rand_pos[:, 0]
        obj_default[:, 1] += self.cfg.reset_position_noise_y * rand_pos[:, 1]
        obj_default[:, 2] = self.object_rest_heights[env_ids] + self.cfg.reset_position_noise_z * rand_pos[:, 2]
        obj_default[:, 3:7] = self._random_quat(len(env_ids))
        obj_default[:, 7:] = 0.0
        obj_default[:, 0:3] += self.scene.env_origins[env_ids]
        self.object.write_root_pose_to_sim(obj_default[:, :7], env_ids=env_ids)
        self.object.write_root_velocity_to_sim(obj_default[:, 7:13], env_ids=env_ids)

        self.lifted_object[env_ids] = False
        self.closest_fingertip_dist[env_ids] = -1
        self.furthest_hand_dist[env_ids] = -1

        goal_world = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_world, self.goal_rot)
