"""Environment configs for DexPBT Kuka-Allegro tasks on Isaac Lab."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


# ---------------------------------------------------------------------------
# Robot: Kuka iiwa7 + Allegro Hand
# Override PD gains from KUKA_ALLEGRO_CFG to match IsaacGymEnvs defaults
# (stiffness=40, damping=5 for both arm and hand; effort 300/0.35)
# ---------------------------------------------------------------------------

DEXPBT_KUKA_ALLEGRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa7_joint_1": 0.0,
            "iiwa7_joint_2": 0.0,
            "iiwa7_joint_3": 0.7854,
            "iiwa7_joint_4": 1.5708,
            "iiwa7_joint_5": -1.5708,
            "iiwa7_joint_6": -1.5708,
            "iiwa7_joint_7": 0.0,
            "(index|middle|ring)_joint_0": 0.0,
            "(index|middle|ring)_joint_1": 0.3,
            "(index|middle|ring)_joint_2": 0.3,
            "(index|middle|ring)_joint_3": 0.3,
            "thumb_joint_0": 1.5,
            "thumb_joint_1": 0.60,
            "thumb_joint_2": 0.34,
            "thumb_joint_3": 0.61,
        },
    ),
    actuators={
        "kuka_allegro_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "iiwa7_joint_(1|2|3|4|5|6|7)",
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit_sim={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 300.0,
                "index_joint_(0|1|2|3)": 0.35,
                "middle_joint_(0|1|2|3)": 0.35,
                "ring_joint_(0|1|2|3)": 0.35,
                "thumb_joint_(0|1|2|3)": 0.35,
            },
            stiffness={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 40.0,
                "index_joint_(0|1|2|3)": 40.0,
                "middle_joint_(0|1|2|3)": 40.0,
                "ring_joint_(0|1|2|3)": 40.0,
                "thumb_joint_(0|1|2|3)": 40.0,
            },
            damping={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 5.0,
                "index_joint_(0|1|2|3)": 5.0,
                "middle_joint_(0|1|2|3)": 5.0,
                "ring_joint_(0|1|2|3)": 5.0,
                "thumb_joint_(0|1|2|3)": 5.0,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------

NUM_ARM_DOFS = 7
NUM_HAND_DOFS = 16
NUM_TOTAL_DOFS = NUM_ARM_DOFS + NUM_HAND_DOFS  # 23
NUM_FINGERTIPS = 4
PAPER_SINGLE_ARM_ENV_STEPS = 5_000_000_000
PAPER_ROLLOUT_LENGTH = 16
PAPER_NUM_ENVS = 8192
PAPER_SINGLE_ARM_MAX_EPOCHS = math.ceil(PAPER_SINGLE_ARM_ENV_STEPS / (PAPER_NUM_ENVS * PAPER_ROLLOUT_LENGTH))


def _obs_size(num_keypoints: int, dexpbt_extra: int = 2) -> int:
    """Compute observation vector length matching IsaacGymEnvs full_state layout."""
    return (
        NUM_TOTAL_DOFS          # dof pos (unscaled)
        + NUM_TOTAL_DOFS        # dof vel
        + 3                     # palm center pos
        + 10                    # palm rot(4) + linvel(3) + angvel(3)
        + 10                    # object rot(4) + linvel(3) + angvel(3)
        + 3 * NUM_FINGERTIPS   # fingertip pos relative to palm
        + 3 * num_keypoints    # keypoints relative to palm
        + 3 * num_keypoints    # keypoints relative to goal
        + 3                     # object scales
        + 1                     # closest keypoint max dist
        + NUM_FINGERTIPS       # closest fingertip distances
        + 1                     # lifted object flag
        + 2                     # log(progress), log(successes)
        + 1                     # reward obs
        + dexpbt_extra          # d_closest, d_hat (DexPBT extras)
    )


def _paper_obs_size(num_keypoints: int) -> int:
    """Observation size from Table I in the paper for single-arm tasks."""
    return (
        NUM_TOTAL_DOFS      # joint angles
        + NUM_TOTAL_DOFS    # joint velocities
        + 3                 # hand position
        + 3                 # hand rotation (Euler XYZ)
        + 3                 # hand velocity
        + 3                 # hand angular velocity
        + 3 * NUM_FINGERTIPS
        + 3 * num_keypoints
        + 3 * num_keypoints
        + 4                 # object rotation (quaternion)
        + 3                 # object velocity
        + 3                 # object angular velocity
        + 3                 # object dimensions
        + 1                 # 1picked
        + 1                 # dclosest
        + 1                 # d_hat_closest
    )


@configclass
class DexPBTKukaAllegroEnvCfg(DirectRLEnvCfg):
    """Base config for DexPBT Kuka+Allegro single-arm tasks."""

    # --- simulation ---
    decimation = 1  # control_freq_inv = 1 in IsaacGymEnvs
    episode_length_s = 10.0  # 600 steps * (1/60) = 10s
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
        render_interval=1,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=8388608,
            # 2^18=262144: must be >= max needed patch count across all 8192 parallel envs.
            # Error "Patch buffer overflow detected, please increase to at least ~166000" was
            # observed early in training with default 5*2^15=163840.  We use 2^18 to give
            # headroom for dynamic scenes with high contact patch counts during aggressive
            # regrasping / throw / reorientation motions.
            gpu_max_rigid_patch_count=262144,
        ),
    )

    # --- scene ---
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=1.2,
        replicate_physics=True,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.5, 1.5, 1.2),
        lookat=(0.0, 0.0, 0.4),
        resolution=(1280, 720),
    )

    # --- assets ---
    robot_cfg: ArticulationCfg = DEXPBT_KUKA_ALLEGRO_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.55, 0.1, 0.35),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    bucket_cfg: RigidObjectCfg | None = None

    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            )
        },
    )

    # --- body names ---
    palm_body_name: str = "iiwa7_link_7"
    fingertip_body_names: list = [
        "index_link_3",
        "middle_link_3",
        "ring_link_3",
        "thumb_link_3",
    ]
    arm_joint_names: list = [
        "iiwa7_joint_1", "iiwa7_joint_2", "iiwa7_joint_3", "iiwa7_joint_4",
        "iiwa7_joint_5", "iiwa7_joint_6", "iiwa7_joint_7",
    ]
    hand_joint_names: list = [
        "index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3",
        "middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3",
        "ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3",
        "thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3",
    ]

    # Offsets from rigid body origin to fingertip contact point (meters, in body frame)
    fingertip_offsets: list = [
        [0.05, 0.005, 0.0],
        [0.05, 0.005, 0.0],
        [0.05, 0.005, 0.0],
        [0.06, 0.005, 0.0],
    ]
    palm_center_offset: list = [-0.00, -0.02, 0.16]

    # --- actions ---
    action_space = NUM_TOTAL_DOFS  # 23
    dof_speed_scale: float = 10.0
    act_moving_average: float = 1.0
    clamp_abs_observations: float = 10.0

    # --- DexPBT reward ---
    success_reward_mode: str = "paper"  # "paper" or "legacy"
    alpha_reach: float = 50.0
    alpha_pick: float = 20.0
    r_picked: float = 300.0
    alpha_targ: float = 200.0
    r_success: float = 1000.0
    kuka_actions_penalty_scale: float = 0.003
    allegro_actions_penalty_scale: float = 0.0003
    lifting_bonus_threshold: float = 0.15

    # --- success / curriculum ---
    success_tolerance: float = 0.075
    target_success_tolerance: float = 0.01
    tolerance_curriculum_interval: int = 3000
    max_consecutive_successes: int = 50
    success_steps: int = 1
    keypoint_scale: float = 1.5
    object_base_size: float = 0.05
    randomize_object_dimensions: bool = True
    with_small_cuboids: bool = True
    with_big_cuboids: bool = True
    with_sticks: bool = True
    success_tolerance_use_keypoint_scale: bool = False

    # --- DexPBT extras ---
    dexpbt_extra_observation_dims: int = 2
    attempt_timeout_steps: int = 600
    paper_observation_mode: bool = False
    paper_max_env_steps: int = PAPER_SINGLE_ARM_ENV_STEPS

    # --- reset noise ---
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    reset_rotation_noise: float = 1.0
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_vel_random_interval: float = 0.5

    # --- subtask (set by subclass) ---
    subtask: str = "reorientation"
    num_keypoints: int = 4

    # observation / state space (computed in __post_init__)
    observation_space: int = 0
    state_space: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.paper_observation_mode:
            obs = _paper_obs_size(self.num_keypoints)
        else:
            obs = _obs_size(self.num_keypoints, self.dexpbt_extra_observation_dims)
        self.observation_space = obs


@configclass
class DexPBTReorientationEnvCfg(DexPBTKukaAllegroEnvCfg):
    """Reorientation task config (default in AllegroKuka.yaml)."""
    subtask: str = "reorientation"
    num_keypoints: int = 4
    episode_length_s: float = 10.0  # 600 steps at dt=1/60
    success_steps: int = 1
    keypoint_scale: float = 1.0  # DexPBT overrides base 1.5 → 1.0
    paper_observation_mode: bool = True


@configclass
class DexPBTRegraspingEnvCfg(DexPBTKukaAllegroEnvCfg):
    """Regrasping task config."""
    subtask: str = "regrasping"
    num_keypoints: int = 1
    episode_length_s: float = 5.0  # 300 steps at dt=1/60
    success_steps: int = 30
    paper_observation_mode: bool = True


@configclass
class DexPBTThrowEnvCfg(DexPBTKukaAllegroEnvCfg):
    """Grasp-and-throw task config."""

    subtask: str = "throw"
    num_keypoints: int = 1
    episode_length_s: float = 5.0  # 300 steps at dt=1/60
    success_tolerance: float = 0.075
    target_success_tolerance: float = 0.075
    success_steps: int = 5
    attempt_timeout_steps: int = 300
    with_small_cuboids: bool = True
    with_big_cuboids: bool = False
    with_sticks: bool = False
    paper_observation_mode: bool = True
    bucket_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/bucket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
            scale=(1.1, 1.6, 3.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, -1.0, 0.45),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
