
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgAlg

class LittledogCPGRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        # num_envs = 4096
        num_envs = 4
        num_observations = 243
        num_actions = 8

    
    class terrain( LeggedRobotCfg.terrain):
        pass

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'leg1_hip_joint': 0.0,
            'leg1_leg_joint': 0.72,
            'leg1_foot_joint': -1.49543,
            'leg2_hip_joint': 0.0,
            'leg2_leg_joint': 0.72,
            'leg2_foot_joint': -1.49543,
            'leg3_hip_joint': 0.0,
            'leg3_leg_joint': 0.72,
            'leg3_foot_joint': -1.49543,
            'leg4_hip_joint': 0.0,
            'leg4_leg_joint': 0.72,
            'leg4_foot_joint': -1.49543,
            'leg5_hip_joint': 0.0,
            'leg5_leg_joint': 0.72,
            'leg5_foot_joint': -1.49543,
            'leg6_hip_joint': 0.0,
            'leg6_leg_joint': 0.72,
            'leg6_foot_joint': -1.49543
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        stiffness = {'joint': 85}  # [N*m/rad] 85
        damping = {'joint': 2}  # [N*m*s/rad]     # [N*m*s/rad] 2
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 5
        
    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/littledog/littledog.urdf'
        name = "littledog"
        foot_name = 'foot'
        penalize_contacts_on = ['base', '_leg', 'hip']
        terminate_after_contacts_on = ['base']
        flip_visual_attachments = False
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
  
    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 100. # forces above this value are penalized
        
        soft_dof_pos_limit = 0.9
        base_height_target = 0.42#0.5# TODO
        
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.00001
            dof_pos_limits = -50.0
            
            termination = -0.0
            tracking_lin_vel = 4.0
            tracking_ang_vel = 1.0
            # tracking_lin_vel = 0# 5.0
            # tracking_ang_vel = 0# 2.0
            lin_vel_z = 0# -1.0
            ang_vel_xy = 0# -0.05
            orientation = 0# -0.
            hip_rotate = 0# -0.5
            dof_vel = 0# -0.001
            dof_acc = 0# -2.5e-7
            base_height = 0# -0.1
            feet_air_time = 0# 1.0
            collision = -1.
            feet_stumble = 0# -0.0 
            action_rate = 0# -0.01
            stand_still = 0# -0.01

class LittledogCPGRoughCfgPPO( LeggedRobotCfgAlg ):
    
    class runner( LeggedRobotCfgAlg.runner ):
        run_name = ''
        experiment_name = 'rough_littledog_CPG'

    class algorithm( LeggedRobotCfgAlg.algorithm):
        entropy_coef = 0.01



  