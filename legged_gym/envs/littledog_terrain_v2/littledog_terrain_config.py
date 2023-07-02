from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgAlg
from legged_gym.envs.base.base_config import BaseConfig
class LittledogTerrainCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env):
        num_envs = 2
        # num_envs = 10
        num_observations = 195
        num_privileged_obs = 108
        num_actions = 18 #! 6 * 3 

    
    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True # True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_foot_points_x = [-0.1,-0.05,0.05,0.1]
        measured_foot_points_y = [-0.1,-0.05,0.05,0.1]
        
        selected = False # False # select a unique terrain type and pass all arguments
        # terrain_kwargs = {'type':"random_uniform_terrain",
        #                   'min_height':-0.02,
        #                   'max_height':0.02,
        #                   'step':0.005} # None # Dict of arguments for selected terrain
        
        # terrain_kwargs = {'type':"wave_terrain",
        #                   'num_waves': 1,
        #                   'amplitude': 0.5} # None # Dict of arguments for selected terrain
        
        terrain_kwargs = {'type':"sloped_terrain",
                          'slope': 0.18} # None # Dict of arguments for selected terrain
        
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.4, 0.3, 0.3, 0, 0]# [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces


    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.50] # x,y,z [m]
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

    class control( LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'joint': 85.}  # [N*m/rad] 85
        damping = {'joint': 2}  # [N*m*s/rad]     # [N*m*s/rad] 2
        # action scale: target angle = actionScale * action + defaultAngle
        #! 我默认输出的范围是 [-1,1]
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        
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
            torques = -0.00002
            dof_pos_limits = -50.0
            
            termination = -0.0
            tracking_lin_vel = 4.0
            tracking_ang_vel = 0.5
            # tracking_lin_vel = 5.0
            # tracking_ang_vel = 2.0
            lin_vel_z = -4.0
            ang_vel_xy = -0.05
            orientation = -0.
            hip_rotate = -1
            dof_vel = -0.001
            dof_acc = -0.0005 #-2.5e-7
            base_height = -0.1
            feet_air_time = 1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.01
            stand_still = -0.01
        class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05
                height_measurements = 5.0
            clip_observations = 100.
            clip_actions = 1. # 1 for CPG, 100 for the other situation



class LittledogTerrainCfgPPO( BaseConfig ):
    seed = 1 
    runner_class_name = "StackedRunner"
    class runner:
        num_steps_per_env = 24 # per iteration
        max_iterations = 15 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'Terrain_Gait'
        run_name = 'Test'
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class ppo:
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=3e-4,
        max_grad_norm=10.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,

    
    class task_representation:
        batch_size = 4096 * 10,
        num_mini_batch = 10 ,
        num_learning_epochs=1,
        capacity = 100,
        max_history_len = 10,
        learning_rate = 3e-4,
        max_grad_norm = 1.0,

    class policy:
        emb_dim = 10
        feature_extractor_dims = [256,256],
        actor_hidden_dims=[256],
        critic_hidden_dims=[256],
        activation='elu',
        init_noise_std=1.0,

    class encoder:
        feature_hidden_dim = [256,256],
        mid_emb_dim = 16,
        max_step_len=10,
        activation = 'elu'
    
    class decoder:
        hidden_dim = [256,256],
        activation = 'elu'
