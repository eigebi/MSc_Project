import numpy as np
import torch
from runner import Runner
from env import UAV_MEC
from Environment_marl_test import Environ


def make_train_env(args):
    env = UAV_MEC(args)
    return env

def make_eval_env(args):
    env = UAV_MEC(args)
    return env

def main(args):
    #seed
    torch.manual_seed(args.seed)
    #torch.mps.manual_seed(args.seed)
    np.random.seed(args.seed)


    up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
    down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
    left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
    right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

    width = 750/2
    height = 1298/2
    n_veh = 8
    n_neighbor = 1
    epsi_final = 0.02
    #env = make_train_env(args)
    #env_eval = make_eval_env(args)
    env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
    env.new_random_game()
    env.renew_positions()# no change
    env.renew_neighbor()# no change
    env.renew_channel()# slow fading, used to generate cluster for spectrum sharing
    env.renew_channels_fastfading()

    #env = make_train_env(args)
    args.env = env

    runner = Runner(args)
    runner.run()



class arguments:
    def __init__(self):
        self.device = torch.device("cpu")
        self.algorithm_name = "happo"

        self.num_V2I = 4
        self.num_V2V = 4
        self.num_agents = self.num_V2I + self.num_V2V

        self.num_env_steps = int(1e6)
  


        self.lr = 0.00001
        self.critic_lr = 0.0001
        self.opti_eps = 1e-5
        self.weight_decay = 0
        
        self.num_env_steps = 100000
        self.episode_length = int(100)#500
        self.hidden_size = 128

        self.seed = 2023

        # interval
        self.save_interval = 1000
        self.use_eval = 100 
        self.eval_interval = 10
        self.log_interval = 1
        

        #
        self.gain = 0.01
        self.use_orthogonal = True
        self.use_policy_active_masks = False
        self.use_naive_recurrent_policy = False
        self.recurrent_N = 1
        self.use_recurrent_policy = False

        #
        self.use_feature_normalization = True
        self.use_ReLU = True
        self.stacked_frames = 1
        self.layer_N = 2

        #
        self.std_x_coef = 1
        self.std_y_coef = 0.5

        self.clip_param = 0.02
        self.ppo_epoch = 5
        self.num_mini_batch = 1
        self.data_chunk_length = 10
        self.value_loss_coef = 1
        self.entropy_coef = 0.01
        self.max_grad_norm = 10    
        self.huber_delta = 10

        self.use_max_grad_norm = True
        self.use_clipped_value_loss = True
        self.use_huber_loss = True
        self.use_popart = True
        self.use_value_active_masks = True
        self.use_policy_active_masks = True

        self.n_rollout_threads = 1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.use_gae = True
        self.use_valuenorm = True
        self. use_proper_time_limits = False

if __name__ == "__main__":
    args = arguments()
    main(args)

