import torch
import numpy as np
import time
from happo_policy import HAPPO_Policy as Policy
from happo_trainer import HAPPO as Trainer
from separated_buffer import SeparatedReplayBuffer

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, args):
        self.num_agents = args.num_agents
        self.envs = args.env
        self.device = args.device
        self.num_env_steps = args.num_env_steps
        self.episode_length = args.episode_length
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size
        self.use_centralized_V = True

        #注意这里space 维度
        self.policy = []
        for agent in range(self.num_agents):
            share_obs_space = self.envs.share_obs_space[agent]
            po = Policy(args, self.envs.obs_space[agent], share_obs_space, self.envs.action_space[agent], device = self.device)
            self.policy.append(po)


        self.trainer = []
        self.buffer = []
        for agent in range(self.num_agents):
            tr = Trainer(args, self.policy[agent], device = self.device)
            self.trainer.append(tr)
            bu = SeparatedReplayBuffer(args, self.envs.obs_space[agent], share_obs_space, self.envs.action_space[agent])
            self.buffer.append(bu)

    def train(self):
        train_infos = []
        action_dim = self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length,1, 1), dtype=np.float32)

        for agent in range(self.num_agents):
            self.trainer[agent].prep_training()
            self.buffer[agent].update_factor(factor)
            #what is available actions?
            available_actions = None if self.buffer[agent].available_actions is None \
                else self.buffer[agent].available_actions[:-1].reshape(-1, *self.buffer[agent].available_actions.shape[2:])
            
            #to fill
            old_actions_logprob, _ = self.trainer[agent].policy.actor.evaluate_actions(self.buffer[agent].obs[:-1].reshape(-1, *self.buffer[agent].obs.shape[2:]),
                                                            self.buffer[agent].rnn_states[0:1].reshape(-1, *self.buffer[agent].rnn_states.shape[2:]),
                                                            self.buffer[agent].actions.reshape(-1, *self.buffer[agent].actions.shape[2:]),
                                                            self.buffer[agent].masks[:-1].reshape(-1, *self.buffer[agent].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent].active_masks[:-1].reshape(-1, *self.buffer[agent].active_masks.shape[2:]))
            
            train_info = self.trainer[agent].train(self.buffer[agent])


            new_actions_logprob, _ = self.trainer[agent].policy.actor.evaluate_actions(self.buffer[agent].obs[:-1].reshape(-1, *self.buffer[agent].obs.shape[2:]),
                                                            self.buffer[agent].rnn_states[0:1].reshape(-1, *self.buffer[agent].rnn_states.shape[2:]),
                                                            self.buffer[agent].actions.reshape(-1, *self.buffer[agent].actions.shape[2:]),
                                                            self.buffer[agent].masks[:-1].reshape(-1, *self.buffer[agent].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent].active_masks[:-1].reshape(-1, *self.buffer[agent].active_masks.shape[2:]))
            #to edit
            factor = factor * _t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,1,1))
            train_infos.append(train_info)
            self.buffer[agent].after_update()
        return train_infos
    
    def run(self):
        self. warm_up()
        info = []
        modes_step = []
        reward_episode = []
        rate_step = []
        start = time.time()
        # def of episodes
        episodes = int(self.num_env_steps) // self.episode_length
        train_episode_reward = 0
        for episode in range(episodes):
            #if episode % 100 == 0:
            #    self.envs.slow_update()
            done_episodes_reward = []
            train_episode_reward = 0
            p_0=0.8
            b=np.ones((4)) # safe budget
            b=b*(1-p_0)*100
            for step in range(self.episode_length):
               
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_state, rewards, dones ,rate,modes,budget = self.envs.step(actions,b)
                b = budget
                rate_step.append(rate)
                modes_step.append(modes)
                #if step % 100 == 0:
                    #self.envs.slow_update()
                # edit dim
                #dones_env = np.all(dones, axis=1)
                dones_env = np.zeros((1),dtype=np.int8) #没有固定的终点
                reward_env = np.mean(rewards)
                train_episode_reward += reward_env
                if dones_env:
                    done_episodes_reward.append(train_episode_reward)
                    train_episode_reward = 0
                
                data = obs, share_state, rewards, dones, [], values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)
            reward_episode.append(train_episode_reward)
            self.compute()
            train_infos = self.train()
            info.append(train_infos[0]['value_loss'])
            #info[1].append(train_infos[0]['policy_loss'])
            #info[2].append(train_infos[-1]['policy_loss'])
            print(train_infos[0]['value_loss'],train_episode_reward)
        text = 'v=0_2db_CAMA_r'
        np.save('info'+text+'.npy', np.array(info))
        np.save('reward'+text+'.npy', np.array(reward_episode))
        np.save('rate'+text+'.npy', np.array(rate_step))
        np.save('modes'+text+'.npy', np.array(modes_step))

    def warm_up(self):
        obs, share_state = self.envs.reset()
        for agent in range(self.num_agents):
            self.buffer[agent].share_obs[0] = share_state[:, agent].copy()
            self.buffer[agent].obs[0] = obs[agent].copy()



    @torch.no_grad()
    def compute(self):
        for agent in range(self.num_agents):
            self.trainer[agent].prep_rollout()
            next_value = self.trainer[agent].policy

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=0)

        #rnn_states[dones_env == True] = np.zeros(
        #    ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        #rnn_states_critic[dones_env == True] = np.zeros(
        #    ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        #masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
        #active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        #active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], [],
                                         [], actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None)

    def log_train(self, train_infos, total_num_steps):
        print("average_step_rewards is {}.".format(np.mean(self.buffer[0].rewards)))
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
