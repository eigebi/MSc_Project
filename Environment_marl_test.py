from __future__ import division
import numpy as np
import time
import random
import math
from gym.spaces import Box


np.random.seed(1234)


class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001 #euclidean distance
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db


class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 4, 1299 / 4]  # center of the grids
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:
    # Vehicle simulator: include all the information for a vehicle

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_neighbor):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []

        self.demand = []
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []

        self.V2I_power_dB = 23  # dBm
        self.V2V_power_dB_List = [23, 15, 5, -100]  # the power levels


        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)

        #self.n_RB = n_veh
        self.n_Veh = n_veh



        self.n_V2IRB = 4
        self.n_V2VRB = 2
        # Iveh: cars only in V2I occupied RBs
        # Vveh: cars can do mode selection
        self.n_Iveh = 4
        self.n_Vveh = 4
        self.n_Veh = self.n_Iveh+self.n_Vveh
        self.num_agents = self.n_Veh
        self.obs_space = [[204] for _ in range(self.num_agents)]
        self.obs_space = [Box(low=-1,high=1,shape=(204,)) for _ in range(self.num_agents)]
        self.share_obs_space = self.obs_space

        self.action_space = []
        for i in range(self.n_Iveh):
            self.action_space.append(Box(low = 0,   high = 0.2))
        for i in range(self.n_Vveh):
            self.action_space.append(Box(low = -0.2, high = 0.2))
            

        self.n_neighbor = n_neighbor
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = int(1e6)  # bandwidth per RB, 1 MHz
        # self.bandwidth = 1500
        #self.demand_size = int((4 * 190 + 300) * 8 * 2)  # V2V payload: 1060 Bytes every 100 ms
        # self.demand_size = 20

        #self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2
        self.V2V_Interference_all = np.zeros((self.n_Vveh, self.n_neighbor, self.n_V2VRB)) + self.sig2
        self.V2I_Interference_all = np.zeros((self.n_Veh,self.n_V2IRB)) + self.sig2
        # V2I atmost n_Vveh+n_Iveh x n_V2IRB

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def share_obs_space(self, agent):
        return self.obs_space[0]

    def reset(self):
        self.new_random_game()
        self.renew_positions()# no change
        self.renew_neighbor()# no change
        self.renew_channel()# slow fading, used to generate cluster for spectrum sharing
        self.renew_channels_fastfading()
        state = get_state(self)
        state = np.concatenate((state,np.ones(4)))
        share_state = np.repeat(state[np.newaxis],self.num_agents,axis=0)[np.newaxis,:,:]
        obs = share_state[0]
        return obs, share_state
    
    def Compute_Interference(self, actions):
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_V2VRB)) + self.sig2

        channel_selection = actions.copy()[:, :, 0]
        power_selection = actions.copy()[:, :, 1]
        channel_selection[np.logical_not(self.active_links)] = -1

        # interference from V2I links
        for i in range(self.n_RB):
            for k in range(len(self.vehicles)):
                for m in range(len(channel_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer V2V links
        for i in range(len(self.vehicles)):
            for j in range(len(channel_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(channel_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or channel_selection[i, j] < 0:
                            continue
                        V2V_Interference[k, m, channel_selection[i, j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]]
                                                                                   - self.V2V_channels_with_fastfading[i][self.vehicles[k].destinations[m]][channel_selection[i,j]] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)






    def add_new_vehicles_by_number(self, n):
        # 一次加4个
        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'd' # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))
            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity*self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============
        # maintain the number of vehicles, and they turn clock-wise when meet the border
        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def renew_neighbor(self):
        """ Determine the neighbors of each vehicles """

        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)
        #using complex number to simplify the computation of distance

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(self.n_neighbor):
                self.vehicles[i].neighbors.append(sort_idx[j + 1])
            destination = self.vehicles[i].neighbors
            # V2V links are built among neighbour vehicles
            self.vehicles[i].destinations = destination

    def renew_channel(self):
        """ Renew slow fading channel """

        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
        self.V2I_pathloss = np.zeros((len(self.vehicles)))

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j], self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j,i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing
        return self.V2V_channels_abs, self.V2I_channels_abs

    def renew_channels_fastfading(self):
        """ Renew fast fading channel """

        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_V2VRB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_V2IRB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape))/ math.sqrt(2))

    def old_Compute_Performance_Reward_Train(self, actions_power):

        actions = actions_power[:, :, 0]  # the channel_selection_part
        power_selection = actions_power[:, :, 1]  # power selection

        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links[i, j]:
                    continue
                V2I_Interference[actions[i][j]] += 10 ** ((self.V2V_power_dB_List[power_selection[i, j]] - self.V2I_channels_with_fastfading[i, actions[i, j]]
                                                           + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        actions[(np.logical_not(self.active_links))] = -1 # inactive links will not transmit regardless of selected power levels
        for i in range(self.n_RB):  # scanning all bands
            indexes = np.argwhere(actions == i)  # find spectrum-sharing V2Vs
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                   - self.V2V_channels_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #  V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[k, 0], indexes[k, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((self.V2V_power_dB_List[power_selection[indexes[j, 0], indexes[j, 1]]]
                                                                              - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))

        self.demand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand[self.demand < 0] = 0 # eliminate negative demands

        self.individual_time_limit -= self.time_fast

        reward_elements = V2V_Rate/10
        reward_elements[self.demand <= 0] = 1

        self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0 # transmission finished, turned to "inactive"

        return V2I_Rate, V2V_Rate, reward_elements

    def Compute_Performance_Reward_Train(self, actions, channels):


        #modes = np.array(actions>=0, dtype=np.int8)

        modes = np.zeros(channels.shape, dtype=np.int8)
        modes[np.where(channels<self.n_Iveh)]=1
        modes = modes.reshape(-1,1)
        power_selection = actions
        self.V2V_Interference_all = np.zeros((self.n_Vveh, self.n_neighbor, self.n_V2VRB)) + self.sig2
        self.V2I_Interference_all = np.zeros((self.n_Veh,self.n_V2IRB)) + self.sig2
        # ------------ Compute V2I rate --------------------
        V2I_Rate = np.zeros(self.n_Vveh+self.n_Iveh)
        V2I_Interference = np.zeros(self.n_Vveh+self.n_Iveh)+ self.sig2   # V2I interference

 

        for i in range(len(self.vehicles)):
            if not modes[i,0]==1:
                continue
            for k in range(self.n_V2IRB):
                ids = np.argwhere(channels==k)
                for j in range(len(ids)):
                    if ids[j]==i: #排除当前车
                        continue
                    V2I_Interference[i] += 10 ** ((power_selection[ids[j],0] - self.V2I_channels_with_fastfading[ids[j], k] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                    #V2I_Interference[i] += 10 ** ((self.V2I_power_dB - self.V2I_channels_with_fastfading[i, k] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        
        self.V2I_Interference_all[np.where(channels<self.n_V2IRB),channels[np.where(channels<self.n_V2IRB)]] = V2I_Interference[np.where(channels<self.n_V2IRB)]
        self.V2I_Interference_all = 10 * np.log10(self.V2I_Interference_all)
        self.V2I_Interference = V2I_Interference 
        V2I_Signals = np.zeros(self.n_Veh)
        id_temp = np.concatenate((np.argwhere(channels<self.n_V2IRB),channels[np.argwhere(channels<self.n_V2IRB)]),axis=1)
        V2I_Rate = np.zeros(self.n_Veh)
        if id_temp.shape[0]>0:
            V2I_Signals[np.where(modes[:,0]==1)] = 10 ** ((power_selection[np.where(modes[:,0]==1)][:,0] - self.V2I_channels_with_fastfading[id_temp[:,0],id_temp[:,1]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
            V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))

        

        # ------------ Compute V2V rate -------------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))+ self.sig2
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        #actions[(np.logical_not(self.active_links))] = -1 # inactive links will not transmit regardless of selected power levels
        for i in range(self.n_V2VRB):  # scanning all bands
            ids = np.argwhere(channels == i+self.n_V2IRB)  # find spectrum-sharing V2Vs
            for j in range(len(ids)):
                receiver_j = self.vehicles[ids[j,0]].destinations[0]
                V2V_Signal[ids[j,0], 0] = 10 ** ((power_selection[ids[j,0]]
                                                                   - self.V2V_channels_with_fastfading[ids[j, 0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                # V2I links interference to V2V links

                #  V2V interference
                for k in range(j + 1, len(ids)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[ids[k][0]].destinations[0]
                    V2V_Interference[ids[j][0], 0] += 10 ** ((power_selection[ids[k][0]]
                                                                              - self.V2V_channels_with_fastfading[ids[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[ids[k][0], 0] += 10 ** ((power_selection[ids[j][0]]
                                                                              - self.V2V_channels_with_fastfading[ids[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    
        self.V2V_Interference_all[np.where(channels>=self.n_V2IRB)[0]-self.n_V2IRB, 0, channels[np.where(channels>=self.n_V2IRB)]-self.n_V2IRB] = V2V_Interference[np.where(channels>=self.n_V2IRB),0]
        self.V2V_Interference_all = 10 * np.log10(self.V2V_Interference_all)
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))



        return V2I_Rate, V2V_Rate, 0,modes

    def act_for_training(self, actions,channels,budget=None):


        action_temp = actions.copy()
        V2I_Rate, V2V_Rate, reward_elements, modes = self.Compute_Performance_Reward_Train(action_temp, channels)

       
        reward = V2I_Rate.reshape(1,-1) + V2V_Rate.reshape(1,-1)
        beta = np.log2(10**(2/10))*0.1#5,3,1

        temp = reward[0,self.n_Iveh:]
        if budget is not None:
            temp = np.array(temp<beta,dtype=np.int8)
            budget -= temp
            if np.min(budget)<0:
                reward = -np.ones((1,8))*0.01
            else:
                reward = np.repeat(np.sum(reward[:,:4]),8).reshape(1,-1)
        else:
            temp[np.where(temp>=beta)]=beta
            reward[0,self.n_Iveh:] = temp
            reward = np.repeat(np.sum(reward),8).reshape(1,-1)
        return reward, V2I_Rate+V2V_Rate[:,0],modes, budget

    def new_random_game(self, n_Veh=0):
        # make a new game

        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))
        self.renew_neighbor()
        self.renew_channel()
        self.renew_channels_fastfading()

        #self.demand = self.demand_size * np.ones((self.n_Veh, self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh, self.n_neighbor))
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')

    def step(self,action,budget):
        '''
        action = action[0] * 23
        action[np.where(action[:self.n_Iveh,:]<0)]=0
        action[np.where(action>23)]=23
        action[np.where(action< -23)]=-23
        '''
        action = action[0]
        action[:self.n_Iveh,:] = (action[:self.n_Iveh,:] + 1) * 0.1
        action[self.n_Iveh:,:] = action[self.n_Iveh:,:] * 0.2
        action[np.where(action[:self.n_Iveh,:]<0)]=0.00001
        action[np.where(action>0.2)]=0.2
        action[np.where(action< -0.2)]=-0.2

        channels = np.array([0,1,2,3,0,0,0,0],dtype=np.int8)
        inf_I = np.zeros((self.n_V2IRB))
        inf_V = [[]for _ in range(self.n_V2VRB)]
        inf_I = self.V2I_channels_abs[:self.n_Iveh]
        for i in range(self.n_Iveh,self.n_Veh):
            if action[i]>=0:
                temp = inf_I + self.V2I_channels_abs[i]
                channels[i]=np.argmin(temp)
                inf_I[channels[i]] += self.V2I_channels_abs[i]
            if action[i]<0:
                if [] in inf_V:
                    channels[i] = inf_V.index([]) + self.n_V2IRB
                    inf_V[inf_V.index([])].append(i)
                    
                else:
                    temp = np.zeros((self.n_V2VRB))
                    for RB, c in enumerate(inf_V):
                        for k in c:
                            temp[RB] += self.V2V_channels_abs[k,i]
                            temp[RB] += self.V2V_channels_abs[i,k]
                    channels[i] = np.argmin(temp) + self.n_V2IRB
        
 

        action_dbm = 10 * np.log10(np.abs(action)/0.001)
        #action_dbm[:self.n_Iveh]=23
        
        reward,rate ,modes,b= self.act_for_training(action_dbm,channels,budget)
        state = get_state(self)
        if budget is not None:
            state = np.concatenate((state,budget))
        share_state = np.repeat(state[np.newaxis],self.num_agents,axis=0)[np.newaxis,:,:]
        obs = share_state
        self.renew_channel()
        self.renew_channels_fastfading()
        
        return obs, share_state, reward, 0,rate,modes,b
    def slow_update(self):
        self.renew_positions()# no change
        self.renew_neighbor()# no change
        self.renew_channel()# slow fading, used to generate cluster for spectrum sharing
        self.renew_channels_fastfading()

# define by practical scenarios
def old_get_state(env, idx=(0,0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    #load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    #time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    
    

def get_state(env):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading - env.V2I_channels_abs.reshape(-1,1) + 10)/35
    V2I_channel = (env.V2I_channels_with_fastfading - 80) / 60
    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading - env.V2V_channels_abs[:,:,np.newaxis]+ 10)/35
    V2V_channel = (env.V2V_channels_with_fastfading - 80) / 60
    V2V_interference = (-env.V2V_Interference_all - 60) / 60
    V2I_interference = (-env.V2I_Interference_all - 60) / 60

    V2I_abs = (env.V2I_channels_abs - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs - 80)/60.0

    #load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    #time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    # return np.concatenate((np.reshape(V2V_channel, -1), V2V_interference, V2I_abs, V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))
    #return np.concatenate((np.reshape(V2I_fast,-1), np.reshape(V2V_fast, -1), np.reshape(V2V_interference,-1), np.reshape(V2I_abs,-1), np.reshape(V2V_abs,-1)))
    return np.concatenate((np.reshape(V2I_interference,-1), np.reshape(V2V_interference, -1), np.reshape(V2I_channel,-1), np.reshape(V2V_channel,-1)))



if __name__ == "__main__":
    up_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
    down_lanes = [i/2.0 for i in [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]]
    left_lanes = [i/2.0 for i in [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
    right_lanes = [i/2.0 for i in [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]]

    width = 750/2
    height = 1298/2
    n_veh = 8
    n_neighbor = 1
    epsi_final = 0.02

    env = Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
    env.new_random_game()
    env.renew_positions()# no change
    env.renew_neighbor()# no change
    env.renew_channel()# slow fading, used to generate cluster for spectrum sharing
    env.renew_channels_fastfading()

    #env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))#不需要 删掉
    env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
    env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

    #state_old = get_state(env, [0, 0], 1, epsi_final)
    #mode power>0: cellular ; power<0: dedicated 
    #action_test = np.array([10,10,10,10,10,10,-10,-10],dtype=np.int16).reshape(-1,1)
    #channels = np.array([0,1,2,3,0,1,4,5],dtype=np.int8)
    action_test = -np.array([10,10,10,10,10,10,10,-10],dtype=np.int16).reshape(-1,1)
    channels = np.array([4,4,4,4,5,5,5,1],dtype=np.int8)
    env.act_for_training(action_test,channels)
    env.renew_channels_fastfading()
    #env.Compute_Interference(action_test)
    get_state(env)

    print(1)