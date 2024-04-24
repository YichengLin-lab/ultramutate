import gym
import utils
from utils import *
from gym.spaces.box import Box
import math

class AompEnv(gym.Env):
    def __init__(self, env_name, multicore_num):
        super(AompEnv, self).__init__()
        # self.observation_space = Box(low=0, high=100, shape=(510, ), dtype=float)
        self.observation_space = (1159, )
        self.action_space = gym.spaces.Discrete(n=300)
        self.pep_ori = None
        self.hla_now = None
        self.pep_now = None
        self.gathered_vector = None
        self.is_bound = None
        self.binding_affinity = None
        self.env_name = env_name
        self.multicore_num = multicore_num
        self.round_pun = 1
        self.steps_taken = 0
        self.binding_affinity = None
    
    def reset(self):
        self.is_bound = True
        self.steps_taken = 0
        while self.is_bound:
            self.hla_now, self.pep_ori = utils.get_random_training_data()
            self.gathered_vector, self.is_bound, self.binding_affinity = utils.get_pep_hla_attention_vector(self.pep_ori, self.hla_now, self.env_name, self.multicore_num)

        self.pep_now = self.pep_ori
        
        return self.gathered_vector    

    def step(self, action):
        self.steps_taken += 1
        self.pep_now = utils.mutate_pep(self.pep_now, action)
        self.gathered_vector, self.is_bound, self.binding_affinity = utils.get_pep_hla_attention_vector(self.pep_now, self.hla_now, self.env_name, self.multicore_num)
        if self.binding_affinity < 0.000001:
            self.binding_affinity = 0.000001

        Rs = 10 if self.is_bound else 0
        reward = - self.round_pun + math.log(self.binding_affinity) + Rs

        homology = utils.get_homology(self.pep_ori, self.pep_now)

        if homology < 0.2: 
            is_done = True
        
        else:
            is_done = self.is_bound

        if self.steps_taken > 10:
            is_done = True

        return self.gathered_vector, reward, is_done, homology
        


class TestEnv(gym.Env):
    def __init__(self, env_name, multicore_num):
        super(TestEnv, self).__init__()
        self.observation_space = Box(low=0, high=100, shape=(510, ), dtype=float)
        self.action_space = gym.spaces.Discrete(n=300)
        self.pep_ori = None
        self.hla_now = None
        self.pep_now = None
        self.gathered_vector = None
        self.is_bound = None
        self.env_name = env_name
        self.if_aomp = False
        self.multicore_num = multicore_num
        self.binding_affinity = None
    
    def reset(self, id_hla_pep_tuple=None):
        if not id_hla_pep_tuple:
            self.is_bound = True
            while self.is_bound:
                self.hla_now, self.pep_ori = utils.get_random_training_data()
                self.gathered_vector, self.is_bound, self.binding_affinity = utils.get_pep_hla_attention_vector(self.pep_ori, self.hla_now, self.env_name, self.multicore_num, if_aomp=self.if_aomp)

            self.pep_now = self.pep_ori
            self.already_action_list = []
        
        else:
            self.hla_now, self.pep_ori = id_hla_pep_tuple
            self.gathered_vector, self.is_bound, self.binding_affinity = utils.get_pep_hla_attention_vector(self.pep_ori, self.hla_now, self.env_name, self.multicore_num, if_aomp=self.if_aomp)
            self.pep_now = self.pep_ori
            self.already_action_list = []

        return self.gathered_vector


    def step(self, action):
        self.pep_now = utils.mutate_pep(self.pep_now, action)
        self.gathered_vector, self.is_bound, self.binding_affinity = utils.get_pep_hla_attention_vector(self.pep_now, self.hla_now, self.env_name, self.multicore_num, if_aomp=self.if_aomp)

        if action in self.already_action_list:
            return self.gathered_vector, -1, True, 0.0
        self.already_action_list.append(action)
        
        homology = utils.get_homology(self.pep_ori, self.pep_now)

        if not utils.judge_available_move(self.pep_ori, action):
            return self.gathered_vector, -1, True, 0.0

        if homology < 0.2:
            is_done = True
        
        else:
            is_done = self.is_bound

        reward = 1 if self.is_bound else -1
        return self.gathered_vector, reward, is_done, homology


