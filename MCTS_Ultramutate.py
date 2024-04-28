import numpy as np
import torch
from ultramutate_env import AompEnv
import utils
import math


# This version of MCTS implement the searching process without ValueNet
# This is mainly because the ValueNet is just not stable enough to be used in the searching process


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_single_move_pred_v(policy_net, state):
    pred_v = policy_net(*utils.convey_gathered_vector_to_net(state, DEVICE))

    return pred_v

def get_single_move_prob(policy_net, state):
    pred_v = policy_net(*utils.convey_gathered_vector_to_net(state, DEVICE)).double()
    pred_sm = torch.nn.functional.softmax(pred_v, dim=1)
    actions = np.arange(300).tolist()
    probabilities = pred_sm.cpu().detach().numpy()
    
    return actions, probabilities.tolist()[0]
    
def roll_out(env, policy_net, pep_ori, pep_now, hla_now):    
    env.hla_now = hla_now
    env.pep_now = pep_now
    env.pep_ori = pep_ori

    env.gathered_vector, env.is_bound, env.binding_affinity= utils.get_pep_hla_attention_vector(env.pep_now, env.hla_now, env.env_name, env.multicore_num)
    state = env.gathered_vector
    
    sf = 5
    bf = 2

    if env.is_bound:
        homology = utils.get_homology(pep_ori, pep_now)
        rollout_value = sf * math.log(homology) + bf
        return rollout_value

    is_done = False
    moves = []
    probabilities = []

    while not is_done:
        actions, probabilities = get_single_move_prob(policy_net, state)
        action = np.random.choice(actions, p=probabilities)
        state, reward_env, is_done, _ = env.step(action)
    
    homology = utils.get_homology(pep_ori, env.pep_now)
    if reward_env > 0:
        rollout_value = sf * math.log(homology) + bf
    else:
        rollout_value = sf * math.log(homology)
    
    return rollout_value


def check_hla_pep_over(pep_ori, pep_now, hla_now, env_name, multicore_num):
    gathered_vector, is_bound, binding_affinity = utils.get_pep_hla_attention_vector(pep_now, hla_now, env_name, multicore_num)
    
    return True if is_bound else False
    

class AlphaGoNode:
    def __init__(self, parent=None, probability=0):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability
        self.u_value = probability

    def select_child(self):
        return max(self.children.items(),
                   key=lambda child: child[1].q_value + child[1].u_value)


    def expand_children(self, moves, probabilities):
        for move, probability in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(parent=self, probability=probability)
        
    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)
        
        self.visit_count += 1
        self.q_value = leaf_value / self.visit_count
        
        if self.parent is not None:
            c_u = 0.1

            self.u_value = c_u * np.sqrt(self.parent.visit_count) * self.prior_value / (1 + self.visit_count)


class MCTS_Ultramutate():
    def __init__(self, policy_agent, fast_policy_agent, multicore_num, lambda_value=0.5, num_simulations=100, depth=20, rollout_limit=100):
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        # self.value = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.multicore_num = multicore_num
        self.env = AompEnv('MCTS', self.multicore_num)
        self.root = AlphaGoNode()
        
    
    def select_move(self, game_state, pep_ori, pep_now, hla_now):
        for sim_num in range(self.num_simulations):
            current_state = game_state
            current_pep_ori = pep_ori
            current_pep = pep_now
            current_hla = hla_now
            node = self.root
            self.env.hla_now = current_hla
            self.env.pep_now = current_pep
            self.env.pep_ori = current_pep_ori
            self.env.gathered_vector, self.env.is_bound, self.env.binding_affinity = utils.get_pep_hla_attention_vector(self.env.pep_now, self.env.hla_now, self.env.env_name, self.multicore_num)

            for depth in range(self.depth):
                print(" Performing MCTS, simulation: ", sim_num+1, " depth: ", depth+1, end='\r')
                if not node.children:
                    is_break = check_hla_pep_over(current_pep_ori, current_pep, current_hla, self.env.env_name, self.env.multicore_num)
                    if is_break:
                        break
                    
                    # moves, probabilities = self.policy.probabilities(current_state) # TODO: policy.probabilities
                    
                    moves, probabilities = get_single_move_prob(self.policy, current_state)
                    node.expand_children(moves, probabilities)
                
                move, node = node.select_child()
                current_state, reward, is_done, homology = self.env.step(move)
                current_pep = self.env.pep_now
                current_hla = self.env.hla_now
                if is_done:
                    break

            env_once = AompEnv('MCTS_ONCE', self.multicore_num)
            rollout_value = roll_out(env_once, self.policy, current_pep_ori, current_pep, current_hla)
            
            weighted_value = rollout_value
            
            node.update_values(weighted_value)

        move_fail_flag = False

        try:
            move_list = [self.root.children.get(i).visit_count for i in range(300)]
        
        except:
            move_fail_flag = True
            move_list = None
        
        pred_v = get_single_move_pred_v(self.policy, game_state)
        self.root = AlphaGoNode()

        
        return move_fail_flag, move_list, pred_v


            

        

