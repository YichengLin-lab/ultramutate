import numpy as np
import torch
import utils
from MCTS_Ultramutate import *
from aomp_DQNNet import *
import time
from aomp_env import *
import argparse
from torch import nn
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sm = nn.Softmax(dim=1)

def judge_list_all_zero(input_list):
    for i in range(len(input_list)):
        if input_list[i] != 0:
            return False
        else:
            continue
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--peptide', type=str, default=None, help='sequence of a single peptide that needs to be optimized')
    parser.add_argument('--HLA', type=str, default=None, help='a single HLA allele type')
    parser.add_argument('--peptide_fasta', type=str, default=None, help='fasta file containing a list of peptides')
    parser.add_argument('--HLA_file', type=str, default=None, help='file containing multiple HLA alleles')
    parser.add_argument('--num_simulations', type=int, default=6, help='number of simulations for MCTS')
    parser.add_argument('--output_dir', type=str, default='results', help='output directory for results')
    args = parser.parse_args()

    assert args.output_dir is not None, "Please specify an output directory for results"
    
    if args.peptide is not None and args.HLA is not None:
        # single peptide and HLA type
        all_test_set_list = [(args.HLA, args.peptide)]
    
    elif args.peptide_fasta is not None and args.HLA_file is not None:
        # multiple peptides and HLA types
        all_test_set_list = utils.get_pep_hla_from_fasta(args.peptide_fasta, args.HLA_file)
    
    specific_test_set_list = all_test_set_list

    # value_net = torch.load("./saved_nets/extreme_value_net_newest.pt", map_location=torch.device('cpu')).eval()
    policy_net = torch.load("PPO_reinforce_results/rl_nets/rl_net_trained.pt", map_location=torch.device('cpu')).to(device).eval()
    sl_net = torch.load("PPO_reinforce_results/rl_nets/rl_net_trained.pt", map_location=torch.device('cpu')).to(device).eval()

    mcts_ultramutate = MCTS_Ultramutate(policy_net, sl_net, multicore_num=0, lambda_value=0.99, num_simulations=args.num_simulations, depth=10, rollout_limit=100) # num_sim=60, depth=10

    env_mcts = TestEnv('Test_mcts', multicore_num=0)

    done_values_mcts = []
    homology_values_mcts = []


    t1 = time.time()

    mcts_dict = {}

    for i in range(len(specific_test_set_list)):
        print("Dealing with HLA-peptide pair: ", specific_test_set_list[i])
        round_moves_mcts = []

        obs_v = env_mcts.reset(id_hla_pep_tuple=specific_test_set_list[i])
        
        if env_mcts.is_bound:
            continue
        
        
        is_done = False


        mcts_dict[(env_mcts.hla_now, env_mcts.pep_ori)] = []


        already_mutation_num = 1

        while not is_done:
            if len(env_mcts.already_action_list) > 4:
                break
            print("\n")
            print("Designing mutation No.", already_mutation_num)
            move_fail_flag, move_list, pred_v = mcts_ultramutate.select_move(obs_v, env_mcts.pep_ori, env_mcts.pep_now, env_mcts.hla_now)
            pred_v = sm(pred_v)
            if move_fail_flag == True:
                move = pred_v.max(dim=1)[1].item()
                pred_v[pred_v == pred_v.max(dim=1)[0].item()] = 0
                while move in env_mcts.already_action_list or not utils.judge_available_move(env_mcts.pep_ori, move):
                    move = pred_v.max(dim=1)[1].item()
            
            else:
                best_visit_count = max(move_list) 
                move = move_list.index(best_visit_count)
                move_list[move] = 0
                while move in env_mcts.already_action_list or not utils.judge_available_move(env_mcts.pep_ori, move):
                    if judge_list_all_zero(move_list):
                        move = pred_v.max(dim=1)[1].item()
                        pred_v[pred_v == pred_v.max(dim=1)[0].item()] = 0

                    else: 
                        best_visit_count = max(move_list)
                        move = move_list.index(best_visit_count)
                        move_list[move] = 0

            obs_v, reward, is_done, homology = env_mcts.step(move)
            round_moves_mcts.append(move)
            already_mutation_num += 1

        done_value = 1 if reward > 0 else 0
        mcts_dict[(env_mcts.hla_now, env_mcts.pep_ori)].append(round_moves_mcts + [done_value, env_mcts.binding_affinity, homology])

        done_values_mcts.append(1 if reward > 0 else 0)
        homology_values_mcts.append(homology)


        
        mcts_dict = utils.transform_result_dict(mcts_dict)

        print("\n")
        print("mcts_results: ", mcts_dict)


    t2 = time.time()

    print("Time used: ", t2 - t1)
    np.save(os.path.join(args.output_dir, "mcts_results.npy"), mcts_dict)
    
    np.save(os.path.join(args.output_dir, "done_values_mcts.npy"), done_values_mcts)
    np.save(os.path.join(args.output_dir, "homology_values_mcts.npy"), homology_values_mcts)
    

    print("Mean done rate: ", np.mean(done_values_mcts))
    print("Mean homology: ", np.mean(homology_values_mcts))