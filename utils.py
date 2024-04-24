import pandas as pd
import numpy as np
import os
import random
import difflib
import math
from torch import nn
import torch
from collections import namedtuple
import warnings
warnings.filterwarnings("ignore")
import subprocess
current_directory = os.path.dirname(os.path.abspath(__file__))


REWARD_BOUND = 10
MAX_HLA_LEN = 34
MAX_PEP_LEN = 15
ALL_HLA_DICT = np.load(os.path.join(current_directory, 'ALL_HLA_DICT.npy'), allow_pickle=True).item()
ALL_HLA_DICT["HLA-A*30:04"] = "YSAMYQENVAHTDENTLYIIYEHYTWAVWAYTWY"
ALL_HLA_DICT["HLA-C*07:01"] = "YDSGYRENYRQADVSNLYLRYDSYTLAALAYTWY"
ALL_HLA_DICT["HLA-A*11:01"] = "YYAMYQENVAQTDVDTLYIIYRDYTWAAQAYRWY"
ALL_HLA_DICT["HLA-A*26:01"] = "YYAMYRNNVAHTDANTLYIRYQDYTWAEWAYRWY"
ALL_HLA_DICT["HLA-B*08:01"] = "YDSEYRNIFTNTDESNLYLSYNYYTWAVDAYTWY"
ALL_HLA_DICT["HLA-B*38:01"] = "YYSEYRNICTNTYENIAYLRYNFYTWAVLTYTWY"
ALL_HLA_DICT["HLA-B*51:01"] = "YYATYRNIFTNTYENIAYWTYNYYTWAELAYLWH"
ALL_HLA_DICT["HLA-B*52:01"] = "YYATYREISTNTYENIAYWTYNYYTWAELAYLWH"
ALL_HLA_DICT["HLA-B*15:10"] = "YYSEYRNICTNTYESNLYLRYDYYTWAELAYLWY"


ALL_ACID_TYPES = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

BATCH_SIZE = 1280
PERCENTILE = 0.90
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def transform_result_dict(ori_dict):
    final_dict = {}
    for key in ori_dict.keys():
        info_dict ={}
        action_string_list = []
        ori_pep = key[1]
        result_pep = ori_pep
        for action_num in ori_dict[key][0][:-3]:
            mutate_pos, mutate_type = action_transfer(action_num)
            action_string = "{}{}{}".format(ori_pep[mutate_pos], mutate_pos, mutate_type)
            action_string_list.append(action_string)
            result_pep = mutate_pep(result_pep, action_num)
        
        info_dict['actions'] = action_string_list
        info_dict['results'] = result_pep
    
    final_dict[key] = info_dict

    return final_dict


def get_pep_hla_from_fasta(peptide_fasta, HLA_file):
    with open(peptide_fasta, 'r') as f:
        lines = f.readlines()

    pep_list = []
    for ele in lines:
        if ele[0] != '>':
            pep_list.append(ele.strip())
    
    with open(HLA_file, 'r') as f:
        lines = f.readlines()
    
    HLAs = [ele.strip() for ele in lines]

    assert len(pep_list) == len(HLAs), "HLAs and peptides are not corresponding!"
    HLA_pep_list = [hla_pep_pair for hla_pep_pair in zip(HLAs, pep_list)]

    return HLA_pep_list


def convey_gathered_vector_to_net(gathered_vector_batch, device):
    """
    Convey the gathered vector to the net.
    :param gathered_vector: the gathered vector.
    :return: the vector to the net.
    """
    if len(gathered_vector_batch.shape) == 1:
        pep_hla_embed_batch, atten_batch,  prefer_batch, contrib_batch = gathered_vector_batch[ :49], gathered_vector_batch[ 49: 49 + 510], gathered_vector_batch[ 49 + 510: 49 + 510 + 300], gathered_vector_batch[ 49 + 510 + 300:]
        pep_hla_embed_batch, atten_batch, prefer_batch, contrib_batch = torch.tensor(pep_hla_embed_batch).unsqueeze(0), torch.tensor(atten_batch).unsqueeze(0), torch.tensor(prefer_batch).unsqueeze(0), torch.tensor(contrib_batch).unsqueeze(0)

    else:
        pep_hla_embed_batch, atten_batch,  prefer_batch, contrib_batch = gathered_vector_batch[:, :49], gathered_vector_batch[:, 49: 49 + 510], gathered_vector_batch[:, 49 + 510: 49 + 510 + 300], gathered_vector_batch[:, 49 + 510 + 300:]
    return pep_hla_embed_batch.long().to(device), atten_batch.float().to(device), prefer_batch.float().to(device), contrib_batch.float().to(device)


def get_attention_vector(attention_csv):
    """
    Get attention vector from attention csv file. Padding considered already.
    :param attention_csv: attention csv file.
    :return: attention vector.
    """
    attention_matrix = pd.read_csv(attention_csv)
    attention_vector = None
    # remove the rows if the first element is 'sum', 'posi' or 'contrib'
    value_lines = []
    for i in range(len(attention_matrix.values)):
        if attention_matrix.values[i][0] != 'sum' and attention_matrix.values[i][0] != 'posi' and attention_matrix.values[i][0] != 'contrib':
            value_lines.append(attention_matrix.values[i])

    # pad the hla_length to 34
    pepetide_length = len(value_lines[0]) - 1
    while len(value_lines) < MAX_HLA_LEN:
        value_lines.append(np.concatenate((['None'], np.zeros(pepetide_length))))


    for line in value_lines:
        attention_line = np.delete(line, 0)
        attention_line = attention_line.astype(float)
        while len(attention_line) < MAX_PEP_LEN:
            attention_line = np.append(attention_line, 0)

        # concatenate attention lines
        if attention_vector is None:
            attention_vector = attention_line.astype(float)
        else:
            attention_vector = np.concatenate((attention_vector, attention_line.astype(float)), axis=0)
    
    return attention_vector

def general_HLA_preference_matrix(pep_seq, hla_type):
    Attention_csv_folder = os.path.join(current_directory, "./Attention/peptideAAtype_peptidePosition")
    Attention_num_csv_folder = os.path.join(current_directory, "./Attention/peptideAAtype_peptidePosition_NUM")
    pep_len = len(pep_seq)
    hla_tr_type = hla_type.replace("*", '_').replace(":", '_')
    npy_file = os.path.join(Attention_csv_folder, "{}_Length{}.npy".format(hla_tr_type, pep_len))
    npy_num_file = os.path.join(Attention_num_csv_folder, "{}_Length{}_num.npy".format(hla_tr_type, pep_len))

    if os.path.exists(npy_file):
        prefer_matrix = np.load(npy_file, allow_pickle=True).item()['positive']
        prefer_num_matrix = np.load(npy_num_file, allow_pickle=True).item()['positive']
        

    else:
        npy_all_file = os.path.join(Attention_csv_folder, "Allsamples_Alllengths.npy")
        prefer_matrix = np.load(npy_all_file, allow_pickle=True).item()[pep_len]['positive']
        npy_num_all_file = os.path.join(Attention_num_csv_folder, "Allsamples_Alllengths_num.npy")
        prefer_num_matrix = np.load(npy_num_all_file, allow_pickle=True).item()[pep_len]['positive']
            

    # to confirm all matrix amino acids are stored in a fixed order.
    if list(prefer_matrix.index) != ALL_ACID_TYPES:
        prefer_matrix = prefer_matrix.loc[ALL_ACID_TYPES]
    
    if list(prefer_num_matrix.index)[:-1] != ALL_ACID_TYPES:
        prefer_num_matrix = prefer_num_matrix.loc[ALL_ACID_TYPES + ['sum']]

    prefer_matrix.loc['sum'] = prefer_matrix.sum(axis = 0)
    prefer_matrix['sum'] = prefer_matrix.sum(axis = 1)

    contrib = np.zeros((20, pep_len))
    
    for aai, aa in enumerate(prefer_matrix.index[:-1]):
        for pi, posi in enumerate(prefer_matrix.columns[:-1]):
            p_aa_posi = prefer_matrix.loc[aa, posi] / prefer_num_matrix.loc[aa, posi]
            p_posi = prefer_num_matrix.loc['sum' , 'sum'] / prefer_matrix.loc['sum', 'sum']

            contrib[aai, pi] = p_aa_posi * p_posi                
    
    contrib = pd.DataFrame(contrib, index = prefer_matrix.index[:-1], columns = prefer_matrix.columns[:-1])
    contrib.fillna(0, inplace = True)

    # remove the rows of prefer_matrix if its index is 'sum'
    value_lines = []
    for i in range(len(prefer_matrix.values)):
        if prefer_matrix.index[i] != 'sum':
            value_lines.append(prefer_matrix.values[i])
    # value_lines = prefer_matrix.values
    prefer_vector = None

    for line in value_lines:
        now_line = line.astype(float)
        while len(now_line) < MAX_PEP_LEN:
            now_line = np.append(now_line, 0)
        
        if prefer_vector is None:
            prefer_vector = now_line
        else:
            prefer_vector = np.concatenate((prefer_vector, now_line.astype(float)), axis=0)
    
    contrib_value_lines = contrib.values
    contrib_prefer_vector = None

    for line in contrib_value_lines:
        now_line = line.astype(float)
        while len(now_line) < MAX_PEP_LEN:
            now_line = np.append(now_line, 0)
        
        if contrib_prefer_vector is None:
            contrib_prefer_vector = now_line
        
        else:
            contrib_prefer_vector = np.concatenate((contrib_prefer_vector, now_line.astype(float)), axis=0)

    return prefer_vector, contrib_prefer_vector


# by now the hla_types are limited in the common_hla_sequence.csv
def get_HLA_seq(hla_type):
    return ALL_HLA_DICT[hla_type]

def get_pep_hla_attention_vector(pep_seq, hla_type, env_name, multi_core_num, python_exec="~/.conda/envs/ultramutate/bin/python", if_aomp=False):
    hla_seq = get_HLA_seq(hla_type)
    pep_embed = get_pep_seq_embedding(pep_seq)
    hla_embed = get_hla_seq_embedding(hla_seq)
    pep_hla_embed = np.array(pep_embed + hla_embed)
    pep_file_folder = os.path.join(current_directory, "./pep_hla_files/{}/{}".format(env_name, multi_core_num))
    hla_file_folder = os.path.join(current_directory, "./pep_hla_files/{}/{}".format(env_name, multi_core_num))

    # if these two folder do not exist, create them
    if not os.path.exists(pep_file_folder):
        os.makedirs(pep_file_folder)
    if not os.path.exists(hla_file_folder):
        os.makedirs(hla_file_folder)


    pep_file = "{}/pep.fasta".format(pep_file_folder)
    hla_file = "{}/hla.fasta".format(hla_file_folder)
    
    output_dir = os.path.join(current_directory, "./round_output/{}/{}".format(env_name, multi_core_num))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(pep_file, 'w') as pf:
        pf.write(">" + pep_seq + "\n" + pep_seq + "\n")
        pf.write("\n")
    
    with open(hla_file, 'w') as hf:
        hf.write(">" + hla_type + "\n" + get_HLA_seq(hla_type) + "\n")
        hf.write("\n")
    # print("{} ./TransAI/pHLAIformer.py --peptide_file {} --HLA_file {} --threshold 0.5 --cut_length 10 --cut_peptide True --output_dir {} --output_attention True --output_heatmap True --output_mutation False".format(python_exec, pep_file, hla_file, output_dir))
    subprocess.run("rm {}/attention/*".format(output_dir), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if if_aomp:
        subprocess.run("rm {}/figures/*".format(output_dir), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("rm {}/mutation/*".format(output_dir), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("{} ./TransAI/pHLAIformer.py --peptide_file {} --HLA_file {} --threshold 0.5 --cut_peptide False --output_dir {} --output_attention True --output_heatmap False --output_mutation True".format(python_exec, pep_file, hla_file, output_dir), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run("{} ./TransAI/pHLAIformer.py --peptide_file {} --HLA_file {} --threshold 0.5 --cut_peptide False --output_dir {} --output_attention True --output_heatmap False --output_mutation False".format(python_exec, pep_file, hla_file, output_dir), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    attention_file_name = "{}_{}_attention.csv".format(hla_type, pep_seq)
    attention_vector = get_attention_vector(os.path.join(output_dir, "attention", attention_file_name))
    prefer_vector, contrib_prefer_vector = general_HLA_preference_matrix(pep_seq, hla_type)
    binding_result_csv = os.path.join(output_dir, "predict_results.csv")

    binding_data = pd.read_csv(binding_result_csv)
    binding_result = binding_data['y_pred'][0]

    # binding_affinity added. Remember to add this to the corresponding codes such as aomp_env.py
    binding_affinity = binding_data['y_prob'][0]


    # gathered_vector = np.array([pep_hla_embed, attention_vector, prefer_vector, contrib_prefer_vector], dtype=object)
    gathered_vector = np.concatenate([pep_hla_embed, attention_vector, prefer_vector, contrib_prefer_vector])
    if binding_result == 0:
        binding_bol = False
    elif binding_result == 1:
        binding_bol = True

    return gathered_vector, binding_bol, binding_affinity


def action_transfer(action_num):
    mutate_pos = action_num // 20
    mutate_type = ALL_ACID_TYPES[action_num % 20]
    return mutate_pos, mutate_type

def mutate_pep(pep_seq, action_num):
    # mutate the peptide in the given position
    mutate_pos, mutate_type = action_transfer(action_num)
    if mutate_pos >= len(pep_seq):
        return pep_seq

    pep_seq_list = list(pep_seq)
    pep_seq_list[mutate_pos] = mutate_type
    
    return ''.join(pep_seq_list)

def judge_available_move(ori_pep, action_num):
    mutate_pos, mutate_type = action_transfer(action_num)
    if mutate_pos >= len(ori_pep):
        print("Action too far!")
        return False

    if ori_pep[mutate_pos] == mutate_type:
        print("Useless Mutation!")
        return False
    
    return True


def get_homology(ori_pep, mutated_pep):
    return difflib.SequenceMatcher(None, ori_pep, mutated_pep).ratio()

def balanced_reward(homology, is_bound):
    if homology >= 0.7 and is_bound:
        return homology * REWARD_BOUND

    # elif is_bound:
    #     return homology * REWARD_BOUND
        
    else:
        return math.log(homology)

def get_best_sup_mutate(pep_seq, hla_type, env_name, multicore_num):
    """
    There could be some samples that are originally binders.(Even though all samples should be filtered by there label, 
    the labels are for TransPHLA training, they might not be the same as TransPHLA's prediction after training).
    In these cases, the firt row of the csv is the original peptide itself and it's a nonsense sample.
    But don't worry, type(right_mutation_best) != str should not pass and return a None
    """
    mutation_csv = "./round_output/{}/{}/mutation/{}_{}_mutation.csv".format(env_name, multicore_num, hla_type, pep_seq)
    mutation_data = pd.read_csv(mutation_csv)
    right_mutation = mutation_data[mutation_data['y_pred'] == 1]
    if len(right_mutation) == 0:
        return None
    
    least_num = right_mutation['mutation_AA_number'].values[0]
    right_mutation_least = right_mutation[right_mutation['mutation_AA_number'] == least_num].sort_values('y_prob', ascending=False)
    right_mutation_best = right_mutation_least['mutation_position_AAtype'].values[0]
    # print(right_mutation_best)
    if type(right_mutation_best) != str:
        return None
    veri_mutations = right_mutation_best.split(',')

    return transfer_mutations_to_act(veri_mutations)


def transfer_mutations_to_act(veri_mutations):
    # mut_pos_list = list(map(lambda x: int(x.split('|')[0]), veri_mutations))
    test_mutation_comp = ','.join(veri_mutations)
    if '|' not in test_mutation_comp or '/' not in test_mutation_comp:
        return None

    action_num_list = []
    for mutation in veri_mutations:
        mut_pos = int(mutation.split('|')[0])
        mut_type = mutation.split('/')[1]
        action_num = (mut_pos - 1) * 20 + ALL_ACID_TYPES.index(mut_type)
        action_num_list.append(action_num)

    return action_num_list


def iterate_sup_train_batches(env, batch_size):
    # EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])
    sm = nn.Softmax(dim=1)
    episode_steps = []
    training_x = []
    training_y = []
    while True:
        obs = env.reset()
        hla_type, pep_seq = env.hla_now, env.pep_ori
        action_num_list = get_best_sup_mutate(pep_seq, hla_type, env.env_name, env.multicore_num)
        # print("len_action_num_list", len(action_num_list))

        if action_num_list is None:
            continue
        step = EpisodeStep(observation=obs, action=action_num_list[0])
        episode_steps.append(step)
        for index in range(1, len(action_num_list)):
            
            obs, reward, is_done, info = env.step(action_num_list[index-1])
            step = EpisodeStep(observation=obs, action=action_num_list[index])
            episode_steps.append(step)
        
        # print("len_episode_steps", len(episode_steps))
        if len(episode_steps) > batch_size:
            training_x.extend(map(lambda step: step.observation, episode_steps[0:batch_size]))
            # training_y.extend(map(lambda step: transfer_NN_action(step.action), episode_steps))
            training_y.extend(map(lambda step: step.action, episode_steps[0:batch_size]))
            yield training_x, training_y
            training_x = []
            training_y = []
            episode_steps = []
    

def transfer_NN_action(action_num):
    # transfer the action_num to a np array with all elements zero except the action_num position
    action_array = np.zeros(20 * 15)
    action_array[action_num] = 1
    return action_array


def get_pep_seq_embedding(seq):
    seq_embedding = []
    for acid in seq:
        seq_embedding.append(ALL_ACID_TYPES.index(acid))

    while len(seq_embedding) < MAX_PEP_LEN:
        seq_embedding.append(20)

    return seq_embedding

def get_hla_seq_embedding(seq):
    seq_embedding = []
    for acid in seq:
        seq_embedding.append(ALL_ACID_TYPES.index(acid))

    while len(seq_embedding) < MAX_HLA_LEN:
        seq_embedding.append(20)

    return seq_embedding


if __name__ == "__main__":
    hla_now, pep_ori = get_random_training_data()
    gathered_vector, binding_bol, binding_affinity = get_pep_hla_attention_vector(pep_ori, hla_now, "UltraMu", 0, python_exec="~/.conda/envs/ultramutate/bin/python", if_aomp=True)
    pass
