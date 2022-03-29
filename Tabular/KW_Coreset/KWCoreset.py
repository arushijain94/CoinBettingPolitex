from PainlessPolicyOpt.Tabular.tabularCMDPEnv import TabularCMDP
from PainlessPolicyOpt.feature import TileCodingFeatures, TabularTileCoding
import os
import _pickle as cPickle
from collections import defaultdict
import argparse
import numpy as np


def tc_features(tc_feature, cmdp, tc_args):
    state_feature_dict = {}
    state_action_to_index_dict = {}
    index_to_broad_ind_dict = {}
    ind = 0
    for s in range(cmdp.S):
        for a in range(cmdp.A):
            new_s = tc_args.convert_state_to_two_coordinate(s)
            state_feature_dict[(s, a)] = tc_feature.get_feature(new_s, a)[0]
            state_action_to_index_dict[(s, a)] = ind
            index_to_broad_ind_dict[ind] = tc_feature.get_feature(new_s, a)[0]
            ind += 1
    max_feature_val = max(state_feature_dict.values())
    ind_to_phi = np.zeros((ind, max_feature_val + 1))
    for key, val in state_feature_dict.items():
        ind_to_phi[state_action_to_index_dict[key], val] = 1
    return state_feature_dict, state_action_to_index_dict, index_to_broad_ind_dict, ind_to_phi


def compute_marginal_gain(Ginv, x):
    return np.dot(x.T, np.dot(Ginv, x))[0][0]  # computes x^T G^{-1} x


def get_KW_coreset_points(d, regularization, ind_to_phi, termination_tolerance=0.5):
    max_C_size = int(d * (d + 1) / 2)
    Ginv = 1. / regularization * np.eye(d)  # initializing G^{-1}
    n = ind_to_phi.shape[0]
    C = []  # coreset initlialization
    for k in range(max_C_size):
        max_mg_gain = 0
        argmax_mg_gain = 0
        for i in range(n):
            if i in C:
                continue
            x = np.expand_dims(ind_to_phi[i], 1)
            mg_gain = compute_marginal_gain(Ginv, x)
            if mg_gain < 0:
                print('Error')
            if mg_gain > max_mg_gain:  # finding the maximum marginal gain for each G^{-1}
                max_mg_gain = mg_gain
                argmax_mg_gain = i
            # add element with the largest marginal gain to coreset
        if max_mg_gain <= termination_tolerance:
            break  # adaptive termination, when mg_gain is smaller than tolerance
        C.append(argmax_mg_gain)
        # Sherman Morrison update to G_inv
        x = np.expand_dims(ind_to_phi[argmax_mg_gain], 1)
        temp1 = np.dot(Ginv, x)
        temp2 = np.dot(x.T, Ginv)
        Ginv = Ginv - ((np.dot(temp1, temp2) / (1 + np.dot(x.T, temp1))))
    return C


def save_KW_coreset(output_dir, C, index_to_broad_ind_dict, state_feature_dict):
    index_to_list_s_a_dict = defaultdict(list)
    for c in C:
        broad_ind = index_to_broad_ind_dict[c]
        for key, val in state_feature_dict.items():
            if val == broad_ind:
                index_to_list_s_a_dict[val].append(key)
    file_name = os.path.join(output_dir, "Cindex_to_list_s_a_dict.pickle")
    with open(file_name, "wb") as output_file:
        cPickle.dump(index_to_list_s_a_dict, output_file)
    file_name = os.path.join(output_dir, "C_ind.pickle")
    with open(file_name, "wb") as output_file:
        cPickle.dump(C, output_file)


if __name__ == '__main__':
    try_locally = True  # variable to be set True when running on local machine
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiling_size', type=int, default=14)
    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"
    kw_dir = "Results/Tabular/KW_Coreset/Tile_" + str(args.tiling_size)
    output_dir = os.path.join(save_dir_loc, kw_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cmdp_env = TabularCMDP(add_constraints=True, multiple_constraints=0)
    # tile coding features
    tabular_tc = TabularTileCoding(100, 1, args.tiling_size)
    tc_feature = TileCodingFeatures(cmdp_env.A, tabular_tc.get_tile_coding_args())
    state_feature_dict, state_action_to_index_dict, index_to_broad_ind_dict, ind_to_phi = tc_features(tc_feature,
                                                                                                      cmdp_env,
                                                                                                      tabular_tc)
    C = get_KW_coreset_points(ind_to_phi.shape[1], 1.0, ind_to_phi, termination_tolerance=0.5)
    save_KW_coreset(output_dir, C, index_to_broad_ind_dict, state_feature_dict)
