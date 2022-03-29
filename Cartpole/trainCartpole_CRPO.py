from CRPO import CRPO_Primal
from PainlessPolicyOpt.feature import CartPoleTileCoding, TileCodingFeatures
from PainlessPolicyOpt.LSTD import LFASampling, LSTD
from helper_utils import update_store_info, save_data, time_log
import os
import argparse
from Cartpole import *
import time
from datetime import timedelta


def run_CRPO_agent(cmdp, learning_rate_pol, eta, num_iterations, output_dir, num_samples, entropy_coeff, feature, seed):
    avg_reward_list, avg_cv_list, cv_list = [], [], []
    lambd_list, avg_lambd_list, V_r_list, V_c_list, Q_weight_list = [], [], [], [], []
    # CRPO Primal: to update the policy
    primal = CRPO_Primal(cmdp, cmdp._num_constraints, learning_rate_pol, eta, entropy_coeff, feature)

    # sampler for sampling the data
    data_sampler = LFASampling(env, env._num_constraints, num_samples, seed, True)

    # estimate Q hat from LSTD
    q_estimator = LSTD(feature.get_feature_size(), feature, cmdp, data_sampler)
    for t in range(num_iterations):
        q_estimator.run_solver(primal)
        v_r = data_sampler.get_V_r()
        V_r_list.append(v_r)
        v_c = data_sampler.get_V_c()
        cv_list.append(cmdp.b - v_c)  # calculates cv with true V_g
        V_c_list.append(v_c)
        primal.set_current_v_c(v_c) # update the primal policy
        w_Q = q_estimator.weights
        primal.update_Q_weights(w_Q)
        primal.increment_t_counter()
        Q_weight_list.append(w_Q)
        update_store_info(V_r_list, V_c_list, cmdp.b, lambd_list, True, avg_reward_list, avg_cv_list,
                          avg_lambd_list)
        if t % 10 == 0:
            # saving result after every 10 iterations
            save_data(output_dir, np.asarray(V_r_list), np.asarray(cv_list),
                      np.asarray(avg_reward_list), np.asarray(avg_cv_list),
                      np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(Q_weight_list))
    save_data(output_dir, np.asarray(V_r_list), np.asarray(cv_list), np.asarray(avg_reward_list),
              np.asarray(avg_cv_list), np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(Q_weight_list))


if __name__ == '__main__':
    start_main_t = time.time()
    try_locally = True  # parameter to try code locally
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate_pol', type=float, default=1.0)

    # eta: slack used fo constraint violation in CRPO
    parser.add_argument('--eta', help="eta", type=float, default=0)

    parser.add_argument('--num_iterations', help="iterations", type=int, default=3)

    # Run is used to keep track of runs with different policy initialization.
    parser.add_argument('--run', help="run number", type=int, default=1)

    parser.add_argument('--num_tilings', type=int, default=8)  # iht number of tiles

    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=5)

    # entropy coefficient
    parser.add_argument('--entropy_coeff', help="entropy coefficient", type=float, default=1e-3)

    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"
    outer_file_name = "LRP" + str(args.learning_rate_pol) + "_eta" + str(args.eta) +\
                      "Iter" + str(args.num_iterations) + "_ntile" + str(args.num_tilings) +\
                      "_num_samples" + str(args.num_samples) + "_Ent" + str(args.entropy_coeff)
    env = CartPoleEnvironment()
    fold_name = os.path.join(save_dir_loc, "Results/LFA/Cartpole/Constraints/CRPO")
    output_dir_name = os.path.join(fold_name, outer_file_name)
    inner_file_name = "R" + str(args.run)
    output_dir = os.path.join(output_dir_name, inner_file_name)
    seed = int(args.run)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # creates a cmdp/mdp based on arguments
    cp_tc = CartPoleTileCoding(num_tilings=args.num_tilings)
    tc_feature = TileCodingFeatures(env.A, cp_tc.get_tile_coding_args())

    run_agent_params = {'cmdp': env,
                        'learning_rate_pol': args.learning_rate_pol,
                        'eta': args.eta,
                        'num_iterations': args.num_iterations,
                        'output_dir': output_dir,
                        'num_samples': args.num_samples,
                        'entropy_coeff': args.entropy_coeff,
                        'feature': tc_feature,
                        'seed': seed
                        }
    run_CRPO_agent(**run_agent_params)
    end_main_t = time.time()
    time_to_finish = timedelta(seconds=end_main_t - start_main_t)
    time_log(time_to_finish, args.num_iterations, output_dir)
