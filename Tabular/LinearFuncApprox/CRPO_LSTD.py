from PainlessPolicyOpt.Tabular.tabularCMDPEnv import TabularCMDP
from PainlessPolicyOpt.Tabular.policy import *
from PainlessPolicyOpt.Tabular.helper_utlities import get_optimal_pol_and_perf, save_data_CRPO, update_store_info_CRPO, \
    store_noisy_estimator_data_CRPO, print_tc_features
from PainlessPolicyOpt.feature import TileCodingFeatures, TabularTileCoding
from PainlessPolicyOpt.LSTD import LSTD, Sampling
import os
import argparse
import time

"""
CRPO
This implements a LFA version with LSTD based sampling to estimate the Q_hat function.
"""


class CRPO_Primal:
    """
    Paper: CRPO,[2021] (https://arxiv.org/abs/2011.05869)
    Updates the policy alternatingly between objective improvement and constraint satisfaction.
    Algorithm 1, Pg 4
    """

    def __init__(self, cmdp, init_policy, num_constraints, learning_rate, eta):
        self.policy = init_policy  # policy class object
        self.num_constraints = num_constraints
        self.learning_rate = learning_rate
        self.eta = eta
        self.cmdp = cmdp

    def md_update(self, Q):
        """
        Mirror Ascent update for policy
        """
        new_policy_prob = self.policy.policy_prob * np.exp(self.learning_rate * Q)
        denom = np.sum(new_policy_prob, axis=1)
        for s in range(self.policy.policy_prob.shape[0]):
            new_policy_prob[s, :] /= denom[s]  # normalizing policy prob
        return new_policy_prob

    def update(self, Q, V_g_perf):
        # MD update: pi_{t+1}(a|s) = pi_{t}(a|s) exp(alpha * Q(s,a))/ [\sum_{a'}pi_{t}(a'|s) exp(alpha * Q(s,a'))]
        # takes Q matrix of [ (num_constraint+1) X S X A ] dim
        # takes V_g_perf of [num_constraints] dim = \sum_{s} \rho(s) V_g(s) where \rho is initial state distribution
        index_constraint_violation = []
        flag_constraint_violation = False
        for c in range(self.num_constraints):
            if V_g_perf[c] < (self.cmdp.b[c] - self.eta):  # violates the constraints
                index_constraint_violation.append(c)  # records which constraints are violated
                flag_constraint_violation = True
        if not flag_constraint_violation:
            # no constraint violation then maximize Q_r objective function
            self.policy.policy_prob = self.md_update(Q[0])  # update policy in direction of Q_r
        else:
            # constraint is violated, choose randomly i \in index_constraint_violation and update policy which maximizes Q_g[i]
            violated_constraint_index = np.random.choice(index_constraint_violation)
            self.policy.policy_prob = self.md_update(Q[violated_constraint_index + 1])


def run_CRPO_agent(cmdp, optimal_performance, learning_rate_pol, eta, num_iterations, output_dir, moving_avg_window,
                   full_average, initial_policy, num_samples, feature, tc_args):
    # initialize storing data information
    og_list, avg_og_list, cv_list, avg_cv_list = [], [], [], []
    V_r_list, V_g_list, policy_list, Q_list = [], [], [], []

    # noisy estimates
    V_r_hat_list, V_g_hat_list, diff_in_Q_r, diff_in_Q_c, Q_hat_list = [], [], [], [], []

    # CRPO Primal: to update the policy
    primal = CRPO_Primal(cmdp, initial_policy, cmdp.num_constraints, learning_rate_pol, eta)

    # sampler for sampling the data
    data_sampler = Sampling(cmdp, num_samples, tc_args, True)

    # estimate Q hat from LSTD
    q_estimator = LSTD(feature.get_feature_size(), feature, cmdp, data_sampler)

    for t in range(num_iterations):
        Q_t_list = []
        Q_hat_t_list = []
        curr_policy = primal.policy
        Q_r = curr_policy.get_Q_function(cmdp.R)
        Q_t_list.append(Q_r)
        policy_list.append(curr_policy.policy_prob)
        V_r = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r))
        V_r_list.append(V_r)
        og_list.append(optimal_performance - V_r)

        # Q estimator
        # update the weights of q_hat matrix
        q_estimator.run_solver(curr_policy.policy_prob)
        q_hat = np.asarray([q_estimator.get_estimated_Q(tc_args.convert_state_to_two_coordinate(s)) for s in
                            range(cmdp.S)])  # dim [S X A X num_constraints+1]
        Q_r_hat = q_hat[:, :, 0]
        Q_hat_t_list.append(Q_r_hat)
        V_r_hat_list.append(np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r_hat)))
        diff_in_Q_r.append(q_estimator.get_error_norm(Q_r, Q_r_hat))
        V_g_rho = np.zeros(cmdp.num_constraints)
        V_g_rho_hat = np.zeros(cmdp.num_constraints)
        error_in_cost_constraint = 0
        for c in range(cmdp.num_constraints):
            Q_c = curr_policy.get_Q_function(cmdp.G[c])
            Q_c_hat = q_hat[:, :, c + 1]
            Q_t_list.append(Q_c)
            Q_hat_t_list.append(Q_c_hat)
            V_g_rho[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c))
            V_g_rho_hat[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c_hat))
            error_in_cost_constraint += q_estimator.get_error_norm(Q_c, Q_c_hat)
        diff_in_Q_c.append(error_in_cost_constraint / cmdp.num_constraints)
        V_g_list.append(V_g_rho)
        V_g_hat_list.append(V_g_rho_hat)
        cv_list.append(cmdp.b - V_g_rho)
        Q_t = np.asarray(Q_t_list)
        Q_hat_t = np.asarray(Q_hat_t_list)
        # update primal policy
        primal.update(Q_hat_t, V_g_rho_hat)
        Q_list.append(Q_t)
        Q_hat_list.append(Q_hat_t)
        update_store_info_CRPO(optimal_performance, V_r_list, V_g_list, cmdp.b, moving_avg_window,
                               t, full_average, avg_og_list, avg_cv_list)
        if t % 100 == 0:
            save_data_CRPO(output_dir, np.asarray(og_list), np.asarray(cv_list),
                           np.asarray(avg_og_list), np.asarray(avg_cv_list),
                           np.asarray(curr_policy.policy_prob), np.asarray(policy_list), np.asarray(Q_list))
            store_noisy_estimator_data_CRPO(output_dir, diff_in_Q_r, diff_in_Q_c, V_g_hat_list,
                                            V_r_hat_list, V_g_list, V_r_list, Q_hat_list)
    save_data_CRPO(output_dir, np.asarray(og_list), np.asarray(cv_list),
                   np.asarray(avg_og_list), np.asarray(avg_cv_list),
                   np.asarray(curr_policy.policy_prob), np.asarray(policy_list), np.asarray(Q_list))
    store_noisy_estimator_data_CRPO(output_dir, diff_in_Q_r, diff_in_Q_c, V_g_hat_list,
                                    V_r_hat_list, V_g_list, V_r_list, Q_hat_list)

if __name__ == '__main__':
    start_main_t = time.time()
    try_locally = True  # variable to be set True when running on local machine
    parser = argparse.ArgumentParser()
    # by default value is set to the optimal learning rates for policy and lambda
    parser.add_argument('--learning_rate', type=float, default=0.75)
    # eta: slack used fo constraint violation in CRPO
    parser.add_argument('--eta', help="eta", type=float, default=0)
    # cmdp: 1 -> creates env with constraints and solve for GDA on both policy and lambda
    # cmdp:0 -> creates a mdp and solve for Online Mirror Ascent on policy
    parser.add_argument('--cmdp', help="create a cmdp:1 or mdp:0", type=int, default=1)

    # multiple_constraints: 0 -> Adds only 1 constraint for cmdp
    # multiple_constraints: 1 -> There are 3 constraints for cmdp
    parser.add_argument('--multiple_constraints', help="multiple constraints: 0, 1", type=int, default=0)

    # full_average:1 -> Stores result with average policy from iteration 0 to t
    # full_average:0 -> Would use Moving average with window size selected from next parameter
    parser.add_argument('--full_average', help="Full average: 0, 1", type=int, default=1)

    # stores result with a moving average window over policy for past k iterations
    parser.add_argument('--moving_avg_window', help="window size", type=int, default=200)

    # Run is used to keep track of runs with different policy initialization.
    parser.add_argument('--run', help="run number", type=int, default=1)

    parser.add_argument('--num_iterations', help="iterations", type=int, default=10)

    # iht table size
    parser.add_argument('--iht_size', help="iht size", type=float, default=100)

    # iht number of tiles
    parser.add_argument('--num_tiles', help="num of tiles", type=int, default=1)

    # iht number of tiles
    parser.add_argument('--tiling_size', help="dim of grid", type=int, default=5)

    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=10)

    parser.add_argument('--gamma', help="discount factor", type=float, default=0.9)

    parser.add_argument('--b_thresh', help="threshold for single constraint", type=float, default=1.5)

    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"
    outer_file_name = "LR" + str(args.learning_rate) + "_Iter" + str(args.num_iterations) + \
                      "_MC" + str(int(args.multiple_constraints)) + "_FullAvg" + str(int(args.full_average)) + \
                      "_iht_size" + str(int(args.iht_size)) + "_ntiles" + str(int(args.num_tiles)) + \
                      "_tileDim" + str(args.tiling_size) + "_num_samples" + str(args.num_samples) + \
                      "_gam" + str(args.gamma) + "_b" + str(args.b_thresh)

    fold_name = os.path.join(save_dir_loc, "Results/Tabular/CRPO/ModelFree/LFA/TileCodingFeatures")
    output_dir_name = os.path.join(fold_name, outer_file_name)
    inner_file_name = "R" + str(args.run)
    output_dir = os.path.join(output_dir_name, inner_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.multiple_constraints = bool(args.multiple_constraints)
    args.full_average = bool(args.full_average)
    # creates a cmdp
    cmdp_env = TabularCMDP(add_constraints=True, multiple_constraints=args.multiple_constraints)
    _, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=True)
    seed_val = 0
    rng = np.random.RandomState(seed_val + args.run)
    initial_policy = Policy(cmdp_env, rng)
    initial_policy.policy_prob = np.ones((cmdp_env.S, cmdp_env.A)) / cmdp_env.A  # initial policy as uniform policy
    # tile coding features
    tabular_tc = TabularTileCoding(args.iht_size, args.num_tiles, args.tiling_size)
    tc_feature = TileCodingFeatures(cmdp_env.A, tabular_tc.get_tile_coding_args())
    print_tc_features(tc_feature, cmdp_env, tabular_tc, output_dir)
    run_agent_params = {'cmdp': cmdp_env,
                        'optimal_performance': optimal_perf,
                        'learning_rate_pol': args.learning_rate,
                        'eta': args.eta,
                        'num_iterations': args.num_iterations,
                        'output_dir': output_dir,
                        'moving_avg_window': args.moving_avg_window,
                        'full_average': args.full_average,
                        'initial_policy': initial_policy,
                        'num_samples': args.num_samples,
                        'feature': tc_feature,
                        'tc_args': tabular_tc
                        }
    run_CRPO_agent(**run_agent_params)
    end_main_t = time.time()
