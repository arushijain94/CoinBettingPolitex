from PainlessPolicyOpt.Tabular.tabularCMDPEnv import TabularCMDP
from PainlessPolicyOpt.Tabular.policy import *
from PainlessPolicyOpt.Tabular.helper_utlities  import get_optimal_pol_and_perf, save_data_CRPO, update_store_info_CRPO, store_noisy_estimator_data_CRPO
from TD import *
import os
import argparse

"""
CRPO: constraint rectified policy optimization, primal approach
 to learn safe policy in Q sampling with one hot encoding
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


def run_CRPO_agent(cmdp, optimal_performance, num_iterations, output_dir, moving_avg_window,
                   full_average, initial_policy, eta, learning_rate, nsamples):
    # initialize storing data information
    og_list, avg_og_list, avg_cv_list, cv_list = [], [], [], []
    V_r_list, V_g_list, policy_list, Q_list = [], [], [], []

    # noisy estimates
    V_r_hat_list, V_g_hat_list, diff_in_Q_r, diff_in_Q_c, Q_hat_list = [], [], [], [], []

    curr_policy = initial_policy
    # CRPO Primal: to update the policy
    primal = CRPO_Primal(cmdp, initial_policy, cmdp.num_constraints, learning_rate, eta)
    # TD Q estimator
    q_estimator = TD(cmdp, nsamples, cmdp.num_constraints, True)
    for t in range(num_iterations):
        Q_t_list = []
        Q_t_hat_list = []
        curr_policy.set_policy_prob(primal.policy.policy_prob)
        Q_r = curr_policy.get_Q_function(cmdp.R)
        V_r = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r))
        V_r_list.append(V_r)
        Q_t_list.append(Q_r)

        # get MC estimates for Q value
        q_estimator.set_policy_prob(curr_policy.policy_prob)
        Q_hat = q_estimator.get_Q()  # returns a num_constraints+1 size array

        Q_r_hat = Q_hat[0]
        Q_t_hat_list.append(Q_r_hat)
        V_r_hat_list.append(np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r_hat)))
        diff_in_Q_r.append(q_estimator.get_error_norm(Q_r, Q_r_hat))  # here Q_l contains Q_r value

        # update V_g
        V_g_rho = np.zeros(cmdp.num_constraints)
        V_g_rho_hat = np.zeros(cmdp.num_constraints)
        error_in_cost_constraint = 0
        for c in range(cmdp.num_constraints):
            Q_c = curr_policy.get_Q_function(cmdp.G[c])
            Q_c_hat = Q_hat[c + 1]
            V_g_rho[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c))
            V_g_rho_hat[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c_hat))
            error_in_cost_constraint += q_estimator.get_error_norm(Q_c, Q_c_hat)
            Q_t_list.append(Q_c)
            Q_t_hat_list.append(Q_c_hat)
        V_g_list.append(V_g_rho)
        V_g_hat_list.append(V_g_rho_hat)
        diff_in_Q_c.append(error_in_cost_constraint / cmdp.num_constraints)
        # update Cv and og list
        cv_list.append(cmdp.b - V_g_rho)
        og_list.append(optimal_performance - V_r)

        # update the primal policy
        Q_t = np.asarray(Q_t_list)  # [(num_constraints+1) X S X A]
        Q_t_hat = np.asarray(Q_t_hat_list)
        primal.update(Q_t_hat, V_g_rho_hat)
        policy_list.append(curr_policy.policy_prob)
        Q_list.append(Q_t)
        Q_hat_list.append(Q_t_hat)
        update_store_info_CRPO(optimal_performance, V_r_list, V_g_list, cmdp.b, moving_avg_window,
                               t, full_average, avg_og_list, avg_cv_list)
        if t % 100 == 0:
            # saving result after every 100 iterations
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
    try_locally = True  # parameter to try code locally
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', help="iterations", type=int, default=200)
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

    # eta: slack used fo constraint violation in CRPO
    parser.add_argument('--eta', help="eta", type=float, default=0)

    # learning_rate of policy
    parser.add_argument('--lrp', help="learning rate policy", type=float, default=0.1)

    # changing MDP by selecting discount factor
    parser.add_argument('--gamma', help="discount factor", type=float, default=0.9)

    # changing MDP by selecting discount factor
    parser.add_argument('--b_thresh', help="cost threshold", type=float, default=1.5)

    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=10)

    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"

    print("eta:", args.eta)
    outer_file_name = "Iter" + str(args.num_iterations) + "_eta" + str(args.eta) + "_lr" + str(args.lrp) + "_MC" + str(
        int(args.multiple_constraints)) + "_FullAvg" + str(int(args.full_average)) + "_nsamples" + str(
        args.num_samples) + "_gam" + str(args.gamma) + "_b" + str(args.b_thresh)

    fold_name = os.path.join(save_dir_loc, "Results/Tabular/CRPO/ModelFree/TDSampling")
    output_dir_name = os.path.join(fold_name, outer_file_name)
    inner_file_name = "R" + str(args.run)
    output_dir = os.path.join(output_dir_name, inner_file_name)
    args.num_iterations = int(args.num_iterations)
    load_data_dir = output_dir
    args.multiple_constraints = bool(args.multiple_constraints)
    args.full_average = bool(args.full_average)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # creates a cmdp
    cmdp_env = TabularCMDP(add_constraints=True, multiple_constraints=args.multiple_constraints)
    cmdp_env.change_mdp(args.gamma, b=[args.b_thresh])
    optimal_policy, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=True)

    # setting discount factor here
    seed_val = 0
    rng = np.random.RandomState(seed_val + args.run)
    initial_policy = Policy(cmdp_env, rng)

    run_agent_params = {'cmdp': cmdp_env,
                        'optimal_performance': optimal_perf,
                        'num_iterations': args.num_iterations,
                        'output_dir': output_dir,
                        'moving_avg_window': args.moving_avg_window,
                        'full_average': args.full_average,
                        'initial_policy': initial_policy,
                        'eta': args.eta,
                        'learning_rate': args.lrp,
                        'nsamples': args.num_samples
                        }
    run_CRPO_agent(**run_agent_params)
