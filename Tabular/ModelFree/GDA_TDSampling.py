from PainlessPolicyOpt.Tabular.tabularCMDPEnv import TabularCMDP
from PainlessPolicyOpt.Tabular.policy import *
from PainlessPolicyOpt.Tabular.helper_utlities import get_lambd_upper_bound, get_lower_upper_limit_lambd, \
    get_optimal_pol_and_perf, save_data, update_store_info, store_noisy_estimator_data
from TD import *
import os
import argparse


"""
GDA: Gradient Descent and Ascent implementation in model-free setting with with TD Sampling
"""

class GDA_Primal:
    """
    Primal provides implementation to update the policy using OMA.
    """

    def __init__(self, learning_rate, policy):
        """
        Parameter:
            learning_rate: alpha
            policy: policy class object
        """
        self.learning_rate = learning_rate
        self.policy = policy

    def update(self, Q):
        """
        policy update -> pi_{t+1} = pi_{t} exp(alpha * Q)
        Parameter:
            Q: [S x A] dim state-action value function
        """
        new_policy_prob = self.policy.policy_prob * np.exp(self.learning_rate * Q)
        denom = np.sum(new_policy_prob, axis=1)
        for s in range(self.policy.policy_prob.shape[0]):
            new_policy_prob[s, :] /= denom[s]  # normalizing policy prob
        self.policy.policy_prob = new_policy_prob


class GDA_Dual:
    """
    Dual: Provides implementation to update dual variable lambda using Gradient Descent.
    """

    def __init__(self, learning_rate, b, lower_limit, upper_limit, num_constraints):
        """
        Parameter:
          b: constraint threshold
          lower_limit: lower limit for projection
          upper_limit: upper limit for projection
          learning_rate: learning rate for dual update
          lambd: initial value of dual variable lambd
        """
        self.b = b
        self.learning_rate = learning_rate
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.lambd = np.ones(num_constraints)  # initial lambd value

    def get_projected_value(self, x):
        return np.maximum(self.lower_limit, np.minimum(x, self.upper_limit))

    def update(self, value_utility):
        """
        lambda update -> lambda_{t+1} = P_[lower, upper](lambda_{t} - alpha*(value_utility - b))
        """
        self.lambd = self.get_projected_value(self.lambd - self.learning_rate * (value_utility - self.b))


def run_GDA_agent(cmdp, optimal_performance, learning_rate_pol, learning_rate_lambd,
                  num_iterations, output_dir, ub_lambd, moving_avg_window, full_average,
                  initial_policy, nsamples, update_dual_variable=True):
    # initialize storing data information
    og_list, avg_og_list, cv_list, avg_cv_list = [], [], [], []
    lambd_list, avg_lambd_list, V_r_list, V_g_list, policy_list = [], [], [], [], []
    Q_l_list = []

    # noisy estimates
    V_r_hat_list, V_g_hat_list = [], []
    diff_in_Q_l, diff_in_Q_r, diff_in_Q_c = [], [], []
    Q_l_hat_list = []

    ll_lambd, ul_lambd = get_lower_upper_limit_lambd(cmdp.num_constraints, ub_lambd, update_dual_variable)

    # Primal update
    primal = GDA_Primal(learning_rate_pol, initial_policy)

    # Dual update
    # [NOTE]: If update_dual_variable is false, it creates a dummy dual object, but doesn't use it to update policy.
    dual = GDA_Dual(learning_rate_lambd, cmdp.b, ll_lambd, ul_lambd, cmdp.num_constraints)

    # TD Q estimator
    q_estimator = TD(cmdp, nsamples, cmdp.num_constraints, update_dual_variable)

    for t in range(num_iterations):
        curr_policy = primal.policy
        Q_r = curr_policy.get_Q_function(cmdp.R)
        Q_l = Q_r
        policy_list.append(curr_policy.policy_prob)
        V_r = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r))
        V_r_list.append(V_r)
        og_list.append(optimal_performance - V_r)

        # get MC estimates for Q value
        q_estimator.set_policy_prob(curr_policy.policy_prob)
        Q_hat = q_estimator.get_Q()  # returns a num_constraints+1 size array

        Q_l_hat = Q_hat[0]
        V_r_hat = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_l_hat))  # noisy V_r estimate
        V_r_hat_list.append(V_r_hat)
        diff_in_Q_r.append(q_estimator.get_error_norm(Q_l, Q_l_hat))  # here Q_l contains Q_r value

        if update_dual_variable:
            curr_lambd = dual.lambd
            V_g_rho = np.zeros(cmdp.num_constraints)
            V_g_rho_hat = np.zeros(cmdp.num_constraints)
            error_in_cost_constraint = 0
            for c in range(cmdp.num_constraints):
                Q_c = curr_policy.get_Q_function(cmdp.G[c])
                Q_c_hat = Q_hat[c + 1]
                V_g_rho[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c))
                V_g_rho_hat[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c_hat))
                Q_l += curr_lambd[c] * Q_c
                Q_l_hat += curr_lambd[c] * Q_c_hat
                error_in_cost_constraint += q_estimator.get_error_norm(Q_c, Q_c_hat)
            diff_in_Q_c.append(error_in_cost_constraint / cmdp.num_constraints)
            diff_in_Q_l.append(q_estimator.get_error_norm(Q_l, Q_l_hat))
            lambd_list.append(curr_lambd)
            V_g_list.append(V_g_rho)
            V_g_hat_list.append(V_g_rho_hat)
            cv_list.append(cmdp.b - V_g_rho)
            # update dual variable
            dual.update(V_g_rho_hat)

        # update primal policy
        primal.update(Q_l_hat)
        Q_l_list.append(Q_l)
        Q_l_hat_list.append(Q_l_hat)
        update_store_info(optimal_performance, V_r_list, V_g_list, cmdp.b, lambd_list, moving_avg_window,
                          t, update_dual_variable, full_average, avg_og_list, avg_cv_list, avg_lambd_list)
        if t % 100 == 0:
            save_data(output_dir, np.asarray(og_list), np.asarray(cv_list),
                      np.asarray(avg_og_list), np.asarray(avg_cv_list),
                      np.asarray(curr_policy.policy_prob),
                      np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(policy_list), np.asarray(Q_l_list))
            store_noisy_estimator_data(output_dir, diff_in_Q_r, diff_in_Q_c, diff_in_Q_l, V_g_hat_list, V_r_hat_list,
                                       V_g_list, V_r_list, Q_l_hat_list)
    save_data(output_dir, np.asarray(og_list), np.asarray(cv_list),
              np.asarray(avg_og_list), np.asarray(avg_cv_list),
              np.asarray(curr_policy.policy_prob),
              np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(policy_list), np.asarray(Q_l_list))
    store_noisy_estimator_data(output_dir, diff_in_Q_r, diff_in_Q_c, diff_in_Q_l, V_g_hat_list, V_r_hat_list,
                               V_g_list, V_r_list, Q_l_hat_list)


if __name__ == '__main__':
    try_locally = True  # variable to be set True when running on local machine
    parser = argparse.ArgumentParser()
    # by default value is set to the optimal learning rates for policy and lambda
    parser.add_argument('--learning_rate_pol', type=float, default=1.0)
    parser.add_argument('--learning_rate_lambd', type=float, default=0.1)

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
    # -------
    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=100)

    # changing MDP by selecting discount factor
    parser.add_argument('--gamma', help="discount factor", type=float, default=0.9)

    parser.add_argument('--b_thresh', help="threshold for single constraint", type=float, default=1.5)

    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"
    outer_file_name = "LRP" + str(args.learning_rate_pol) + "_LRL" + str(args.learning_rate_lambd) \
                      + "_Iter" + str(args.num_iterations) + "_MC" + str(int(args.multiple_constraints)) \
                      + "_FullAvg" + str(int(args.full_average)) + "_nsamples" + str(args.num_samples) \
                      + "_gam" + str(args.gamma) + "_b" + str(args.b_thresh)

    fold_name = os.path.join(save_dir_loc, "Results/Tabular/GDA/ModelFree/TDSampling")
    output_dir_name = os.path.join(fold_name, outer_file_name)
    inner_file_name = "R" + str(args.run)
    output_dir = os.path.join(output_dir_name, inner_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.multiple_constraints = bool(args.multiple_constraints)
    args.full_average = bool(args.full_average)

    # creates a cmdp/mdp based on arguments
    if not args.cmdp:
        # creates a mdp without constraints
        cmdp_env = TabularCMDP(add_constraints=False)
        lambd_star_upper_bound = 0
        _, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=False)
    else:
        # creates a cmdp
        cmdp_env = TabularCMDP(add_constraints=True, multiple_constraints=args.multiple_constraints)
        cmdp_env.change_mdp(args.gamma, [args.b_thresh])
        lambd_star_upper_bound = get_lambd_upper_bound(args.multiple_constraints, cmdp_env.gamma, cmdp_env.b)
        optimal_policy, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=True)

    seed_val = 0
    rng = np.random.RandomState(seed_val + args.run)
    initial_policy = Policy(cmdp_env, rng)
    run_agent_params = {'cmdp': cmdp_env,
                        'optimal_performance': optimal_perf,
                        'learning_rate_pol': args.learning_rate_pol,
                        'learning_rate_lambd': args.learning_rate_lambd,
                        'num_iterations': args.num_iterations,
                        'output_dir': output_dir,
                        'ub_lambd': np.asarray(lambd_star_upper_bound),
                        'moving_avg_window': args.moving_avg_window,
                        'full_average': args.full_average,
                        'initial_policy': initial_policy,
                        'nsamples': args.num_samples,
                        'update_dual_variable': bool(args.cmdp)
                        }
    run_GDA_agent(**run_agent_params)
