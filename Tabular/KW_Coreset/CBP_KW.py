from PainlessPolicyOpt.Tabular.tabularCMDPEnv import TabularCMDP
from PainlessPolicyOpt.Tabular.policy import *
from PainlessPolicyOpt.Tabular.helper_utlities import get_lambd_upper_bound, get_lower_upper_limit_lambd, \
    get_KW_coreset, get_optimal_pol_and_perf, save_data, update_store_info, store_noisy_estimator_data, \
    print_tc_features, time_log
from PainlessPolicyOpt.feature import TileCodingFeatures, TabularTileCoding
from PainlessPolicyOpt.LSTD import LSTD, KW_Sampling
import os
import argparse
import time
from datetime import timedelta
import copy

"""
Coin betting with sampling Q value using LSTD with d dim of features< |S X A| and policy update without storing the previous policy.
We will use tile coding for the same.

To get policy $\pi_t$ update, we need all \hat Q values from t \in {1... t-1}.
At each time step t, we store the Q values for all s \in S.
"""


class CB_Primal:
    """
    Paper: Coin Betting and Parameter-Free Online Learning, Orabona and Pal [2016] (https://arxiv.org/abs/1602.04128)
    Creates a CB algorithm to update the simplex policy using Learning with Expert Advice (LEA) based on KT potential,
    Algorithm 2, Pg 8
    """

    def __init__(self, env, T, feature, num_constraints, discount_factor, ul_lambd=0.0):
        self.t = 0
        self.A = env.A
        self.num_iter = T
        self.num_constraints = num_constraints
        self.feature = feature
        # weight matrix of Q_l until time t
        self.w_Q_until_t = []  # with dim of each element [T, num_features, num_constraints +1]
        # lambda vector until time t
        self.lambd_until_t = []
        self.init_policy = self.get_init_policy()
        self.grad_max_policy = self.get_max_grad_policy(discount_factor, ul_lambd)

    def get_init_policy(self):
        # returns a uniform policy
        return np.ones(self.A) / self.A

    def update_Q_weights(self, w_Q):
        """
        w_Q: weights of Q [feature size, num_constraints+1]
        """
        self.w_Q_until_t.append(w_Q)

    def get_max_grad_policy(self, gamma, ul_lambd):
        # To normalize the gradient of policy, value of max of grad of policy
        # Note: Assumption here is that 0<= reward, cost < =1.
        # If assumption change then multiply with max reward/cost
        gamma = 0.9  # keeping gamma value fixed for exp with varying gammas
        return (1 + np.sum(ul_lambd)) / (1 - gamma)

    def update_lambda_until_t(self, current_lambd=[]):
        lambd_t = [1]
        lambd_t.extend(current_lambd)
        self.lambd_until_t.append(lambd_t)

    def get_Q_l_until_time_t(self, state):
        """
        state: current state
        """
        T = len(self.lambd_until_t)
        q_l_until_t = np.zeros((T, self.A))
        # convert to two coordinate state representation for TC
        for t in range(T):
            for a in range(self.A):
                # summing all features and then multiplying with lambda to get Q_l
                q_l_until_t[t, a] = np.dot(
                    np.sum(np.asarray(
                        [self.w_Q_until_t[t][k] for k in self.feature.get_feature(state, a)]), axis=0),
                    self.lambd_until_t[t])
        return q_l_until_t

    def increment_t_counter(self):
        self.t += 1

    def policy_prob(self, state):
        """
        Use Coin betting simplex algorithm to get policy at a current iteration
        Parameter:
            state: state
        Returns:
            \pi_t(s): [num of actions] prob vector
        """
        if self.t == 0:
            return self.get_init_policy()
        # get q_l vector until time t which is normalized
        q_l_t = self.get_Q_l_until_time_t(state) / self.grad_max_policy
        # maintains sum of a*w until time t
        sum_a_w = np.ones(self.A)
        # maintains sum of a until time t
        sum_a_t = np.zeros(self.A)
        # w value at time t
        w_t = np.ones(self.A)
        # current pi_t
        pi_t = self.get_init_policy()
        for t in range(self.t):
            # advantage a at time t
            a_t = self.get_advantage_val(q_l_t[t], pi_t)
            for a in range(self.A):
                if w_t[a] <= 0:
                    a_t[a] = max(0, a_t[a])
            sum_a_t += a_t
            sum_a_w += np.multiply(a_t, w_t)
            w_t = np.multiply(sum_a_t, sum_a_w) / (t + 1)  # element-wise product
            p_t = np.multiply(self.init_policy, np.maximum(0, w_t))
            pi_t = self.get_normalized_p(p_t)
        return pi_t

    def get_advantage_val(self, q, action_prob):
        # Q(s,a) - V(s)
        return q - np.dot(q, action_prob)

    def get_normalized_p(self, p):
        p_norm = p.sum()  # 1 norm
        if p_norm == 0:
            return self.init_policy
        else:
            return p / p_norm


class CB_Dual:
    """
    Paper: Training Deep Networks without Learning Rates Through Coin Betting,
     Orabona [2016] (https://arxiv.org/abs/1705.07795)
    Creates a CB algorithm to update the dual variable \lambda using COCOB-backprop, Algorithm 2, Pg 6
    """

    def __init__(self, num_constraints, alpha, lower_limit_lambd, upper_limit_lambd):
        self.lower_limit_lambd = lower_limit_lambd
        self.upper_limit_lambd = upper_limit_lambd
        self.num_constraints = num_constraints
        self.curr_lambd = self.get_projected_lambd(np.ones(num_constraints))  # lambda intialised to 1
        self.initial_lambd = self.curr_lambd
        self.alpha_lambd = alpha
        self.G_lambd = np.zeros(num_constraints)  # G_t = G_{t-1} + |g_t|, where G_0 = L
        self.L_lambd = np.zeros(num_constraints)  # L_t = max(L_{t−1}, |g_t|)
        self.reward_lambd = np.zeros(num_constraints)  # Reward_t = Reward_{t-1} + (w_t - w_1)g_t, where R_0 = 0
        self.theta_lambd = np.zeros(num_constraints)  # theta_{t} = theta_{t-1} + g_t, where theta_0 = 0
        self.w_lambd = np.ones(num_constraints)  # w_t = w_1 + \beta(L + Reward_t), where w_1 = projected(1)

    def update(self, g):
        """
        Implements COCOB Algorithm 2 for updating lambda variable a d-dim vector
        Parameter:
            g: negative value of un-normalized gradient of the regret for dual variable
            Here,
            R(\lambda^*, T) = \sum_t <\lambda_t - \lambda^*, (\hat V_c - b)>,
            where V_c is cost value function and b is threshold for cost constraint.
        """
        self.L_lambd = np.maximum(self.L_lambd, np.abs(g))
        self.G_lambd += np.abs(g)
        self.reward_lambd = np.maximum(self.reward_lambd + (self.w_lambd - self.initial_lambd) * g, 0)
        self.theta_lambd += g
        beta = self.theta_lambd / (
                self.L_lambd * np.maximum(self.G_lambd + self.L_lambd, self.alpha_lambd * self.L_lambd))
        self.curr_lambd = self.get_projected_lambd(self.initial_lambd + beta * (self.L_lambd + self.reward_lambd))

    def get_projected_lambd(self, lambd):
        """
        Parameter:
            lambd
        Returns:
            projected value of lambd
        Here, lambd \in [lower_limit_lambd, upper_limit_lambd]
        """
        return np.maximum(self.lower_limit_lambd, np.minimum(self.upper_limit_lambd, lambd))


def get_policy_for_all_states(primal, S, A, tc_args):
    # helper function to get policy for all states
    policy_prob = np.zeros((S, A))
    for s in range(S):
        new_s_rep = tc_args.convert_state_to_two_coordinate(s)
        policy_prob[s] = primal.policy_prob(new_s_rep)
    return policy_prob


def run_CB_agent(cmdp, optimal_performance, num_iterations, output_dir, ub_lambd, moving_avg_window,
                 full_average, alpha_lambd, num_samples, feature, tc_args, Cindex_to_sa_pair,
                 update_dual_variable=True):
    # initialize storing data information
    og_list, avg_og_list, avg_cv_list, cv_list = [], [], [], []
    lambd_list, avg_lambd_list, V_r_list, V_g_list, policy_list, Q_l_list = [], [], [], [], [], []

    # noisy estimates
    V_r_hat_list, V_g_hat_list, Q_l_hat_list = [], [], []
    diff_in_Q_l, diff_in_Q_r, diff_in_Q_c = [], [], []

    curr_policy = Policy(cmdp, np.random.RandomState(0))
    ll_lambd, ul_lambd = get_lower_upper_limit_lambd(cmdp.num_constraints, ub_lambd, update_dual_variable)

    # CB Primal: to update the policy
    primal = CB_Primal(cmdp, num_iterations, feature, cmdp.num_constraints, cmdp.gamma, ul_lambd)

    # CB Dual: to update the dual variable lambd
    # [NOTE]: if mdp is passed then it creates a dummy dual object
    dual = CB_Dual(cmdp.num_constraints, alpha_lambd, ll_lambd, ul_lambd)

    # sampler for sampling the data
    data_sampler = KW_Sampling(cmdp, num_samples, tc_args, Cindex_to_sa_pair, update_dual_variable)

    # estimate Q hat from LSTD
    q_estimator = LSTD(feature.get_feature_size(), feature, cmdp, data_sampler)

    for t in range(num_iterations):
        # set current policy for evaluation of Q values
        curr_policy_prob = get_policy_for_all_states(primal, cmdp.S, cmdp.A, tc_args)
        curr_policy.set_policy_prob(curr_policy_prob)
        Q_r = curr_policy.get_Q_function(cmdp.R)
        Q_l = copy.deepcopy(Q_r)
        V_r = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_r))
        V_r_list.append(V_r)
        # Q estimator: update the weights of q_hat matrix
        q_estimator.run_solver(curr_policy_prob)
        q_hat = np.asarray([q_estimator.get_estimated_Q(tc_args.convert_state_to_two_coordinate(s)) for s in
                            range(cmdp.S)])  # dim [S X A X num_constraints+1]
        V_r_hat_list.append(np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, q_hat[:, :, 0])))
        Q_r_hat = q_hat[:, :, 0]
        diff_in_Q_r.append(q_estimator.get_error_norm(Q_r, Q_r_hat))
        Q_l_hat = copy.deepcopy(Q_r_hat)
        if update_dual_variable:
            curr_lambd = dual.curr_lambd
            lambd_list.append(curr_lambd)
            primal.update_lambda_until_t(curr_lambd)
            V_g_rho = np.zeros(cmdp.num_constraints)
            V_g_rho_hat = np.zeros(cmdp.num_constraints)
            error_in_cost_constraint = 0
            for c in range(cmdp.num_constraints):
                Q_c = curr_policy.get_Q_function(cmdp.G[c])
                Q_c_hat = q_hat[:, :, c + 1]
                V_g_rho[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c))
                V_g_rho_hat[c] = np.dot(cmdp.p0, np.einsum('sa,sa->s', curr_policy.policy_prob, Q_c_hat))
                Q_l += curr_lambd[c] * Q_c
                Q_l_hat += curr_lambd[c] * Q_c_hat
                error_in_cost_constraint += q_estimator.get_error_norm(Q_c, Q_c_hat)
            diff_in_Q_c.append(error_in_cost_constraint / cmdp.num_constraints)
            diff_in_Q_l.append(q_estimator.get_error_norm(Q_l, Q_l_hat))
            V_g_list.append(V_g_rho)
            V_g_hat_list.append(V_g_rho_hat)
            # update the dual variable lambda by passing the gradient of regret of lambda
            grad_lambd = -(V_g_rho_hat - cmdp.b)
            dual.update(grad_lambd)
            cv_list.append(cmdp.b - V_g_rho)  # calculates cv with true V_g
        else:
            primal.update_lambda_until_t()  # maintaining a dummy lambda variable
        # update the primal policy
        primal.update_Q_weights(q_estimator.weights)
        primal.increment_t_counter()
        og_list.append(optimal_performance - V_r)  # update og with true V_r
        policy_list.append(curr_policy.policy_prob)
        Q_l_list.append(Q_l)
        Q_l_hat_list.append(Q_l_hat)
        update_store_info(optimal_performance, V_r_list, V_g_list, cmdp.b, lambd_list, moving_avg_window,
                          t, update_dual_variable, full_average, avg_og_list, avg_cv_list, avg_lambd_list)
        if t % 100 == 0:
            # saving result after every 100 iterations
            save_data(output_dir, np.asarray(og_list), np.asarray(cv_list),
                      np.asarray(avg_og_list), np.asarray(avg_cv_list),
                      np.asarray(curr_policy.policy_prob),
                      np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(policy_list), np.asarray(Q_l_list))
            store_noisy_estimator_data(output_dir, diff_in_Q_r, diff_in_Q_c, diff_in_Q_l, V_g_hat_list, V_r_hat_list,
                                       V_g_list, V_r_list, Q_l_hat_list)
    save_data(output_dir, np.asarray(og_list), np.asarray(cv_list),
              np.asarray(avg_og_list), np.asarray(avg_cv_list), np.asarray(curr_policy.policy_prob),
              np.asarray(lambd_list), np.asarray(avg_lambd_list), np.asarray(policy_list), np.asarray(Q_l_list))
    store_noisy_estimator_data(output_dir, diff_in_Q_r, diff_in_Q_c, diff_in_Q_l, V_g_hat_list, V_r_hat_list,
                               V_g_list, V_r_list, Q_l_hat_list)


if __name__ == '__main__':
    start_main_t = time.time()
    try_locally = True  # parameter to try code locally
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iterations', help="iterations", type=int, default=1)
    # cmdp: 1 -> creates env with constraints and solve for Coin Betting on both policy and lambda
    # cmdp:0 -> creates a mdp and solve for Coin Betting on policy
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

    # alpha_lambd: parameter used for updating the lambda variable for cmdp
    parser.add_argument('--alpha_lambd', help="alpha COCOB", type=float, default=8)

    # iht table size
    parser.add_argument('--iht_size', help="iht size", type=float, default=100)

    # iht number of tiles
    parser.add_argument('--num_tiles', help="num of tiles", type=int, default=1)

    # iht number of tiles
    parser.add_argument('--tiling_size', help="dim of grid", type=int, default=14)

    # num of samples for estimating Q value function
    parser.add_argument('--num_samples', help="num of samples", type=int, default=1000)

    parser.add_argument('--gamma', help="discount factor", type=float, default=0.9)

    parser.add_argument('--b_thresh', help="threshold for single constraint", type=float, default=1.5)

    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"
    outer_file_name = "Iter" + str(args.num_iterations) + "_alpha" + str(args.alpha_lambd) + "_MC" + str(
        int(args.multiple_constraints)) + "_FullAvg" + str(int(args.full_average)) + "_iht_size" + str(
        int(args.iht_size)) + "_ntiles" + str(int(args.num_tiles)) + "_tileDim" + str(args.tiling_size) + \
                      "_num_samples" + str(args.num_samples) + "_gam" + str(args.gamma) + "_b" + str(args.b_thresh)
    fold_name = os.path.join(save_dir_loc, "Results/Tabular/CB/ModelFree/LFA/KW")
    output_dir_name = os.path.join(fold_name, outer_file_name)
    inner_file_name = "R" + str(args.run)
    output_dir = os.path.join(output_dir_name, inner_file_name)
    args.num_iterations = int(args.num_iterations)
    load_data_dir = output_dir
    args.multiple_constraints = bool(args.multiple_constraints)
    args.full_average = bool(args.full_average)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # creates a cmdp/mdp based on arguments
    if not args.cmdp:
        # creates a mdp without constraints
        cmdp_env = TabularCMDP(add_constraints=False)
        lambd_star_upper_bound = 0
        _, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=False)
    else:
        # creates a cmdp
        cmdp_env = TabularCMDP(add_constraints=True, multiple_constraints=args.multiple_constraints)
        lambd_star_upper_bound = get_lambd_upper_bound(args.multiple_constraints, cmdp_env.gamma, cmdp_env.b)
        _, optimal_perf, _ = get_optimal_pol_and_perf(cmdp_env, constraint=True)
    tabular_tc = TabularTileCoding(args.iht_size, args.num_tiles, args.tiling_size)
    tc_feature = TileCodingFeatures(cmdp_env.A, tabular_tc.get_tile_coding_args())
    print_tc_features(tc_feature, cmdp_env, tabular_tc, output_dir)
    #################################
    # get coreset index
    dir_KW_corset = os.path.join(save_dir_loc, "Results/Tabular/KW_Coreset/Tile_" + str(args.tiling_size))
    Cindex_to_list_s_a_dict = get_KW_coreset(dir_KW_corset)
    run_agent_params = {'cmdp': cmdp_env,
                        'optimal_performance': optimal_perf,
                        'num_iterations': args.num_iterations,
                        'output_dir': output_dir,
                        'ub_lambd': np.asarray(lambd_star_upper_bound),
                        'moving_avg_window': args.moving_avg_window,
                        'full_average': args.full_average,
                        'alpha_lambd': args.alpha_lambd,
                        'num_samples': args.num_samples,
                        'feature': tc_feature,
                        'tc_args': tabular_tc,
                        'Cindex_to_sa_pair': Cindex_to_list_s_a_dict,
                        'update_dual_variable': bool(args.cmdp)
                        }
    run_CB_agent(**run_agent_params)
    end_main_t = time.time()
    time_to_finish = timedelta(seconds=end_main_t - start_main_t)
    time_log(time_to_finish, args.num_iterations, output_dir)
