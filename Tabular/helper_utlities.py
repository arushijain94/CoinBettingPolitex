import os
from policy import Policy
from PainlessPolicyOpt.Tabular.tabularCMDPEnv import *
import _pickle as cPickle

def get_lambd_upper_bound(multiple_constraints, gamma, b=[1.5]):
    """
    Solve \tilde\mu = max_{\mu} <\mu, g>
    delta = <\tilde\mu, g> - b
    lambda^* <= 1/[(1-gamma)* delta]
    """
    cmdp_initialize = TabularCMDP(add_constraints=True, multiple_constraints=multiple_constraints)  # creates a mdp
    cmdp_initialize.gamma = gamma
    cmdp_initialize.b =  b
    lambda_ub = []
    for i in range(cmdp_initialize.num_constraints):
        cmdp_initialize.R = cmdp_initialize.G[i]  # changing the reward function to cost function
        _, _, tilde_mu = get_optimal_pol_and_perf(cmdp_initialize, constraint=False)
        delta = np.multiply(tilde_mu, cmdp_initialize.G[i]).sum() - cmdp_initialize.b[i]
        lambda_ub.append(1 / ((1 - cmdp_initialize.gamma) * delta))
    return np.asarray(lambda_ub)


def get_lower_upper_limit_lambd(num_constraints, optimal_lambd, update_dual_variable):
    if not update_dual_variable:
        return 0, 0
    return np.zeros(num_constraints), 2 * optimal_lambd

def get_optimal_pol_and_perf(cmdp, constraint=True):
    """
    calculates the optimal policy and performance given the cmdp model
    """
    optimal_policy_prob, mu_star = cmdp.solve_primal_LP(constraint=constraint)
    policy = Policy(cmdp, np.random.RandomState(0))
    policy.set_policy_prob(optimal_policy_prob)
    optimal_performance = np.round(cmdp.p0 @ policy.get_V_function(cmdp.R), 3)
    return policy, optimal_performance, mu_star


def save_data(output_dir, OG, CV, OG_avg, CV_avg, opt_pol_prob, lambda_t,
              lambda_avg_t, policy_t, Q_l):
    """
    saves data
    """
    np.save(os.path.join(output_dir, 'OG.npy'), OG)
    np.save(os.path.join(output_dir, 'CV.npy'), CV)
    np.save(os.path.join(output_dir, 'OG_avg.npy'), OG_avg)
    np.save(os.path.join(output_dir, 'CV_avg.npy'), CV_avg)
    np.save(os.path.join(output_dir, 'optimal_cmdp_policy.npy'), opt_pol_prob)
    np.save(os.path.join(output_dir, 'lambda.npy'), lambda_t)
    np.save(os.path.join(output_dir, 'lambda_avg.npy'), lambda_avg_t)
    np.save(os.path.join(output_dir, 'policy_prob.npy'), policy_t)
    np.save(os.path.join(output_dir, 'Q_l.npy'), Q_l)


def save_data_CRPO(output_dir, OG, CV, OG_avg, CV_avg, opt_pol_prob, policy_t, Q):
    """
    saves data
    """
    np.save(os.path.join(output_dir, 'OG.npy'), OG)
    np.save(os.path.join(output_dir, 'CV.npy'), CV)
    np.save(os.path.join(output_dir, 'OG_avg.npy'), OG_avg)
    np.save(os.path.join(output_dir, 'CV_avg.npy'), CV_avg)
    np.save(os.path.join(output_dir, 'optimal_cmdp_policy.npy'), opt_pol_prob)
    np.save(os.path.join(output_dir, 'policy_prob.npy'), policy_t)
    np.save(os.path.join(output_dir, 'Q.npy'), Q)


def update_store_info(optimal_performance, V_r_list, V_g_list, b,
                      lambd_list, moving_avg_window, curr_t, update_dual_variable,
                      full_average, avg_og_list, avg_cv_list, avg_lambd_list):
    """
    appends new information to the store list
    """
    if full_average == True:
        # avg optimality gap = optimal perf - avg policy V_r (perf)
        avg_og_list.append(optimal_performance - np.mean(V_r_list, axis=0))
        if update_dual_variable:
            # avg constraint violation = b - avg policy V_g
            avg_cv_list.append(b - np.mean(V_g_list, axis=0))
            avg_lambd_list.append(np.mean(lambd_list, axis=0))
    elif curr_t > moving_avg_window:
        # Tail Averaging with window size as moving_avg_window
        # tail avg optimality gap = optimal perf - tail averaged policy V_r (perf)
        avg_og_list.append(optimal_performance - np.mean(V_r_list[-moving_avg_window:], axis=0))
        if update_dual_variable:
            # tail avg constraint violation = b - tail avg policy V_g
            avg_cv_list.append(b - np.mean(V_g_list[-moving_avg_window:], axis=0))
            avg_lambd_list.append((np.mean(lambd_list[-moving_avg_window:], axis=0)))

def update_store_info_CRPO(optimal_performance, V_r_list, V_g_list, b,
                           moving_avg_window, curr_t, full_average, avg_og_list,
                           avg_cv_list):
    """
    appends new information to the store list for CRPO
    """
    if full_average == True:
        # avg optimality gap = optimal perf - avg policy V_r (perf)
        avg_og_list.append(optimal_performance - np.mean(V_r_list, axis=0))
        avg_cv_list.append(b - np.mean(V_g_list, axis=0))
    elif curr_t > moving_avg_window:
        # Tail Averaging with window size as moving_avg_window
        # tail avg optimality gap = optimal perf - tail averaged policy V_r (perf)
        avg_og_list.append(optimal_performance - np.mean(V_r_list[-moving_avg_window:], axis=0))
        avg_cv_list.append(b - np.mean(V_g_list[-moving_avg_window:], axis=0))


def store_noisy_estimator_data(output_dir, diff_Qr, diff_Qc, diff_Ql, V_g_hat, V_r_hat, V_g, V_r, Q_l_hat):
    """
    store noisy estimator information
    """
    np.save(os.path.join(output_dir, 'diff_Qr.npy'), np.asarray(diff_Qr))
    np.save(os.path.join(output_dir, 'diff_Qc.npy'), np.asarray(diff_Qc))
    np.save(os.path.join(output_dir, 'diff_Ql.npy'), np.asarray(diff_Ql))
    np.save(os.path.join(output_dir, 'V_g_hat.npy'), np.asarray(V_g_hat))
    np.save(os.path.join(output_dir, 'V_r_hat.npy'), np.asarray(V_r_hat))
    np.save(os.path.join(output_dir, 'V_g.npy'), np.asarray(V_g))
    np.save(os.path.join(output_dir, 'V_r.npy'), np.asarray(V_r))
    np.save(os.path.join(output_dir, 'Q_l_hat.npy'), np.asarray(Q_l_hat))

def store_noisy_estimator_data_CRPO(output_dir, diff_Qr, diff_Qc, V_g_hat, V_r_hat, V_g, V_r, Q_hat):
    """
    store noisy estimator information
    """
    np.save(os.path.join(output_dir, 'diff_Qr.npy'), np.asarray(diff_Qr))
    np.save(os.path.join(output_dir, 'diff_Qc.npy'), np.asarray(diff_Qc))
    np.save(os.path.join(output_dir, 'V_g_hat.npy'), np.asarray(V_g_hat))
    np.save(os.path.join(output_dir, 'V_r_hat.npy'), np.asarray(V_r_hat))
    np.save(os.path.join(output_dir, 'V_g.npy'), np.asarray(V_g))
    np.save(os.path.join(output_dir, 'V_r.npy'), np.asarray(V_r))
    np.save(os.path.join(output_dir, 'Q_hat.npy'), np.asarray(Q_hat))

# tabular experiment functions
def print_tc_features(tc_feature, cmdp, tc_args, output_dir):
    feature_file_name = os.path.join(output_dir, "tc_features.txt")
    with open(feature_file_name, 'w') as f:
        f.write('---Arguments to TC\n')
        for k, v in tc_args.get_tile_coding_args().items():
            f.write('%s:%s\n' % (k, v))
        list_tc_feature = []
        for s in range(cmdp.S):
            for a in range(cmdp.A):
                new_s = tc_args.convert_state_to_two_coordinate(s)
                line = str((s, a)) + ":" + str(tc_feature.get_feature(new_s, a))
                list_tc_feature.append(tc_feature.get_feature(new_s, a)[0])
                # print(line)
                f.write(line)
                f.write('\n')
        unique_features = len(set(list_tc_feature))
        original_s_a = cmdp.S * cmdp.A
        overlap_feature = (original_s_a - unique_features) / original_s_a * 100
        f.write('Percentage of features overlaped: %s\n' % (overlap_feature))
        print("percentage of overlap", overlap_feature)


def time_log(time_to_finish, num_iteration, output_dir):
    file_name = os.path.join(output_dir, "time.txt")
    with open(file_name, 'w') as f:
        f.write('Time to finish code:%s\n' % (time_to_finish))
        f.write('Num of iterations:%s\n' % (num_iteration))
        print("time to finish:", time_to_finish)


def get_KW_coreset(read_dir):
    """
    read coreset index from the file
    """
    file_name = os.path.join(read_dir, "Cindex_to_list_s_a_dict.pickle")
    with open(file_name, 'rb') as fp:
        Cindex_to_list_s_a_dict = cPickle.load(fp)
    return Cindex_to_list_s_a_dict


