import os
from Cartpole import *

def entropy_to_ubl_dict_cartpole(entropy_val):
    entropy_ubl_dict = {0.001:70, 0.01:25, 0.1:40, 0.0:20}
    return entropy_ubl_dict.get(entropy_val, 70)


def get_lambd_upper_bound(upper_bound=None):
    """
    return a hard coded upper bound of lambda
    """
    env = CartPoleEnvironment()
    num_constraint = env._num_constraints
    if upper_bound == None:
        return np.ones(num_constraint)*70
    else:
        return np.ones(num_constraint) * upper_bound


def get_lower_upper_limit_lambd(num_constraints, optimal_lambd, update_dual_variable):
    if not update_dual_variable:
        return 0, 0
    return np.zeros(num_constraints), 2 * optimal_lambd


def update_store_info(V_r_list, V_c_list, b, lambd_list, update_dual_variable, avg_reward_list, avg_cv_list,
                      avg_lambd_list):
    """
    appends new information to the store list
    """
    avg_reward_list.append(np.mean(V_r_list, axis=0))
    if update_dual_variable:
        # avg constraint violation = b - avg policy V_g
        avg_cv_list.append(b - np.mean(V_c_list, axis=0))
        avg_lambd_list.append(np.mean(lambd_list, axis=0))


def save_data(output_dir, reward, CV, reward_avg, CV_avg, lambda_t,
              lambda_avg_t, weight_Q):
    """
    saves data
    """
    np.save(os.path.join(output_dir, 'reward.npy'), reward)
    np.save(os.path.join(output_dir, 'CV.npy'), CV)
    np.save(os.path.join(output_dir, 'reward_avg.npy'), reward_avg)
    np.save(os.path.join(output_dir, 'CV_avg.npy'), CV_avg)
    np.save(os.path.join(output_dir, 'lambda.npy'), lambda_t)
    np.save(os.path.join(output_dir, 'lambda_avg.npy'), lambda_avg_t)
    np.save(os.path.join(output_dir, 'W_Q.npy'), weight_Q)

def time_log(time_to_finish, num_iteration, output_dir):
    file_name = os.path.join(output_dir, "time.txt")
    with open(file_name, 'w') as f:
        f.write('Time to finish code:%s\n' % (time_to_finish))
        f.write('Num of iterations:%s\n' % (num_iteration))
        print('Time to finish code:%s\n' % (time_to_finish))

def print_b_thresholds(env, output_dir):
    file_name = os.path.join(output_dir, "b.txt")
    with open(file_name, 'w') as f:
        f.write('b:%s\n' % (env.b))
