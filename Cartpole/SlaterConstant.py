from PainlessPolicyOpt.feature import CartPoleTileCoding, TileCodingFeatures
from PainlessPolicyOpt.LSTD import LFASampling, LSTD
import os
import argparse
from Cartpole import *
from CB import CB_Primal


def get_slater_constant(env, num_iter, feature, num_traj_sampling, entropy_coeff, output_dir, gamma, constraint_ind):
    """
    solve for \mu* = \max_{\mu} <\mu^T, c>
    """
    cv = []
    ub_lambda = []
    cb_primal = CB_Primal(env=env, T=num_iter, feature=feature, num_constraints=0, discount_factor=env.gamma,
                          entropy_coeff=entropy_coeff, ul_lambd=0.0)
    sampler = LFASampling(env, env._num_constraints, num_traj_sampling, 0)
    # estimate Q hat from LSTD
    q_estimator = LSTD(feature.get_feature_size(), feature, env, sampler)
    slater_constant = 1.0
    for t in range(num_iter):
        w_c = q_estimator.run_solver(cb_primal)[:, constraint_ind + 1]  # get only cost weights for Q matrix
        cb_primal.update_lambda_until_t()
        cb_primal.update_Q_weights(w_c)
        cb_primal.increment_t_counter()
        v_c = sampler.get_V_c()
        slater_constant = v_c - env.b
        ub_lambd_val = 1.0 / ((1 - gamma) * slater_constant)
        cv.append(env.b - v_c)
        ub_lambda.append(ub_lambd_val)
        if t % 10 == 0:
            np.save(os.path.join(output_dir, "cv.npy"), np.asarray(cv))
            np.save(os.path.join(output_dir, "ub_lambd.npy"), np.asarray(ub_lambda))
    np.save(os.path.join(output_dir, "cv.npy"), np.asarray(cv))
    np.save(os.path.join(output_dir, "ub_lambd.npy"), np.asarray(ub_lambda))
    return ub_lambd_val


if __name__ == '__main__':
    try_locally = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tilings', type=int, default=8)
    parser.add_argument('--solve_constraint', type=int, default=0)
    parser.add_argument('--num_iter', help="number of iterations for updating policy", type=int, default=120)
    parser.add_argument('--entropy_coeff', help="entropy coefficient", type=float, default=1e-2)
    parser.add_argument('--num_traj_sampling', help="number of trajectories for sampling data", type=int, default=50)
    args = parser.parse_args()
    if try_locally:
        save_dir_loc = "./"
    else:
        save_dir_loc = "/network/scratch/j/jainarus/CMDPData/"

    outer_file_name = "Constraint" + str(args.solve_constraint) + "_Tile" + str(args.num_tilings) + "_Ent" + str(
        args.entropy_coeff)
    env = CartPoleEnvironment()
    output_dir = os.path.join(save_dir_loc, "Results/LFA/Cartpole/Constraints/SC", outer_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # cartpole environment
    cp_tc = CartPoleTileCoding(num_tilings=args.num_tilings)
    tc_feature = TileCodingFeatures(env.A, cp_tc.get_tile_coding_args())
    # solve for upper bound \lambda* by computing the slater constant
    get_slater_constant(env, args.num_iter, tc_feature, args.num_traj_sampling, args.entropy_coeff, output_dir,
                        env.gamma, args.solve_constraint)
