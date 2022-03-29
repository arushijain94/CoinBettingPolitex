import os
import sys
from PainlessPolicyOpt.plot_helper import *

"""
Plots the best hyperparameters of CBP, GDA, CRPO on Cartpole environment 
"""
nruns=5
max_len = 200
def plot_best_hyperparam(data_dir, base_data_dir, label_names):
    save_dir = os.path.join(base_data_dir, "Plots", "BestHyperParams")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    colors = ['r', 'b', 'g', 'y', 'o']
    line_width = [2,2,2]
    cv_eps= 6
    markers = ['o', 's', 'D', '*']
    entropy = [0.0, 0.1, 0.01, 0.001]
    for e in entropy:
        fig_og, ax_og = plt.subplots(figsize=(5, 5))
        fig_cv, ax_cv = plt.subplots(figsize=(5, 5))
        fig_cv1, ax_cv1 = plt.subplots(figsize=(5, 5))
        for i in range(len(data_dir)):
            path = os.path.join(base_data_dir, data_dir[i])
            exps = os.listdir(path)
            best_og = -sys.maxsize+1
            best_exp_name = "Nothing Satisfy!"
            og_to_print = [[],[]]
            cv_to_print = [[], []]
            cv1_to_print = [[], []]
            n_iter = 0
            flag_satisfy_eps = False
            for exp_num,exp in enumerate(exps):
                og_data = []
                cv_data = []
                cv_data1 = []
                if "Iter200_" not in exp or not exp.endswith("_Ent"+str(e)):
                    continue
                for r in range(1,nruns+1):
                    run_path = os.path.join(path, exp, "R" + str(r))
                    if not os.path.exists(run_path):
                        continue
                    data = get_data(run_path)
                    if not data:
                        continue
                    og, cv, cv1 = data[0], data[1], data[2]
                    if len(og)<max_len:
                        continue
                    og_data.append(og)
                    cv_data.append(cv)
                    cv_data1.append(cv1)
                if len(og_data) ==0:
                    continue
                og,ci_og,niter_og = rolling_mean_along_axis(np.asarray(og_data), W = 20)
                cv,ci_cv,niter_cv = rolling_mean_along_axis(np.asarray(cv_data), W=20)
                cv1, ci_cv1, niter_cv1 = rolling_mean_along_axis(np.asarray(cv_data1), W=20)
                plot_lines(og, ci_og, niter_og, colors[i], 1, ax_og)
                plot_lines(cv, ci_cv, niter_cv, colors[i], 1, ax_cv)
                plot_lines(cv1, ci_cv1, niter_cv1, colors[i], 1, ax_cv1)
                # finding the best hyperparameter
                best_og, best_exp_name, updated_flag = update_best_val(cv, cv1, og, cv_eps, best_og, best_exp_name, exp)
                if updated_flag:
                    og_to_print[0] = og
                    og_to_print[1] = ci_og
                    cv_to_print[0] = cv
                    cv_to_print[1] = ci_cv
                    cv1_to_print[0] = cv1
                    cv1_to_print[1] = ci_cv1
                    n_iter = niter_og
                    flag_satisfy_eps = True
            label_name = label_names[i]
            print("------------Algo:", label_name, "best exp:", best_exp_name, " entropy:", e)
            if flag_satisfy_eps:
                plot_lines(og_to_print[0], og_to_print[1], n_iter, colors[i], line_width[i], ax_og, label_name)
                plot_lines(cv_to_print[0], cv_to_print[1], n_iter, colors[i], line_width[i], ax_cv, label_name, cv=1)
                plot_lines(cv1_to_print[0], cv1_to_print[1], n_iter, colors[i], line_width[i], ax_cv1, label_name, cv=1)
            else:
                print("No error eps satisfy for:", label_name)
        set_label_save(fig_og, ax_og, "Return", save_dir, "BReturn_eps_"+str(cv_eps) + "_e"+str(e), y_lim=(30, 93))
        set_label_save(fig_cv, ax_cv, "Constraint violation 1", save_dir, "BCV1_eps_"+str(cv_eps)+ "_e"+str(e), y_lim=(-8,61))
        set_label_save(fig_cv1, ax_cv1, "Constraint violation 2", save_dir, "BCV2_eps_" + str(cv_eps)+ "_e"+str(e), y_lim=(-8,70))
        plt.close('all')

def update_best_val(cv, cv1, og, eps, best_og, best_exp_name, exp_name):
    num_last_iter = 30
    updated_flag=False
    if (-eps<=np.mean(cv[-num_last_iter:])<=0) and (-eps<=np.mean(cv1[-num_last_iter:])<=0):
        last_og = np.mean(og[-num_last_iter:])
        if last_og>best_og:
            best_og = last_og
            best_exp_name = exp_name
            updated_flag=True
    return best_og, best_exp_name, updated_flag

def set_label_save(fig, ax, y_label, save_dir, figname, y_lim=(None, None)):
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    ax.legend(loc='best', prop={"size": 10}, ncol=1)
    ax.set_ylim(y_lim)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, figname + ".png"), facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0)
    fig.clf()

def plot_lines(mean_val, ci, niter, color, line_width,ax, label_name="", alpha=0.2, cv=0):
    if len(label_name)!=0:
        ax.plot(np.arange(niter), mean_val, linestyle="-", linewidth=line_width, color=color, label=label_name)
    else:
        ax.plot(np.arange(niter), mean_val, linestyle="-", linewidth=line_width, color=color, alpha=alpha)
    ax.fill_between(np.arange(niter), (mean_val - ci), (mean_val + ci), color=color, alpha=.1)
    if cv ==1:
        ax.axhline(y=0, linestyle="--", color='black')

def get_data(loc):
    if not os.path.exists(os.path.join(loc, 'reward.npy')):
        return None
    reward = np.load(os.path.join(loc, 'reward.npy'))[:max_len]
    cv = np.load(os.path.join(loc, 'CV.npy'))[:max_len,0]
    cv1 = np.load(os.path.join(loc, 'CV.npy'))[:max_len,1]
    return reward, cv, cv1

if __name__ == '__main__':
    base_data_dir = "./Results/LFA/Cartpole/Constraints/"
    data_dir = ["CB", "GDA", "CRPO"]
    label_names = ["CBP", "GDA", "CRPO"]
    plot_best_hyperparam(data_dir, base_data_dir, label_names)
