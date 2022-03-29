import os
import sys
from PainlessPolicyOpt.plot_helper import *

"""
Takes the result of model-based (true Q values) for CB and GDA,
plot the [avg_OG, avg_CV, avg_lambda] for diff values of: alpha (CB); LR of policy and lambda (GDA)   
"""
nruns=5
def plot_best_hyperparam(data_dir, base_data_dir, label_names):
    save_dir = os.path.join(base_data_dir, "Plots", "ModelBased", "BestHyperParam")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    colors = ['r', 'b', 'g', 'y', 'o']
    line_width = [3,3,3]
    eps_list = [0.25]
    for eps in eps_list:
        fig_og, ax_og = plt.subplots(figsize=(5, 5))
        fig_cv, ax_cv = plt.subplots(figsize=(5, 5))
        for i in range(len(data_dir)):
            path = os.path.join(base_data_dir, data_dir[i])
            exps = os.listdir(path)
            num_exp = len(exps)
            best_og = sys.maxsize
            best_exp_name = "Nothing Satisfy!"
            og_to_print = [[],[]]
            cv_to_print = [[], []]
            n_iter = 0
            flag_satisfy_eps = False
            for exp_num,exp in enumerate(exps):
                og_data = []
                cv_data = []
                if "Iter1500" not in exp:
                    continue
                for r in range(1,nruns+1):
                    run_path = os.path.join(path, exp, "R" + str(r))
                    if not os.path.exists(run_path):
                        continue
                    data = get_data(run_path)
                    if not data:
                        continue
                    og, cv = data[0], data[1]
                    og_data.append(og)
                    cv_data.append(cv)
                og,ci_og,niter_og = rolling_mean_along_axis(np.asarray(og_data), W = 20)
                cv,ci_cv,niter_cv = rolling_mean_along_axis(np.asarray(cv_data), W=20)
                plot_lines(og, ci_og, niter_og, colors[i], 1, ax_og, alpha=0.25, label_name="")
                plot_lines(cv, ci_cv, niter_cv, colors[i], 1, ax_cv, alpha=0.25, label_name="")
                best_og, best_exp_name, updated_flag = update_best_val(cv, og, eps, best_og, best_exp_name, exp)
                if updated_flag:
                    og_to_print[0] = og
                    og_to_print[1] = ci_og
                    cv_to_print[0] = cv
                    cv_to_print[1] = ci_cv
                    n_iter = niter_og
                    flag_satisfy_eps = True
            label_name = label_names[i]
            print("*************Algo:", label_name, "best exp:", best_exp_name)
            if flag_satisfy_eps:
                plot_lines(og_to_print[0], og_to_print[1], n_iter, colors[i], line_width[i], ax_og, 1, label_name)
                plot_lines(cv_to_print[0], cv_to_print[1], n_iter, colors[i], line_width[i], ax_cv, 1, label_name)
            else:
                print("No error eps satisfy for:", label_name)
        set_label_save(fig_og, ax_og, "Optimality gap", save_dir, "OG_eps_"+str(eps), y_lim=(-0.05,1.5))
        set_label_save(fig_cv, ax_cv, "Constraint violation", save_dir, "CV_eps_"+str(eps), y_lim=(-0.75,1.4))
    plt.close('all')

def update_best_val(cv, og, eps, best_og, best_exp_name, exp_name):
    num_last_iter = 100
    updated_flag=False
    if -eps<=np.mean(cv[-num_last_iter:])<=0:
        last_og = np.mean(og[-num_last_iter:])
        if best_og>last_og:
            best_og = last_og
            best_exp_name = exp_name
            updated_flag=True
    return best_og, best_exp_name, updated_flag


def set_label_save(fig, ax, y_label, save_dir, figname, y_lim=(None,None)):
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    ax.set_ylim(y_lim)
    ax.legend(loc='upper right', prop={"size": 10}, ncol=1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, figname + ".png"), facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0)
    fig.clf()

def plot_lines(mean_val, ci, niter, color, line_width,ax, alpha, label_name=""):
    if len(label_name)==0:
        ax.plot(np.arange(niter), mean_val, linestyle="-", linewidth=line_width, color=color, alpha=alpha)
    else:
        ax.plot(np.arange(niter), mean_val, linestyle="-", linewidth=line_width, color=color, label = label_name, alpha=alpha)
    ax.fill_between(np.arange(niter), (mean_val - ci), (mean_val + ci), color=color, alpha=.05)
    ax.axhline(y=0, linestyle="--", color='black')

def get_data(loc):
    if not os.path.exists(os.path.join(loc, 'OG_avg.npy')):
        return None
    og_avg = np.load(os.path.join(loc, 'OG_avg.npy'))
    cv_avg = np.mean(np.load(os.path.join(loc, 'CV_avg.npy')), axis=1) # taking mean of cv across num of constraints
    return og_avg, cv_avg

if __name__ == '__main__':
    local_try=True
    if not local_try:
        base_data_dir = "/network/scratch/j/jainarus/CMDPData/Results/Tabular"
    else:
        base_data_dir = "./Results/Tabular"

    data_dir = ["CB/ModelBased", "GDA/ModelBased", "CRPO/ModelBased"]
    label_names = ["CBP", "GDA", "CRPO"]
    plot_best_hyperparam(data_dir, base_data_dir, label_names)
