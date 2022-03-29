import os
from PainlessPolicyOpt.plot_helper import *

"""
Takes the result of model-based (true Q values) for CB and GDA,
plot the [avg_OG, avg_CV, avg_lambda] for different value of gam (0.7 0.8 0.9) with constant hyper parmeters 
"""
nruns = 5
max_data_length = 300


def plot_param_sensitivity(data_dir, base_data_dir):
    # assuming first dir is CB, second is GDA, third CRPO
    save_dir = os.path.join(base_data_dir, "Plots", "ModelBased", "DiscountFinal")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig_og, ax_og = plt.subplots(figsize=(5, 5))
    fig_cv, ax_cv = plt.subplots(figsize=(5, 5))
    colors = ['r', 'b', 'g', 'gray', 'm', 'orange', 'y', 'pink', 'lime', 'yellow', 'chocolate']
    marker = ['o', 's', 'D', '^', '*', 'x', 'p']
    linestyles = ['-', '-.', ':']
    line_width = [2, 1, 1]
    algo_name = ["CBP", "GDA", "CRPO"]
    gam = [0.7, 0.8]
    for g_ind, gam_val in enumerate(gam):
        for i in range(len(data_dir)):
            path = os.path.join(base_data_dir, data_dir[i])
            exps = os.listdir(path)
            num_exp = len(exps)
            for exp_num, exp in enumerate(exps):
                if "_gam" + str(gam_val) + "_" not in exp:
                    continue
                print(exp)
                og_data = []
                cv_data = []
                exp_name = exp.split("_gam")[1]
                print(exp_name)
                for r in range(1, nruns + 1):
                    run_path = os.path.join(path, exp, "R" + str(r))
                    if not os.path.exists(run_path):
                        continue
                    data = get_data(run_path)
                    if not data:
                        continue
                    og, cv = data[0], data[1]
                    og_data.append(og)
                    cv_data.append(cv)
                    print(run_path)
                og, ci_og, niter_og = rolling_mean_along_axis(np.asarray(og_data), W=3)
                cv, ci_cv, niter_cv = rolling_mean_along_axis(np.asarray(cv_data), W=3)
                label_name = r'$\gamma=$' + str(gam_val) + " " + algo_name[i]
                plot_lines(og, ci_og, niter_og, colors[i], line_width[i], ax_og, linestyles[i],
                           label_name=label_name, marker=marker[g_ind])
                plot_lines(cv, ci_cv, niter_cv, colors[i], line_width[i], ax_cv, linestyles[i],
                           label_name=label_name, marker=marker[g_ind])
    set_label_save(fig_og, ax_og, "Optimality gap", save_dir, "OG", y_lim=(-0.02, 0.3))
    set_label_save(fig_cv, ax_cv, "Constraint violation", save_dir, "CV", y_lim=(-0.25, 0.3))
    plt.close('all')


def set_label_save(fig, ax, y_label, save_dir, figname, y_lim=(None, None)):
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    ax.legend(loc='best', prop={"size": 10}, ncol=2)
    ax.set_ylim(y_lim)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, figname + ".png"), facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0)
    fig.clf()


def plot_lines(mean_val, ci, niter, color, line_width, ax, line_style="-", label_name="", marker=""):
    if len(label_name):
        ax.plot(np.arange(niter), mean_val, linestyle=line_style, linewidth=line_width, color=color, label=label_name,
                marker=marker, markevery=30)
    else:
        ax.plot(np.arange(niter), mean_val, linestyle=line_style, linewidth=line_width, color=color)
    ax.fill_between(np.arange(niter), (mean_val - ci), (mean_val + ci), color=color, alpha=.1)
    ax.axhline(y=0, linestyle="--", color='black')


def get_data(loc):
    if not os.path.exists(os.path.join(loc, 'OG_avg.npy')):
        return None
    og_avg = np.load(os.path.join(loc, 'OG_avg.npy'))[:max_data_length]
    cv_avg = np.mean(np.load(os.path.join(loc, 'CV_avg.npy')), axis=1)[
             :max_data_length]  # taking mean of cv across num of constraints
    return og_avg, cv_avg


if __name__ == '__main__':
    local_try = True
    if not local_try:
        base_data_dir = "/network/scratch/j/jainarus/CMDPData/Results/Tabular"
    else:
        base_data_dir = "./Results/Tabular"
    data_dir = ["CB/ModelBased/Discount", "GDA/ModelBased/Discount", "CRPO/ModelBased/Discount"]
    plot_param_sensitivity(data_dir, base_data_dir)
