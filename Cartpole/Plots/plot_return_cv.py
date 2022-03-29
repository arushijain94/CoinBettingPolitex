import os
from PainlessPolicyOpt.plot_helper import *

"""
Takes the result of Cartpole for CRPO, GDA, CRPO
and plot return, cv. 
"""
nruns = 5
max_data_length = 200


def plot_param_sensitivity(data_dir, base_data_dir):
    colors = ['r', 'b', 'g', 'y', 'orange', 'chocolate', 'grey', 'magenta', 'brown', 'gold', 'teal', 'pink', 'thistle',
              'plum', 'cyan', 'olive']
    line_width = [2, 1, 1]
    entropy = [0.0, 0.1, 0.01, 0.001]  # , 0.0]
    line_styles = ["-", "--", ":"]
    markers = ['o', 's', 'D', '*']
    algo_name = ["CB", "GDA", "CRPO"]
    for e in entropy:
        save_dir = os.path.join(base_data_dir, "Plots", data_dir[0])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig_og, ax_og = plt.subplots(figsize=(5, 5))
        fig_cv, ax_cv = plt.subplots(figsize=(5, 5))
        fig_cv1, ax_cv1 = plt.subplots(figsize=(5, 5))
        ind = 0
        for i in range(len(data_dir)):
            path = os.path.join(base_data_dir, data_dir[i])
            exps = os.listdir(path)
            for exp_num, exp in enumerate(exps):
                og_data = []
                cv_data = []
                cv_data1 = []
                if "Iter200_" not in exp or not exp.endswith("_Ent" + str(e)) or "_num_samples50" not in exp:
                    continue
                for r in range(1, nruns + 1):
                    run_path = os.path.join(path, exp, "R" + str(r))
                    if not os.path.exists(run_path):
                        continue
                    data = get_data(run_path)
                    if not data:
                        continue
                    og, cv, cv1 = data[0], data[1], data[2]
                    if len(og) < max_data_length:
                        continue
                    og_data.append(og)
                    cv_data.append(cv)
                    cv_data1.append(cv1)
                if len(og_data) > 0:
                    og, ci_og, niter_og = rolling_mean_along_axis(np.asarray(og_data), W=30)
                    cv, ci_cv, niter_cv = rolling_mean_along_axis(np.asarray(cv_data), W=30)
                    cv1, ci_cv1, niter_cv1 = rolling_mean_along_axis(np.asarray(cv_data1), W=30)
                    label_name = exp
                    plot_lines(og, ci_og, niter_og, colors[ind], line_width[i], ax_og, "-", label_name, cv=0)
                    plot_lines(cv, ci_cv, niter_cv, colors[ind], line_width[i], ax_cv, "-", label_name, cv=1)
                    plot_lines(cv1, ci_cv1, niter_cv1, colors[ind], line_width[i], ax_cv1, "-", label_name, cv=1)
                    ind += 1
        set_label_save(fig_og, ax_og, "Return", save_dir, "return" + "_e" + str(e))
        set_label_save(fig_cv, ax_cv, "Constraint violation 1", save_dir, "CV1" + "_e" + str(e))
        set_label_save(fig_cv1, ax_cv1, "Constraint violation 2", save_dir, "CV2" + "_e" + str(e))
        plt.close('all')


def set_label_save(fig, ax, y_label, save_dir, figname):
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    ax.legend(loc='best', prop={"size": 6}, ncol=2)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, figname + ".png"), facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0)
    fig.clf()


def plot_lines(mean_val, ci, niter, color, line_width, ax, linestyle, label_name, cv=0):
    if cv == 1:
        ax.axhline(y=0, linestyle="--", color='black')
    ax.plot(np.arange(niter), mean_val, linestyle=linestyle, linewidth=line_width, color=color, label=label_name)
    ax.fill_between(np.arange(niter), (mean_val - ci), (mean_val + ci), color=color, alpha=.1)


def get_data(loc):
    if not os.path.exists(os.path.join(loc, 'reward.npy')):
        return None
    rew = np.load(os.path.join(loc, 'reward.npy'))[:max_data_length]
    cv = np.load(os.path.join(loc, 'CV.npy'))[:max_data_length]
    return rew, cv[:, 0], cv[:, 1]


if __name__ == '__main__':
    base_data_dir = "./Results/LFA/Cartpole/Constraints"
    data_dir = ["CBP", "GDA", "CRPO"]
    for j in range(len(data_dir)):
        plot_param_sensitivity(data_dir[j:j + 1], base_data_dir)
