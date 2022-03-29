import os
from PainlessPolicyOpt.plot_helper import *

"""
Vary the number of feature dimension for linear function approximation. Fig 10. 
"""
nruns = 1
max_data_length = 400
num_samples = 1000
gam = [0.9]


def plot_param_sensitivity(data_dir, base_data_dir):
    save_dir = os.path.join(base_data_dir, "Plots", "ModelFree", "TileCoding", "FeatureVary",
                            "Sample" + str(num_samples))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig_og, ax_og = plt.subplots(figsize=(5, 5))
    fig_cv, ax_cv = plt.subplots(figsize=(5, 5))
    colors = ['r', 'b', 'g', 'y', 'orange', 'm', 'black', 'chocolate', 'grey']
    line_width = [2, 1, 1]
    n_tiles = [10, 14, 20]
    feature_overlap = [40, 56, 80]
    line_styles = ["-", "--"]
    algo_name = ["CBP", "GDA", "CRPO"]
    markers = ['o', 's', 'D', '*']
    ind = 0
    for i in range(len(data_dir)):
        for g_ind, g_val in enumerate(gam):
            for t_ind, tile_val in enumerate(n_tiles):
                path = os.path.join(base_data_dir, data_dir[i])
                exps = os.listdir(path)
                num_exp = len(exps)
                for exp_num, exp in enumerate(exps):
                    og_data = []
                    cv_data = []
                    if "_gam" + str(0.9) not in exp or "Iter800_" not in exp:
                        continue
                    if "_tileDim" + str(tile_val) not in exp or "_num_samples300" not in exp:
                        continue
                    for r in range(1, nruns + 1):
                        run_path = os.path.join(path, exp, "R" + str(r))
                        if not os.path.exists(run_path):
                            continue
                        data = get_data(run_path)
                        if not data:
                            continue
                        # print(run_path)
                        og, cv = data[0], data[1]
                        og_data.append(og)
                        cv_data.append(cv)
                    og, ci_og, niter_og = rolling_mean_along_axis(np.asarray(og_data), W=10)
                    cv, ci_cv, niter_cv = rolling_mean_along_axis(np.asarray(cv_data), W=10)
                    label_name = algo_name[i]
                    plot_lines(og, ci_og, niter_og, colors[i], line_width[i], ax_og, "-", "",
                               label_name)
                    plot_lines(cv, ci_cv, niter_cv, colors[i], line_width[i], ax_cv, "-", "",
                               label_name)
                    ind += 1
    set_label_save(fig_og, ax_og, "Optimality gap", save_dir, "OG")
    set_label_save(fig_cv, ax_cv, "Constraint violation", save_dir, "CV")
    plt.close('all')


def set_label_save(fig, ax, y_label, save_dir, figname):
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    ax.legend(loc='best', prop={"size": 10}, ncol=1)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, figname + ".png"), facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0)
    fig.clf()


def plot_lines(mean_val, ci, niter, color, line_width, ax, linestyle, marker="", label_name=""):
    if len(marker):
        ax.plot(np.arange(niter), mean_val, linestyle=linestyle, linewidth=line_width, color=color, label=label_name,
                marker=marker, markevery=40)
    else:
        ax.plot(np.arange(niter), mean_val, linestyle=linestyle, linewidth=line_width, color=color, label=label_name)


def get_data(loc):
    if not os.path.exists(os.path.join(loc, 'OG_avg.npy')):
        return None
    og_avg = np.load(os.path.join(loc, 'OG_avg.npy'))[:max_data_length]
    cv_avg = np.mean(np.load(os.path.join(loc, 'CV_avg.npy')), axis=1)[:max_data_length]
    return og_avg, cv_avg


if __name__ == '__main__':
    local_try = True
    if not local_try:
        base_data_dir = "/network/scratch/j/jainarus/CMDPData/Results/Tabular"
    else:
        base_data_dir = "./Results/Tabular"
    data_dir = ["CB/ModelFree/TileCodingFeatures", "GDA/ModelFree/TileCodingFeatures",
                "CRPO/ModelFree/TileCodingFeatures"]
    plot_param_sensitivity(data_dir, base_data_dir)
