import os
from PainlessPolicyOpt.plot_helper import *

"""
Model-free TD Sampling with [1000, 2000, 3000, 5000] samples with TD. Plot the results with diff number of samples.
"""
nruns=5
max_data_length = 1500

def plot_param_sensitivity(data_dir, base_data_dir, best_hypers):
    save_dir = os.path.join(base_data_dir, "Plots", "ModelFree", "TDSampling","VarySamples")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig_og, ax_og = plt.subplots(figsize=(5, 5))
    fig_cv, ax_cv = plt.subplots(figsize=(5, 5))
    colors = ['r', 'b', 'g', 'y', 'o']
    line_width = [1,1,1]
    gam = [0.9]
    samples = [1000, 2000, 3000]
    line_styles = ["-", "--"]
    markers = ['o', 's', 'D', '*']
    algo_name = ["CBP", "GDA", "CRPO"]
    for i in range(len(data_dir)):
        for g_ind, gam_val in enumerate(gam):
            for s_ind, sample in enumerate(samples):
                path = os.path.join(base_data_dir, data_dir[i])
                exps = os.listdir(path)
                for exp_num,exp in enumerate(exps):
                    og_data = []
                    cv_data = []
                    if best_hypers[i] not in exp or "Iter1500_" not in exp:
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
                    label_name= str(sample)+" "+ algo_name[i]
                    plot_lines(og, ci_og, niter_og, colors[i],line_width[i], ax_og, "-",label_name, markers[s_ind])
                    plot_lines(cv, ci_cv, niter_cv,  colors[i],line_width[i], ax_cv, "-", label_name, markers[s_ind])
    set_label_save(fig_og, ax_og, "Optimality gap", save_dir, "OG")
    set_label_save(fig_cv, ax_cv, "Constraint violation", save_dir, "CV")
    plt.close('all')

def set_label_save(fig, ax, y_label, save_dir, figname):
    ax.set_xlabel("Iterations")
    ax.set_ylabel(y_label)
    ax.legend(loc='upper right', prop={"size": 9}, ncol=3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.grid(True)
    fig.savefig(os.path.join(save_dir, figname + ".png"), facecolor='w', edgecolor='w',
                orientation='portrait', format=None, transparent=True, bbox_inches='tight', pad_inches=0)
    fig.clf()

def plot_lines(mean_val, ci, niter, color, line_width, ax, linestyle, label_name="", marker=""):
    if len(marker):
        ax.plot(np.arange(niter), mean_val, linestyle=linestyle, linewidth=line_width, color=color, label = label_name,
                marker=marker, markevery=200)
    else:
        ax.plot(np.arange(niter), mean_val, linestyle=linestyle, linewidth=line_width, color=color)
    ax.fill_between(np.arange(niter), (mean_val - ci), (mean_val + ci), color=color,alpha=.1)

def get_data(loc):
    if not os.path.exists(os.path.join(loc, 'OG_avg.npy')):
        return None
    og_avg = np.load(os.path.join(loc, 'OG_avg.npy'))[:max_data_length]
    cv_avg = np.mean(np.load(os.path.join(loc, 'CV_avg.npy')), axis=1)[:max_data_length] # taking mean of cv across num of constraints
    if len(cv_avg)<max_data_length:
        return None
    return og_avg, cv_avg


if __name__ == '__main__':
    local_try=True
    if not local_try:
        base_data_dir = "/network/scratch/j/jainarus/CMDPData/Results/Tabular"
    else:
        base_data_dir = "./Results/Tabular"

    data_dir = ["CB/ModelFree/TDSampling", "GDA/ModelFree/TDSampling", "CRPO/ModelFree/TDSampling"]
    best_hypers = ["_alpha8_", "LRP1.0_LRL0.1_", "lr0.75_"]
    plot_param_sensitivity(data_dir, base_data_dir, best_hypers)



