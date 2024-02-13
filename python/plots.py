import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def load_experiment_data(file_path):
    experiment_data = {}
    for idx, file in enumerate(os.listdir(file_path)):
        experiment_data[idx] = json.load(open(os.path.join(file_path, file)))
    return experiment_data

def extract_r_data(exp_data, sample_sizes, k_sizes):
    df = pd.DataFrame(index=range(len(exp_data)), columns=pd.MultiIndex.from_product([sample_sizes, k_sizes]))
    for iter_key in exp_data.keys():
        for inner_dict in exp_data[iter_key]:
            n = inner_dict['N']
            k = inner_dict['K']
            value = inner_dict['Value']
            df.at[iter_key, (n, k)] = value
    return convert_raw_df_to_average(df, drop_col_1=False)

def convert_raw_df_to_average(df: pd.DataFrame, drop_col_1=True):
    df_piv = df.mean(axis=0).to_frame().reset_index().pivot(index='level_0', columns='level_1', values=0)
    
    df_piv = df_piv.astype(float)
    if drop_col_1:
        df_piv = df_piv.drop(columns=[1])
    df_piv = df_piv.T
    df_piv = df_piv.sort_index(ascending=False)
    df_piv.index.name = 'k'
    df_piv.columns.name = 'n'
    return df_piv

def create_heatmap(df, title='', ax: plt.Axes | None = None):
    if ax is None: _, ax = plt.subplots()
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    cbar_kwargs = {'label': 'average', 'orientation': 'vertical', 'shrink': 0.85}
    
    ax = sns.heatmap(
        df, annot=True, fmt=".3f", cmap=cmap, linewidth=0.5, color='black',  xticklabels=1, yticklabels=1, cbar_kws=cbar_kwargs, ax=ax, square=True, annot_kws={'fontsize': 8.5}
        )
    ax_cbar = ax.collections[0].colorbar
    ax_cbar.set_label('             average', fontsize=10, labelpad=0, rotation=360, y=1.05, x=-5)
    ax_cbar.ax.yaxis.set_ticks_position('right')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title, loc='left', fontdict={'fontsize': 12, 'fontweight': 'bold'})
    return ax

def heatmap_r_experiments():
    sample_sizes = [1024, 2048, 4096, 8192, 16384, 32768]
    k_sizes = [2,3,4,5,6,7,8]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    fig.supylabel('Num. components $k$', fontsize=12, fontweight='bold', va='center', x=0.5)
    fig.supxlabel('Sample size $n$', fontsize=12, fontweight='bold', ha='center', y=0.05, x=0.65)
    
    e1_r_results_path = './E1 Simulation_json'
    e1_r_data = load_experiment_data(e1_r_results_path)
    e1_r_df = extract_r_data(e1_r_data, sample_sizes, k_sizes)
    ax1 = create_heatmap(e1_r_df, 'E1', ax=ax1)
    
    e2_r_results_path = './E2 Simulation_json'
    e2_r_data = load_experiment_data(e2_r_results_path)
    e2_r_df = extract_r_data(e2_r_data, sample_sizes, k_sizes)
    ax2 = create_heatmap(e2_r_df, 'E2', ax=ax2)

    plt.savefig('r_e1_e2_heatmap_vert.pdf', bbox_inches='tight', dpi=300)

def f1(x):
    if x <= 0.4:
        return 1
    elif x >= 0.6:
        return 1
    else:
        return 0
    
def f2(x):
    if x <= 0.5:
        return 2-4*x
    else:
        return 4*x-2
    
def plot_combined_densities():
    x = np.linspace(0, 1, 10000)
    y1 = [f1(xi) for xi in x]
    y2 = [f2(xi) for xi in x]
    
    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(x, y1, linestyle='-', color='k', alpha=0.8)
    ax.fill_between(x, y1, alpha=0.45, color='r', label='$f_1$')
    ax.plot(x, y2, linestyle='--', color='k', alpha=0.8)
    ax.fill_between(x, y2, alpha=0.35, color='b', label='$f_2$')
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.0, fontsize=10, frameon=False)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.grid(alpha=0.5, linestyle=':')
    ax.set_xticks([i/5 for i in range(6)])
    plt.savefig('f1_f2.pdf', bbox_inches='tight', dpi=300)

def main():
    heatmap_r_experiments()
    plot_combined_densities()
    
if __name__ == '__main__':
    main()
