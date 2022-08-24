import torch
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from pathlib import Path
from config import results_root
from scipy.stats import ttest_ind
from data_handling.utils import pickle_load


sorted_composers = ['bach', 'scarlatti', 'haydn', 'mozart', 'beethoven', 'schubert', 'chopin', 'schumann', 'liszt',
                    'brahms', 'debussy', 'scriabin', 'rachmaninoff']
concept_names = {'1': 'alberti bass', '6': 'difficult-to-play music', '7': 'contrapuntal texture'}


def plot_significance_result(layer, alberti_scores, difficult_scores, texture_scores):
    """ Plots results of significance tests for given layer, and 3 predetermined concepts. Dark colour significant, light colour non-significant. """
    # create plot
    fig, ax = plt.subplots(1, len(concept_names), sharey=True, figsize=(40, 10))
    ax[0].set_ylabel('Average TCAV score', fontsize=18)
    significant_c = cm.get_cmap('inferno')(0.1)
    insignificant_c = cm.get_cmap('inferno')(0.9)

    # compute significance test, store for table later on
    res_dict = {0: {c: -1. for c in sorted_composers},
                1: {c: -1. for c in sorted_composers},
                2: {c: -1. for c in sorted_composers}}
    for i, score in enumerate([alberti_scores, difficult_scores, texture_scores]):
        for j in range(len(sorted_composers)):
            current = [score[sorted_composers[j]][c][layer][0] for c in score[sorted_composers[j]].keys() if
                       not c.startswith('9')]
            _, pval = ttest_ind(current,
                                [score[sorted_composers[j]][c][layer][1] for c in score[sorted_composers[j]].keys() if
                                 not c.startswith('9')])
            bar_height = np.mean(current)
            ax[i].bar([j], [bar_height], color=significant_c if pval < (0.05 / 13.) else insignificant_c)
            if pval < (0.05 / 13.):
                ax[i].annotate('*', xy=(ax[i].patches[j].get_x() + 0.3, ax[i].patches[j].get_height() + 0.005),
                               fontsize=21)
            res_dict[i][sorted_composers[j]] = (pval, bar_height)
        ax[i].set_title(concept_names[list(concept_names.keys())[i]], fontsize=20)
        ax[i].set_xticks(np.arange(len(sorted_composers)))
        ax[i].set_xticklabels(sorted_composers, fontsize=18)
        plt.setp(ax[i].get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
    plt.show()
    return res_dict


def show_table(pvals):
    """ Formats above plot into table similar to table 3 in the paper. """
    data = [['+' if pvals[k][c][0] < (0.05 / 13.) and pvals[k][c][1] >= 0.5 else
             ('-' if pvals[k][c][0] < (0.05 / 13.) and pvals[k][c][1] < 0.5 else ' ')
             for c in pvals[k].keys()] for k in pvals.keys()]
    dataframe = pd.DataFrame(data, columns=sorted_composers,
                             index=['alberti bass', 'difficult-to-play music', 'contrapuntal texture'])

    print(dataframe)


def main():
    layer = 'layer4'
    # prepare scores
    dir_path = Path(results_root) / 'significance_tests'
    alberti_files = list(dir_path.rglob('*1*/*.pkl'))
    difficult_files = list(dir_path.rglob('*6*/*.pkl'))
    texture_files = list(dir_path.rglob('*7*/*.pkl'))

    alberti_scores = {f.parent.name.rsplit('_')[1]: pickle_load(f) for f in alberti_files}
    alberti_scores = {comp: {
        c: {layer: torch.sum(alberti_scores[comp][c][layer] > 0., dim=0) / alberti_scores[comp][c][layer].shape[0]}
        for c in alberti_scores[comp].keys()} for comp in sorted_composers}
    difficult_scores = {f.parent.name.rsplit('_')[1]: pickle_load(f) for f in difficult_files}
    difficult_scores = {comp: {
        c: {layer: torch.sum(difficult_scores[comp][c][layer] > 0., dim=0) / difficult_scores[comp][c][layer].shape[0]}
        for c in difficult_scores[comp].keys()} for comp in sorted_composers}
    texture_scores = {f.parent.name.rsplit('_')[1]: pickle_load(f) for f in texture_files}
    texture_scores = {comp: {
        c: {layer: torch.sum(texture_scores[comp][c][layer] > 0., dim=0) / texture_scores[comp][c][layer].shape[0]}
        for c in texture_scores[comp].keys()} for comp in sorted_composers}

    # plot
    pvals = plot_significance_result(layer, alberti_scores, difficult_scores, texture_scores)

    # make table
    show_table(pvals)


if __name__ == '__main__':
    main()
