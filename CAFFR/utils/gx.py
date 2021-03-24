"""
    Graphics utils

"""

from .consts import *
from.misc import timeStr
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

sns.set(context="paper", style="whitegrid")
# GX constants

Y_TICKS_ROT = 30
R_CLR = sns.xkcd_rgb["medium green"]
H_CLR = sns.xkcd_rgb["denim blue"]
ALPHA_FILL = 0.1
ALPHA_LINE = 0.6
ALPHA_HEU = 0.9
LINE_W = 2
POINT_W = 1
LINE_TYPE = '-.'
MARK_HEU = 'o'
SIZE = (10,6)

def make_graph(rollout_results_step,
                heurist_results_step,
                mean = True,
                title_head='',
                unitX = "Steps",
                unitY='Cost',
                dpi=200):
    """
    Function to make a graph from rollout and heuristics results.
    It saves with the str name_save
    """
    
    fig = plt.figure(figsize=SIZE,dpi=dpi)

    # First graph - Avg. Cost per step of all test
    # Acumulative cost per step per test
    R_RESULTS_STEPS = np.array(rollout_results_step) 
    H_RESULTS_STEPS = np.array(heurist_results_step)
    x = range(R_RESULTS_STEPS.shape[1]) if mean else range(len(R_RESULTS_STEPS))
    if mean:
        mean_r = np.mean(R_RESULTS_STEPS, axis=0)
        std_r = np.std(R_RESULTS_STEPS, axis=0)
    else:
        mean_r = R_RESULTS_STEPS
    plt.plot(x, mean_r, label='Rollout', alpha=ALPHA_LINE, color=R_CLR, lw=LINE_W)
    if mean:
        plt.fill_between(x, mean_r-std_r, mean_r+std_r, alpha=ALPHA_FILL, color=R_CLR)
    #plt.scatter(x, mean_r-std_r, alpha=ALPHA_LINE, color=R_CLR, lw=POINT_W, marker=r_mark)
    #plt.scatter(x, mean_r+std_r, alpha=ALPHA_LINE, color=R_CLR, lw=POINT_W, marker=r_mark)
    if mean:
        mean_h = np.mean(H_RESULTS_STEPS, axis=0) if mean else H_RESULTS_STEPS
        std_h = np.std(H_RESULTS_STEPS, axis=0)
    else:
        mean_h = H_RESULTS_STEPS
    plt.plot(x, mean_h, label='Heuristic', alpha=ALPHA_HEU, color=H_CLR, lw=LINE_W, ls=LINE_TYPE)
    #plt.fill_between(x, mean_h-std_h, mean_h+std_h, alpha=ALPHA_FILL, color=H_CLR)
    if mean:
        plt.scatter(x, mean_h-std_h, alpha=ALPHA_HEU, color=H_CLR, s=POINT_W, marker=MARK_HEU)
        plt.scatter(x, mean_h+std_h, alpha=ALPHA_HEU, color=H_CLR, s=POINT_W, marker=MARK_HEU)
    plt.xlabel(unitX)
    plt.ylabel('Average Accumulated' + unitY)
    plt.yticks(rotation=Y_TICKS_ROT)
    plt.title('{} {}/{}'.format(title_head, unitX, unitY))
    plt.legend()

def make_graphs():
    # TODO
    None


def cleanPlot():
    plt.clf()

def saveFig(name):
    name += " -- {}.png".format(timeStr())
    plt.savefig(name)
    cleanPlot()