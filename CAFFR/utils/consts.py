"""
 Consts and ubiquos functions for CAFFR(Cellular Automata Forest Fire Rollout)

 Numba, for njit and cuda kernels management
"""

# Imports
import numpy as np
import math

from numba import cuda
import numba as nb

# Numba random methods. 
from numba.cuda.random import create_xoroshiro128p_states as rdm_states_gen
from numba.cuda.random import xoroshiro128p_uniform_float32 as rdm_uniform_sample
from numba.typed import List

# Functions
ceil = math.ceil
exp = math.exp
floor = math.floor
inf = math.inf
pow = math.pow

def createRandomGen(seed):
    assert seed > 0, "Please provide a positive integer as a seed."
    return np.random.Generator(np.random.SFC64(seed))

# These values needs to be here an declared for the compiler. One can change
# then before to create any kernel

# Pascal has SM with 64 or 128 units
THREADSPREAD = 128

NPTFLOAT = np.float32
NPTINT = np.int16
NPTINT8 = np.int8
NBTFLOAT = nb.float32
NBTINT = nb.int16
NBTINT8 = nb.int8

EMPTY = 0
TREE = 1
FIRE = 7

N_SAMPLES = 30
LOOKAHEAD = 7
K = 50
# This are to make the movements, change them
# before calling the kernel if necessary
ACTION_SET = [1,2,3,4,5,6,7,8,9]
L_AS = len(ACTION_SET)

#            n_row, n_col
GRID_SIZE = (   16,   16)
CRN_SIZE = (K, 16, 16)

# Design sizes, do not change until the kernel functions are changed as well
PARAMETERS_SIZE = 8
COSTS_SIZE = 6
PROBS_SIZE = 2