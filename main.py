from CAFFR.envs import EnvMakerForestFire
from CAFFR.heuristics import heuristic
from CAFFR.rollout import rollout
from CAFFR.utils.misc import expDir, savePickle

# -----------------------Preparing the experiment----------------------------
# Environment parameters
N_ROW = 50              #Grid Rows
N_COL = 50              #Grid Columns
Init_Row=24              #Helicopter initial row
Init_Col=24              #Helicopter initial col
IP_TREE = 0.4           #init tree
IP_FIRE = 0.0
P_FIRE = 0.005           #Probability to turn a green cell into ared cell
P_TREE = 0.05           #Probability to turn an empty cell into a green cell
FREEZE = 10            #Movements of Helicopter after update Automata
# Symbols for cells
TREE = 0
FIRE = 2
EMPTY = 1
## Environment cost shape
C_TYPE = 'hit_ratio'
C_TREE = 0.0          # Costs associated with its type cell
C_FIRE = 0.0           #
C_EMPTY = 0           #
C_HIT = 0.01             # associated to put down a fire.
C_STEP = 0.0            # Per step given on the environment
C_MOVE = -0.001            # Cost to chage position
# Experiment parameters
N_ITERS = 10
N_STEPS = 50            # updates of the grid
N_SAMPLES = 20
K_Rollout = 3 * FREEZE
LOOKAHEAD = 3
MCTS_P = 1.0
GAMMA = 0.98
SEED = 1313

expdir = expDir("rollout", "forest_fire_helicopter")

if __name__ == '__main__':
    env = EnvMakerForestFire(
        n_row = N_ROW, n_col = N_COL, 
        p_tree = P_TREE, p_fire = P_FIRE,
        init_pos_row = Init_Row, init_pos_col = Init_Col, 
        moves_before_updating = FREEZE - 1,  
        tree = TREE, empty = EMPTY, fire = FIRE,
        reward_type = C_TYPE, reward_tree = C_TREE, reward_fire = C_FIRE,
        reward_empty = C_EMPTY, reward_hit = C_HIT, reward_step = C_STEP,
        reward_move = C_MOVE, ip_fire = IP_FIRE, ip_tree = IP_TREE, ip_rock=0.0, ip_lake=0.0)
    H = heuristic.Heuristic_m2_v3
    exp = rollout.Experiment(env, H, H_mode = 23,
        N_ITERS=N_ITERS, N_STEPS=N_STEPS, N_SAMPLES=N_SAMPLES, 
        K=K_Rollout, LOOKAHEAD=LOOKAHEAD, MIN_OBJECTIVE=False, 
        GAMMA=GAMMA, MCTS_P=MCTS_P, seed = SEED)
    results = exp.run(GIF=False, GRAPH = False) # This will generate the graphs and GIF from the experiment.
    savePickle(results, expdir, "results")

