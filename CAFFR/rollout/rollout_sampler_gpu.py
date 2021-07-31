# -*- coding: utf-8 -*-
"""
v 0.4

@author: bobobert

All new rollout sampler function using the cuda compiler from the libraty Numba.
"""

from CAFFR.utils.consts import *
from CAFFR.envs.forest_fire_cuda import helicopter_step
from CAFFR.heuristics.heuristic_cuda import Heuristic

def sampler(env,
            rg,
            trajectory = None,
            h_mode=0,
            gamma=1.0,
            n_samples = 30,
            k=50,
            lookahead=1,
            min_obj=True,
            action_set=None,
            mcts_p = 1.0,
            seed=1):
    """
    Function to pass an Helicopter environment and the other parameters
    to sample all the actions from the generated tree given the lookahead
    on a GPU CUDA Capable device. 

    Make sure your PC has the adecuate requeriments on CUDATOOLKIT and 
    Numba. 

    At the moment the GPU kenerl is compiled with each call of this function.
    Parameters
    ----------
    env: class Type Helicopter
        The environment which actual state will the initial state for all
        samples
    random: 
    h_mode: int
        From the heurititcs programmed in this package. Pass the mode one
        wants the heuristic to run during all the samples.
    trajectory: numba_cuda array
        If once created and MCTS_P == 1, it can be reused.
    gamma: float
        Disconunt value that will be applied to each step cost. This is done
        in a manner that the farther costs can have a less impact on the sample
        cost overall.
    n_samples: int
        Quantity of samples to execute per trajectory on the tree.
    k: int
        Steps to execute with the heuristic given. It can be zero if no heuristic
        is required
    lookahead: int
        The depth of the trajectory tree.
    min_obj: bool
        If the objective is to minimize set to true, otherwise False.
    action_set: list
        If one wants to change the action_set from the package give a list. Otherwise
        leave it in None.
    """
    #Checking for cuda devices
    is_device = False
    for _ in cuda.gpus:
        is_device = True
    assert is_device, \
        "No Cuda capable gpu found. Please check your numba and cudatoolkit if you have a cuda capable deivce."
    assert (env.boundary == 'invariant') and \
        (env.forest_mode == 'stochastic'), \
        "The actual environment configuration is not supported in this sampler"
    assert k>lookahead, "The budget has to be bigger than the lookahead planned"
    #Cost function mode
    c_mode = 0
    env_cost = env.reward_type
    if env_cost == 'custom':
        c_mode = 0
    elif env_cost == 'quad':
        c_mode = 1
    elif env_cost == 'ratio':
        c_mode = 2
    elif env_cost == 'hit':
        c_mode = 3
    elif env_cost == 'hit_ratio':
        c_mode = 4
    elif env_cost == "hit_fire":
        c_mode = 5
    # CHANGING GLOBALS VARIABLES
    # I regret this decisiont
    SAMPLER_CONST = globals()
    SAMPLER_CONST["GRID_SIZE"] = env.grid.shape
    SAMPLER_CONST["CRN_SIZE"] = (k, GRID_SIZE[0], GRID_SIZE[1])
    
    if action_set is not None:
        SAMPLER_CONST['ACTION_SET'] = action_set
        SAMPLER_CONST['L_AS'] =  len(action_set)
    
    # Calculating kernel size for max capacity
    max_trajectories = ceil(pow(L_AS, lookahead) * pow(mcts_p, lookahead - 1))
    blockspread = ceil(max_trajectories / THREADSPREAD)
    # Setting Random generators
    random_states = rdm_states_gen(THREADSPREAD*blockspread, seed=seed)
    #Loading the actual state of the environment to the device
    d_grid, d_probs, d_params, d_costs = load_forest_fire_2_device(
        grid=env.grid,
        p_fire=env.p_fire,
        p_tree= env.p_tree,
        gamma=gamma,
        k=k,
        reward_tree=env.reward_tree,
        reward_fire=env.reward_fire,
        reward_empty=env.reward_empty,
        reward_hit=env.reward_hit,
        reward_move=env.reward_move,
        pos=(env.pos_row, env.pos_col),
        actual_steps_before_update=env.remaining_moves,
        steps_before_update=env.moves_before_updating,
        h_mode=h_mode,
        c_mode=c_mode,
        empty= env.empty,
        tree = env.tree,
        fire = env.fire,
    )
    
    ACTION_SET_NB = List(ACTION_SET)
    def runSample(trajectories):
        t = np.array(trajectories)
        d_trajectories = cuda.to_device(t)
        d_results = cuda.device_array(d_trajectories.shape[0], dtype=NPTFLOAT)
        d_crn = cuda.device_array([k, *env.grid.shape], dtype = NPTFLOAT)
        sample_results, c_samples = np.zeros(d_trajectories.shape[0], dtype=NPTFLOAT), 1 / n_samples
        # Obtain samples
        for _ in range(n_samples):
            gen_crn[blockspread, THREADSPREAD](random_states, d_crn) # New generation 
            sample_trajectories[blockspread, THREADSPREAD](d_grid, d_crn,#generateCRN(rg, env.grid, k),
                                                            d_probs, d_params, d_costs, d_trajectories,
                                                            random_states, d_results)
            sample_results += d_results.copy_to_host() * c_samples
        return trajectories, sample_results

    # Start expanding the tree
    ## base case
    if mcts_p == 1.0:
        if trajectory is None:
            trajectories = List(List([a]) for a in ACTION_SET)
            for _ in range(1, lookahead):
                # Do all
                trajectories = expandLeafs(trajectories, ACTION_SET_NB)
        else:
            trajectories = trajectory

        trajectories, results = runSample(trajectories)
        return minMax(trajectories, results, ACTION_SET_NB, min_obj), trajectories

    for l in range(lookahead):
        if l == 0: # Create new trajectories
            trajectories = List(List([a]) for a in ACTION_SET)
        else: # Expand the selected trajectories with new leafs
            trajectories = expandLeafs(trajectories, ACTION_SET_NB)
        # Pass to device
        trajectories, results = runSample(trajectories)
        trajectories, results = selectTrajectories(trajectories, results, mcts_p ,min_obj)
    
    return minMax(trajectories, results, ACTION_SET_NB, min_obj), None

@nb.njit
def selectTrajectories(trajectories:List, results:np.ndarray, mcts_p:float, min_obj:False):
    if mcts_p == 1.0:
        return trajectories, results
    
    selectedTrajectories = List()
    idsSorted = np.argsort(results)
    upTo = ceil(results.shape[0] * mcts_p)
    if min_obj:
        ids = idsSorted[:upTo]
    else:
        fromTo = results.shape[0] - upTo
        ids = idsSorted[fromTo:]
    for idSorted in ids:
        selectedTrajectories.append(trajectories[idSorted])
    return selectedTrajectories, results[ids]

def generateCRN(rg, grid, budget):
    throws = rg.uniform(size = [budget, *grid.shape])
    throws = throws.astype(NPTFLOAT)
    return cuda.to_device(throws)


def load_forest_fire_2_device(grid=None, grid_size=(20,20),
                            ip_tree=0.5, p_fire=0.001, p_tree=0.005,
                            gamma = 0.99, k = 50,
                            reward_tree = -3.0, reward_fire = 1.5, reward_empty = 0.0, 
                            reward_hit = -0.20, reward_move = 1.0,
                            pos=(7,7),
                            actual_steps_before_update = 4,
                            steps_before_update = 4,
                            h_mode=0, c_mode=0,
                            empty=EMPTY, fire=FIRE, tree=TREE):
    """
    This is a function that returns an explicit list to the device memory for it to access
    it for the other functions running on the device threads. Run before the kernel call to load
    the initial state.
    Parameters
    ----------
    grid: numpy array or tuple int 
        Is None provided, it generates one random grid with EMPTY(0), TREE(1) cell type. Fire can 
        be on the grid after the next iterations, this cell type is (7).
    grid_size: tuple of ints
        If grid=None, this tuple must be given to generate a grid to the size to match.
    ip_tree: float
        Probability for trees to spawn in the initial generation of the grid. As the distribution
        is uniform, this probability is in average the ratio of the trees on the grid.
    p_fire: float
        Probability for a lightning to hit a cell.
    p_tree: float
        Probability for a tree to grow from an empty cell type.
    pos: tuple of int
        Position in (row, col) for the helicopter agent.
    actual_steps_before_update: int
        The number of steps left on the actual position of the agent for the environment
        to update its grid.
    steps_before_update: int
        The steps the agent can take before the environment updates.
    h_mode: int
        An integer to call
    """
    def check_probs(probs):
        probs = np.array(probs, dtype=NPTFLOAT)
        s_probs = np.sum(probs)
        if s_probs > 1:
            # Normalizing if the probability is larger
            probs = probs / s_probs
        return probs
    # Grid loading/generating
    if grid is None:
        # Generating grid, no Fire
        grid = np.random.choice([TREE, EMPTY], size=grid_size, p=[ip_tree, 1 - ip_tree])
    else:
        # Saving the a copy of the grid
        grid = grid.copy()
    g_rows, g_cols = grid.shape
    # Updating the probabilities passed
    #                      p_fire, p_tree
    #                        0      1
    probs = check_probs([p_fire, p_tree])
    # This will be a int type array
    #            pos_row, pos_col, actual_steps_before_update, steps_before_update, k, seed, h_mode c_mode  
    #               0        1              2                          3            4   5       6      7   
    parameters = [pos[0], pos[1], actual_steps_before_update, steps_before_update, k,  0, h_mode, c_mode,\
         g_rows, g_cols, empty, tree, fire]
    # rows cols empty tree fire
    #   8   9    10     11   12
    parameters = np.array(parameters, dtype=NPTINT)
    #        costs_tree,  cost_fire,   cost_empty,   cost_hit,   cost_move,   gamma
    #             0           1            2             3           4          5
    costs = [reward_tree, reward_fire, reward_empty, reward_hit, reward_move, gamma]
    costs = np.array(costs, dtype=NPTFLOAT)

    # Load to device
    grid_mem = cuda.to_device(np.array(grid, dtype=NPTINT8))
    probs_mem = cuda.to_device(probs)
    params_mem = cuda.to_device(parameters)
    costs_mem = cuda.to_device(costs)

    return grid_mem, probs_mem, params_mem, costs_mem # Pass references on device.

@nb.njit
def expandLeafs(leafs:List, action_set:List):
    # leafs is a List of List's
    new_leafs = List()
    for leaf in leafs:
        for action in action_set:
                new_leaf = leaf.copy()
                new_leaf.append(action)
                new_leafs.append(new_leaf)
    return new_leafs

@cuda.jit
def gen_crn(random_state, target):
    worker = cuda.grid(1)
    if worker < target.shape[0]: # A worker to gen each step
        r, c = target.shape[1], target.shape[2]
        for i in range(r):
            for j in range(c):
                target[worker, i, j] = rdm_uniform_sample(random_state, worker)

@cuda.jit
def sample_trajectories(grid,
                        crns,
                        probs, 
                        parameters, 
                        costs, 
                        trajectories, 
                        random_states, 
                        results):
    """
    New function to sample all the trajectories individually from all the posible
    trayctories and the samples; for each there's a repeated trajectory. This is made this way
    to use the most of the device resources most of the time to accelerate the process.
    Very ad-hoc for the forest fire environment
    """
    
    # Starting linear addressing of the samples
    worker = cuda.grid(1)

    if worker < trajectories.shape[0]:
        # A local copy of the grid
        local_grid = cuda.local.array(GRID_SIZE, dtype=NBTINT8)
        crn = cuda.local.array(GRID_SIZE, dtype=NBTINT8)
        # Updated works to generate a copy from here to better behavior
        updated_grid = cuda.local.array(GRID_SIZE, dtype=NBTINT8)
        for i in range(parameters[8]):
            for j in range(parameters[9]):
                local_grid[i,j] = grid[i,j]
                updated_grid[i,j] = grid[i,j]

        # Doing a sample given a trajectory
        # Initial conditions
        sample_cost = 0.0
        trajectory, budget = trajectories[worker], parameters[4]
        ## Grid conditions
        pos_row, pos_col = parameters[0], parameters[1]
        steps_to_update = parameters[2]
        ## Gamma
        GAMMA = costs[5]
        gamma = GAMMA * 1.0
        
        # Begin the sample
        for k in range(budget):
            # Extracting CRN
            crn = crns[k]
            # Calculate action
            if k < trajectory.shape[0]:
                action = trajectory[k]
            else:
                action = Heuristic(local_grid, pos_row, pos_col, steps_to_update,
                                    parameters, probs, costs, random_states, worker)

            # Take step
            updated_grid, pos_row, pos_col, steps_to_update, cost = \
                helicopter_step(local_grid, updated_grid, crn,
                                pos_row, pos_col, steps_to_update, action,
                                parameters, probs, costs)
            sample_cost += gamma * cost #Save cost
            gamma *= GAMMA # Updating gamma
            # Updating local_grid
            for i in range(parameters[8]):
                for j in range(parameters[9]):
                    local_grid[i,j] = updated_grid[i,j]
           
        # Saving cost into sample
        results[worker] = sample_cost # Finishing worker, saving result into results memory

@nb.njit
def minMax(trajectories: List, results: np.ndarray, action_set:List, min_obj:bool):
    """
    A custon function to calculate the means of the trajectories
    given only the first action of the trajectory. Then calculates
    the minimum.
    """
    best_action, best_cost, obj = 5, np.inf, 1

    if not min_obj:
        # Maximize
        obj = -1
        best_cost *= obj
    
    # Obtain mean from the action_root from all the trajectories
    result_action = [[0.0,0.0] for i in action_set]
    for i in range(results.shape[0]):
        action_root = trajectories[i][0] - 1
        value = results[i]
        c = result_action[action_root][1]
        prev_v = result_action[action_root][0]
        result_action[action_root][0] = (prev_v * (c) + value) / (c + 1)
        result_action[action_root][1] += 1

    #best_actions = List()
    for i in range(len(action_set)):
        c = result_action[i][0]
        if obj*c < obj*best_cost:
            best_cost = c
            #best_actions = List()
            best_actions = List([i + 1])
        elif obj*c == obj*best_cost:
            best_actions.append(i + 1)
            
    best_action = best_actions[np.random.randint(len(best_actions))]
    return best_action, best_cost, None

