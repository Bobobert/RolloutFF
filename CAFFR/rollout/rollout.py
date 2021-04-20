# -*- coding: utf-8 -*-
"""
Created on jun 2020

v0.4

@author: bobobert, MauMontenegro

Version for GPU CUDA Capable ONLY
"""

from CAFFR.utils.consts import *
from CAFFR.rollout.rollout_sampler_gpu import sampler

#Misc for looks
import tqdm

# TODO change function to files
from CAFFR.utils.gx import *
from CAFFR.utils.misc import *

# TODO Erase Epsilon
# Add the MCTS costs

# Delete all the policy references

def Rollout_gpu(
    env,
    rg,
    trajectory = None,
    H:int = 0, gamma: float = 1.0, K: int = 10,
    lookahead:int = 1, N_samples:int = 10, 
    mcts_proportion: float = 1.0, 
    min_objective: bool =True,
    seed = -1, **kwargs):
    """Funtion to do a rollout from the actions available and the use of the Heuristic H.
    The argument k is to truncate how many steps the heuristic is going to execute. This
    version is edited to run with the sampler of the rollout_sample_gpu which needs other 
    considerations in this version. 
    It does not longer support to pass an function or object
    as the Heuristic. One needs to program it on the file under the Heuristic function and
    add a mode-key, which is the one is pass here.
    
    This function is wrote in base in the rollout algorithm as presented in the book of
    Dimitry Bertsekas Reinforcement Learning and Optimal Control.
    
    - BERTSEKAS, Dimitri P. Reinforcement learning and optimal control. Athena Scientific, 2019.

    It returns the best action from the rollout, and the expected value from taking it.
    action type, float.
    Parameters
    ----------
    env : Variable to reference the environmet to work with.
        It is expect to be an object class Helicopter for this version.
        The next methods and variables are expected in env:
        - grid
    H : int
        From the rollout_sampler_gpu.Heuristic() select a mode that has been
        programed inside that function to work in the device.
    trajectory: numba_cuda_array
        If available can be passed. If MCTS_P == 1.0 skips the recreation of 
        the array.
    gamma : float
        Discount factor for the cost function.
    K : int
        Integer of number of steps to execute a trajectory
    lookahead : int
        Numbert of steps that the rollout algorithm can take in greedy form, forming a series of controls,
        that minimizes of maximizes the cost function. This increases the cost of the computation, and 
        discards all the controls but except the first.
    mcts_p: float
        In [0,1], proportion of trajectories to keep when executing the MCTS algorithm. 1 = All trajectories
    N_samples : int
        Number of samples required to calculate the expected value of the
        cost function.
    mcts_proportion: float
        For a Monte Carlo Tree Search, given a proportion in [0,1] of how many branches for
        the trajectories to cut
    min_objective : bool 
        Variable to define if the objective is to maximize of minimize the cost function.
        This is problem-model dependant.
    rg : numpy.random.Generator type
        This function now needs an external random generator from the numpy library.
    """

    (best_action, best_cost, avgs), trajectory = sampler(env, rg, trajectory, h_mode=H, gamma=gamma, n_samples=N_samples, k=K,
                                    lookahead=lookahead, min_obj=min_objective, seed=seed,
                                    mcts_p = mcts_proportion)

    return best_action, best_cost, trajectory

class Experiment():
    """
    Class design to run a complete rollout experiment with the options to generate graphs,
    animations, save results in pickle form. 
    As this is a RL-Rollout implementation it needs a base heuristic or base policy to 
    call and to compare to start improving the min/max of the cost function.

    -- This is the GPU only version. It does not support Heuristic as function or object --
    
    Parameters
    ----------
    ENV : Environment Object
        Reference to the environment object. This funcion is constructed to the EnvMakerForestFire
        environment.
        The next methods and variables are expected in env:
        - action_set
        - step()
        - Encode()
        - copy()
        - make_checkpoint()
        - load_checkpoint()
        - frame()
        - make_gif()

    H : Function object
        Object or function that references the heurist to execute. These type of functions
        requiere to support their inputs as a dict with at least the 'observation', and 'env'
        keys on it. It must return an action type.
    
    H_mode : int
        Inside the rollout_sampler_gpu.Heuristic there is a set of heuristics to address
        passing this argument. For a heuristic to work in this version, it needs to be writen
        inside the function to compile to device. For the results to be accurate they need to
        output the same values per state.

    PI : Policy object
        Here one can pass an policy already started. If None it generates a new one.

    N_ITERS : int
        Number of tests to run the experiment  with .run() 
        The result of this is the average of the costs between all the tests for each run.
        Notice that inside a run every test starts with the same initial state.
        Each test has execute all the follwing variables. 

    N_STEPS : int 
        Number of steps that the environment takes. Speaking about the Helicopter environment, the variable
        freeze has an effect to update the environment each FREEZE steps. Therefore, the agent in total execute
        N_STEPS * FREEZE steps.

    N_SAMPLES : int
        Number of samples required to calculate the expected value of the
        cost function.

    K : int
        Integer of number of steps to execute a trajectory

    LOOKAHEAD : int
        Numbert of steps that the rollout algorithm can take in greedy form, forming a series of controls,
        that minimizes of maximizes the cost function. This increases the cost of the computation, and 
        discards all the controls but except the first.

    GAMMA : float 
        Discount factor for the cost function.

    EPSILON : float
        It must be a quantity of probability in the range 0<epsilon<=1 to take an exploration
        action. This makes up the behaviour in a epsilon-greedy technique. In this case greedy 
        been the heuristic H.

    EPSILON_DECAY : float
        The rate between 0 and 1, in which the value of epsilon decays every time it is used on the
        rollout executions.

    MIN_OBJECTIVE : bool 
        Variable to define if the objective is to maximize of minimize the cost function.
        This is problem-model dependant.

    RUN_GIF : bool
        Variable to control the behavior if the last execution of run generates frame for each
        agent step for being able to generate a .gif with the .gif method.

    Methods
    -------
    Experiment.run()

    Experiment.run_multiple_LH()

    Experiment.policy_test()

    Experiment.make_graph()
        
    Experiment.make_gif()
        
    Experiment.pickle()
        Dump the sequence of costs obtainen and the policy object

    Experiment.reset()
        Cleans buffers for graphs and gifs. Restart countes. 

    """
    def __init__(
        self,
        ENV,
        H,
        H_mode=0,
        N_ITERS = 10,
        N_STEPS = 25,
        N_SAMPLES = 29,
        K = 100,
        MCTS_P = 1.0,
        LOOKAHEAD = 1,
        GAMMA = 0.99,
        EPSILON = 0,
        EPSILON_DECAY = 0.99,
        MIN_OBJECTIVE = False,
        RUN_GIF = False,
        seed = 1,
        **kwargs):

        # Assertions

        def check_ints(suspect):
            assert (suspect >= 1) and isinstance(suspect, int),\
                "This number must an integer of at least 1. {} = {} given instead.".format(type(suspect), suspect)
            return suspect
        def check_prob(suspect):
            assert (suspect <= 1) and (suspect >= 0),\
                "This value must be between 0 and 1. {} was given".format(GAMMA)
            return suspect
        assert isinstance(MIN_OBJECTIVE, bool),\
            "With a True/False indicate if minimize is the objective. Invalid type {} passed".format(type(MIN_OBJECTIVE))
        # Saving references to objects and classes.
        self.env = ENV
        self.env_h = None # Var to env copy for applying the heuristic
        self.H = H
        self.H_mode = H_mode
        self.min_obj = MIN_OBJECTIVE
        self.seed = seed

        # Loading variables
        self.N_ITERS = check_ints(N_ITERS)
        self.N_STEPS = check_ints(N_STEPS)
        self.N_SAMPLES = check_ints(N_SAMPLES)
        if K < 0:
            self.K = -1
        else:
            self.K = check_ints(K)
        self.mcts_p = check_prob(MCTS_P)
        self.LOOKAHEAD = check_ints(LOOKAHEAD)
        self.gamma = check_prob(GAMMA)
        self.epsilon_op = check_prob(EPSILON)
        self.epsilon = check_prob(EPSILON)
        self.epsilon_decay = check_prob(EPSILON_DECAY)

        self.last_time = 0
        self.logger = Logger()
        self.logger(" - GPU Experiment -",False, False)
        env_desc = "Environment Parameters -- Grid: {} Cost_f: '{}'\n Cost_Tree: {} Cost_Fire: {} Cost_hit: {}\n\
            Cost_Empty: {} Cost_step: {} Cost_move: {}\n\
            Min_obj: {} P_Fire: {} P_Tree: {}\n Steps_To_Update {}".format(ENV.grid.shape, ENV.reward_type, ENV.reward_tree, ENV.reward_fire, ENV.reward_hit,
            ENV.reward_empty, ENV.reward_step, ENV.reward_move,
            MIN_OBJECTIVE, ENV.p_fire, ENV.p_tree, ENV.moves_before_updating)
        self.logger(env_desc,False,False)

        # This class has its own random generator.
        self.rg = createRandomGen(seed)
        self.runs_rollout_results = []
        self.runs_rollout_results_step = []
        self.runs_heu_results = []
        self.runs_heu_results_step = []
        self.runs_rollout_archive = []
        self.runs_heu_archive = []
        self.c_runs = 0
        self.theres_run_gif = False
        self.theres_test_gif = False
        self.RUN_GIF = RUN_GIF
        self.frames_run_r = []
        self.frames_run_h = []
        self.frames_test_r = []
        self.run_h_tcell = []
        self.run_r_tcell = []

        self.mod = "Cost"
    
    def __del__(self):
        return None

    def reset(self):
        # Free memory.
        self.env.checkpoints = []
        self.env_h.checkpoints = []
        self.runs_rollout_results = []
        self.runs_rollout_results_step = []
        self.runs_heu_results = []
        self.runs_heu_results_step = []
        self.runs_rollout_archive = []
        self.runs_heu_archive = []
        self.frames_run_r = []
        self.frames_run_h = []
        self.frames_test_r = []
        self.theres_run_gif = False
        self.theres_test_gif = False
        self.c_runs = 0
        self.epsilon = self.epsilon_op
        self.rg = createRandomGen(self.seed)

    def run(self, GIF=None, GRAPH=True):
        """
        Creates an initial state from reseting the environment and runs all the number of 
        iterations and so on.

        Parameters
        ----------
        GIF : bool 
            Variable to indicate if you desired to generate frames for the last 
            train loop of the run, if the class was initialized with this behavior on this one
            changes nothing. Default False.
        GRAPH : bool
            Draws and saves the graphs from the experiment. If there's not a graph generated and
            one does not restarts the class 
        """

        if not GIF is None:
            RUN_GIF = GIF
        else:
            RUN_GIF = self.RUN_GIF
        # Lists to save the results from the N_ITERS
        RO_RESULTS, H_RESULTS, R_RESULTS = [], [], []
        RO_RESULTS_C, H_RESULTS_C, R_RESULTS_C = [], [], []
        RO_RATIO, H_RATIO, R_RATIO = [], [], []
        # Measuring time of execution. 
        self.logger("Run {} - Metadata: {}\n |".format(self.c_runs, self.metadata_str), True, True, True)
        trajectory = None
        # First loop to execute an rollout experiment.
        for n_test in range(self.N_ITERS):
             # Reseting env and storing the initial observations
            observation = self.env.reset()
            observation_1 = observation
            #Making copy of the env to apply the heuristic
            self.env_h = self.env.copy()
            env_r = self.env.copy()
            # Making checkpoints
            #checkpoint_env = self.env.make_checkpoint()
            #checkpoint_env_h = self.env_h.make_checkpoint()
            # Passing a new identical random generator
            self.env.rg = createRandomGen(self.seed)
            self.env_h.rg = createRandomGen(self.seed)
            env_r.rg = createRandomGen(self.seed)          
            self.logger(" |-- Test : {} of {}".format(n_test+1, self.N_ITERS))
            # Making a checkpoint from the initial state generated.         
            #self.env.load_checkpoint(checkpoint_env)
            #self.env_h.load_checkpoint(checkpoint_env_h)
            # Setting up vars to store costs
            rollout_cost, heuristic_cost, random_cost = 0, 0, 0
            rollout_cost_step, heuristic_cost_step, random_cost_step =[], [], []
            ro_ratio, h_ratio, r_ratio = [], [], []
            # Making the progress bar
            bar = tqdm.tqdm(range((self.env.moves_before_updating + 1) * self.N_STEPS))
            for i in bar:
                #Calls Rollout Strategy and returns action,qvalue
                r_action, q_value, trajectory = Rollout_gpu(
                                                self.env, 
                                                self.rg,
                                                trajectory=trajectory,
                                                H=self.H_mode,
                                                gamma=self.gamma,
                                                epsilon=self.epsilon,
                                                K=self.K,
                                                mcts_proportion=self.mcts_p,
                                                lookahead=self.LOOKAHEAD,
                                                N_samples=self.N_SAMPLES,
                                                min_objective=self.min_obj,
                                                seed = self.seed)
                #Calls Heuristic and return best action
                To_H = {"env": self.env_h, "observation": observation_1}
                h_action = self.H(To_H)
                #Helicopter take an action based on Rollout strategy and heuristic
                observation, ro_cost, _, _ = self.env.step(r_action)
                observation_1, h_cost, _, _ = self.env_h.step(h_action)
                _, r_cost, _, _ = env_r.step(np.random.randint(1, 10))
                if RUN_GIF and (n_test == self.N_ITERS - 1):
                    # Framing just the last round
                    self.env.frame(title="Rollout step {}-th".format(i))
                    self.env_h.frame(title="Heuristic step {}-th".format(i))
                #Update Rollout Total cost
                rollout_cost += ro_cost  #Acumulative cost for rollout          
                rollout_cost_step.append(rollout_cost)  #List of cost over time
                #Update Heuristic Total cost
                heuristic_cost += h_cost
                heuristic_cost_step.append(heuristic_cost)
                random_cost += r_cost
                random_cost_step.append(random_cost)
                # Register ratios
                ro_ratio += [calculateRatio(self.env)]
                h_ratio += [calculateRatio(self.env_h)]
                r_ratio += [calculateRatio(env_r)]
                #Generate a message
                msg =    " |   |      |"
                msg += "\n |   |      |-- Agent step {}".format(i)
                msg += "\n |   |      |   |-- Rollout with action {} and cost : {}".format(r_action, ro_cost)
                msg += "\n |   |      |   '-- Heuristic with action {} and cost : {}".format(h_action, h_cost)
                bar.write(msg)
                self.logger(msg, False, False)
            bar.close()
            msg =    " |   |"
            msg += "\n |   |-- Test {} results".format(n_test+1)
            msg += "\n |       |-- Total Rollout cost : {}".format(rollout_cost)
            msg += "\n |       '-- Total Heuristic cost : {}".format(heuristic_cost)
            msg += "\n |"
            self.logger(msg)
            #Costs p/test
            RO_RESULTS.append(rollout_cost)
            H_RESULTS.append(heuristic_cost)
            R_RESULTS.append(random_cost)
            #Cumulative costs p/test
            RO_RESULTS_C.append(rollout_cost_step)
            H_RESULTS_C.append(heuristic_cost_step)
            R_RESULTS_C.append(random_cost_step)
            # Ratios
            RO_RATIO += [ro_ratio]
            H_RATIO += [h_ratio]
            R_RATIO += [r_ratio]
        msg = " | Run {} done.".format(self.c_runs)
        msg+= "\nMetadata: {}\n |".format(self.metadata_str)
        self.logger(msg, True, True, True)

        # Saving to the class
        self.runs_rollout_results += RO_RESULTS
        self.runs_rollout_results_step += RO_RESULTS_C
        self.runs_heu_results += H_RESULTS
        self.runs_heu_results_step += H_RESULTS_C
        if GRAPH:
            self.make_graph(title_head='Run:{} H:{} LH:{}'.format(self.c_runs,self.H_mode,self.LOOKAHEAD))
        self.c_runs += 1
        # Saving data to generate gif
        if RUN_GIF: 
            self.frames_run_r += self.env.frames
            self.env.frames = []
            self.frames_run_h += self.env_h.frames
            self.env_h.frames = []
            self.theres_run_gif = True
            self.make_gif(RUN=True)
        def createDict(scalar_accumulated, steps_accumulated, ratios):
            return {"return":scalar_accumulated,
            "step":steps_accumulated,
            "ratio":ratios}
        return {"rollout":createDict(RO_RESULTS, RO_RESULTS_C, RO_RATIO),
                "heuristic":createDict(H_RESULTS, H_RESULTS_C, H_RATIO),
                "random":createDict(R_RESULTS, R_RESULTS_C, R_RATIO)}

    def run_multiple_LH(self, LHS = [1], GRAPH=True, n_cols=2, dpi=200, save_arr=3600):
        """
        Creates an initial state from reseting the environment and runs all the number of train
        iterations and so on. This updates the policy with more states or with better actions.

        Parameters
        ----------
        LHS: list of ints
            The list of lookaheads to run tests on them. 
        GRAPH : bool
            Draws and saves the graphs from the experiment. If there's not a graph generated and
            one does not restarts the class
        n_cols : int
            Number of columns in which the graph for the individual accumulate cost will be displayed.
        dpi : int
            Quantity of desired pixels per inch in the saved graph.
        save_arr : int
            The ammount of seconds in which one desire to dump the actual values of the run on a
            npy file. This is check every end of the Test.
        """

        def GET_TCELL(ENVS, tree = True):
            counts = []
            for e in ENVS:
                c = 0
                for i in range(e.grid.shape[0]):
                    for j in range(e.grid.shape[1]):
                        if (e.grid[i,j] == e.tree) and tree:
                            c += 1
                        elif (e.grid[i,j] == e.fire) and not tree:
                            c+=1
                c = c / e.grid.shape[0] / e.grid.shape[1]
                counts += [c]
            return np.array(counts)

        def DUMP_FILES(mod=''):
            checkDir("Logs")
            time_s = timeStr()
            f1 = open("./Logs/rollout COSTS_STEP LHS {} -- {}{}.npy".format(LHS, mod, time_s), 'wb')
            np.save(f1, COSTS_STEP)
            f1.close()
            f2 = open("./Logs/rollout COSTS LHS {} -- {}{}.npy".format(LHS, mod, time_s), 'wb')
            np.save(f2, COSTS)
            f2.close()
            f3 = open("./Logs/rollout RTCELL LHS {} -- {}{}.npy".format(LHS, mod, time_s), 'wb')
            np.save(f3, END_R_TCELLS)
            f3.close()
            f4 = open("./Logs/rollout TCELL LHS {} -- {}{}.npy".format(LHS, mod, time_s), 'wb')
            np.save(f4, END_TCELLS)      
            f4.close()
            self.logger("Numpy Arrays save on ./Logs with time {}".format(time_s), time=False)
        
        l_LHS = len(LHS)
        # Storing in 0 for the heuristic
        H_env = self.env.copy()
        observation = H_env.reset()
        ENVS = [H_env]
        OBS = [observation]
        # Acumulate costs (LHS, N_ITERS)
        COSTS = np.zeros((l_LHS+1,self.N_ITERS), dtype=np.float32)
        # Per step costs (LHS, N_ITERS, TOT_STEPS)
        COSTS_STEP = np.zeros((l_LHS+1, self.N_ITERS, self.env.moves_before_updating * self.N_STEPS), dtype=np.float32)
        # Tree cells avg per experiment
        INIT_TCELL = np.zeros((l_LHS + 1, self.N_ITERS), dtype=np.float32)
        END_TCELLS = INIT_TCELL.copy()
        END_FCELLS = INIT_TCELL.copy()
        END_R_TCELLS = END_TCELLS.copy()
        #TCELLS = np.zeros(l_LHS + 1, dypet=np.float32) # ACCUMULATE AVG / (TRAN*STEP)
        CHECKPOINTS = []
        # Reseting env and storing the initial observations
        for i in range(l_LHS):
            ENVS.append(H_env.copy())
            OBS.append(observation)
        # making checkpoints
        for i in range(l_LHS + 1):
            CHECKPOINTS += [ENVS[i].make_checkpoint()]
        # Measuring time of execution.
        if save_arr > 0:
            SAVE = True
        else:
            SAVE = False
        start = Time.time()
        self.logger("Run for LHs {} - Metadata: {}\n |".format(LHS, self.metadata_str), True, True, True)
        # First loop to execute an rollout experiment.
        for n_test in range(self.N_ITERS):
            ## Checking time to make a dump file or not
            if SAVE and ((Time.time() - start) > save_arr):
                start = Time.time()
                DUMP_FILES(mod='DUMP@{} '.format(n_test))
            ## Making a checkpoint from the initial state generated.         
            for i in range(l_LHS + 1):
                ENVS[i].load_checkpoint(CHECKPOINTS[i])
            # Saving initial number of cells.
            init_test_tcells = GET_TCELL(ENVS)
            INIT_TCELL[:,n_test] += init_test_tcells
            end_test_fcells = np.zeros(l_LHS + 1,dtype=np.float32)
            # Sync the random generators
            M_SEED = int(self.rg.random()*10**4)
            for i in range(l_LHS + 1):
                ENVS[i].rg = np.random.Generator(np.random.SFC64(M_SEED))
            self.logger(" |-- Test : {} of {}".format(n_test+1, self.N_ITERS))
            # Setting up vars to store costs
            # Making the progress bar
            bar = tqdm.tqdm(range(self.env.moves_before_updating * self.N_STEPS), miniters=0)
            for stp in bar:
                actions = []
                #Calls Heuristic and return best action
                To_H = dict()
                To_H['env'] = ENVS[0]
                To_H['observation'] = OBS[0]
                actions += [self.H(To_H)]
                #Calls Rollout Strategy and returns action,qvalue
                for i in range(1, l_LHS + 1):
                    r_action, _ = Rollout_gpu(
                        ENVS[i], 
                        H=self.H_mode,
                        gamma=self.gamma,
                        epsilon=self.epsilon,
                        K=self.K,
                        lookahead=LHS[i-1],
                        N_samples=self.N_SAMPLES,
                        min_objective=self.min_obj,
                        rg=self.rg)
                    actions += [r_action]
                #Update epsilon it goes from stochastic to deterministic 
                self.epsilon = self.epsilon * self.epsilon_decay
                #Helicopter take an action based on Rollout strategy and heuristic
                for i in range(1 + l_LHS):
                    OBS[i], cost, _, _ = ENVS[i].step(actions[i])
                    COSTS[i,n_test] += cost #Acumulative cost for rollout per LH
                    COSTS_STEP[i,n_test,stp] = COSTS[i,n_test] #List of cost over time
                #Generate a message
                msg =    " |   |      |"
                msg += "\n |   |      |-- Agent step {}".format(stp)
                msg += "\n |   |      |   '-- Actions: {} Costs : {}".format(actions, COSTS_STEP[:, n_test, stp])
                bar.write(msg)
                self.logger(msg, False, False)
                end_test_fcells += GET_TCELL(ENVS, False) / (self.env.moves_before_updating * self.N_STEPS)
            end_test_cells = GET_TCELL(ENVS) 
            END_TCELLS[:,n_test] += end_test_cells
            END_FCELLS[:,n_test] += end_test_fcells
            END_R_TCELLS[:, n_test] += end_test_cells/init_test_tcells
            bar.close()
            msg =    " |   |"
            msg += "\n |   |-- Test {} results".format(n_test+1)
            msg += "\n |   |    '-- Actions: {} Costs : {}".format(actions, COSTS[:, n_test])
            msg += "\n |"
            self.logger(msg)
        msg = " | Run for LHS {} done.".format(LHS)
        msg+= "\nMetadata: {}\n |".format(self.metadata_str)
        self.logger(msg, True, True, True)
        time_s = timeStr()
        # Averaging the results.

        if GRAPH:
            # Making graph here
            checkDir("Runs")

            sns.set(context="paper", style="whitegrid")
            n_rows = ceil(l_LHS / n_cols)
            #fig = plt.figure(figsize=(3*n_cols,2*n_rows),dpi=dpi)
            fig, axs = plt.subplots(n_rows,n_cols, 
                    figsize=(2*n_cols+1,1.7*n_rows),dpi=dpi,sharex=True, sharey=True,
                    gridspec_kw={'hspace': 0, 'wspace': 0})
            fig.suptitle('Rollout Avg. {}/Step'.format(self.mod))

            # ASTHETICS
            y_ticks_rotation = 30
            x_ticks_rotation = 15
            alpha_fill = 0.25
            alpha_line = 0.8
            alpha_p = 0.5
            lw = 2
            l_h = "H_mode {}".format(self.H_mode)
            c_h = sns.xkcd_rgb['cobalt blue']
            alpha_h = 0.9
            alpha_fill_h = 0.09
            mar_s_h = 0.8
            filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
            colors = sns.color_palette("husl", l_LHS + 1)
            alpha_bl = 0.5
            # END

            # First graph - Avg. Cost per step of all test
            # Acumulative cost per step per test
            x_1 = range(self.env.moves_before_updating * self.N_STEPS)
            x_2 = range(1, self.N_ITERS + 1)
            mean_h = np.mean(COSTS_STEP[0], axis=0)
            std_h = np.std(COSTS_STEP[0],axis=0)
            evry = ceil(len(x_1)*0.04)
            
            for i in range(1, l_LHS+1):
                ax = axs[(i-1)//n_cols,(i-1)%n_cols]
                mean = np.mean(COSTS_STEP[i], axis=0)
                std = np.std(COSTS_STEP[i], axis=0)
                l = "Rollout LH {}".format(LHS[i-1])
                c = colors[i]
                m = filled_markers[i]
                ax.errorbar(x_1, mean_h,yerr=std_h,errorevery=evry, label=l_h, alpha=alpha_h, color=c_h, lw=lw//2, ls='-.')
                ax.plot(x_1, mean, label=l, alpha=alpha_line, color=c, lw=lw)
                ax.fill_between(x_1, mean-std, mean+std, alpha=alpha_fill, color=c)
                ax.set(xlabel='Step', ylabel='Average '+ self.mod)
                ax.legend()
            for ax in axs.flat:
                ax.label_outer()
            plt.savefig(
                "./Runs/Rollout avg cost-step LHS {} {} -- {}.png".format(LHS,self.metadata_str, time_s))
            plt.clf() # cleaning figure

            # Doing graph cost per test.
            # Cost per test
            fig = plt.figure(figsize=(5,3),dpi=dpi)
            for i in range(l_LHS + 1):
                if i == 0:
                    l = "H_mode {}".format(self.H_mode)
                    ls_ = '-.'
                    al = alpha_h
                else:
                    l = "Rollout LH {}".format(LHS[i-1])
                    ls_ = '-'
                    al = alpha_line
                plt.plot(x_2, COSTS[i], label=l, color=colors[i], alpha=al, ls=ls_)
            plt.xlabel('Test')
            plt.ylabel(self.mod)
            plt.yticks(rotation=y_ticks_rotation)
            plt.title('Rollout {}/Test'.format(self.mod))
            plt.legend()
            plt.savefig(
                "./Runs/Rollout cost-test LHS {} {} -- {}.png".format(LHS, self.metadata_str, time_s))
            plt.clf()

            # Doing Graph Forest perservation
            c_tree = 'green'
            c_fire = 'red'
            c_ic = 'orange'
            fig = plt.figure(figsize=(2+0.8*l_LHS, 3.3),dpi=dpi)
            X3 = range(l_LHS + 1)
            mean_i_cells = np.mean(INIT_TCELL, axis=1)
            mean_e_cells = np.mean(END_TCELLS, axis=1)
            std_e_cells = np.std(END_TCELLS, axis=1)
            plt.plot(X3, mean_i_cells, label='Initial',ls='dashdot', color=c_ic)
            plt.plot(X3, mean_e_cells, label='Final', color=c_tree)
            plt.plot(X3, [mean_e_cells[0] for i in X3], color=c_tree, ls="--", 
                    alpha=alpha_bl, label="H-baseline")
            plt.fill_between(X3, mean_e_cells - std_e_cells, mean_e_cells + std_e_cells, 
                            alpha=alpha_fill, color=c_tree)
            plt.ylabel("Trees/Total Cells")
            plt.xlabel("Lookahead")
            labels = ['H']
            for i in LHS:
                labels.append(i)
            plt.xticks(X3, labels)
            plt.yticks(rotation=y_ticks_rotation)
            #plt.xticks(rotation=x_ticks_rotation)
            plt.legend()
            plt.title("Forest Perservation Average Area - H_mode {}".format(self.H_mode))
            plt.savefig(
                "./Runs/Rollout TCELL LHS {} {} -- {}.png".format(LHS, self.metadata_str, time_s))
            plt.clf()

            #Graph fire density
            fig = plt.figure(figsize=(2+0.8*l_LHS, 3.3),dpi=dpi)
            mean_e_cells = np.mean(END_FCELLS, axis=1)
            std_e_cells = np.std(END_FCELLS, axis=1)
            plt.plot(X3, 100*mean_e_cells, color=c_fire, label='Fire')
            plt.plot(X3, [mean_e_cells[0]*100 for i in X3], color=c_fire, ls="--", 
                    alpha=alpha_bl, label='H-baseline')
            plt.fill_between(X3, 100*(mean_e_cells - std_e_cells), 100*(mean_e_cells + std_e_cells), 
                            alpha=alpha_fill, color=c_fire)
            plt.ylabel("Fire Density (%)")
            plt.xlabel("Lookahead")
            labels = ['H']
            for i in LHS:
                labels.append(i)
            plt.xticks(X3, labels)
            plt.yticks(rotation=y_ticks_rotation)
            #plt.xticks(rotation=x_ticks_rotation)
            plt.legend()
            plt.title("Wildland Average Fire Density - H_mode {}".format(self.H_mode))
            plt.savefig(
                "./Runs/Rollout FCELL LHS {} {} -- {}.png".format(LHS, self.metadata_str, time_s))
            plt.clf()

        if True:
            DUMP_FILES()
        return None

    def make_graph(self, title_head="Rollout", dpi = 200):
        checkDir("Runs")

        self.runs_rollout_archive += [self.runs_rollout_results_step]
        self.runs_heu_archive += [self.runs_heu_results_step]

        meta = self.metadata_str
        make_graph(self.runs_rollout_results_step,
                    self.runs_heu_results_step,
                    title_head= title_head + " Run Average",
                    unitX = "step",
                    unitY = "cost",
                    dpi = dpi)

        saveFig( "./Runs/Rollout avg cost-step {}".format(meta))

        make_graph(self.runs_rollout_results, 
                    self.runs_heu_results,
                    mean = False,
                    title_head = title_head + " Test Average",
                    unitX = "test",
                    unitY = "cost",
                    dpi = dpi)
        saveFig( "./Runs/Rollout avg cost-test {}".format(meta))
        
        # Clean the buffers
        self.runs_rollout_results = []
        self.runs_rollout_results_step = []
        self.runs_heu_results = []
        self.runs_heu_results_step = []

    def make_gif(self, RUN=False, TEST=False, fps=5):
        """
        Make and save .gif files from all the agent steps done in the environment with 
        the rollout and heuristic choices in the last train of the last run done.
        For this function to work is necessary to indicate on the initial vairbales 
        RUN_GIF = True, to to check it when about to make a .run(GIF=True)
        
        Parameters
        ----------
        RUN : bool
            If True, generates a .gif from the rollout agent from the last run. It's the last due to resouces
            management.
        TEST : bool
            If True, and a .policy_test() has been executed, then it generates the gif for the best run test.
        """
        checkDir("Runs")
        time_s = timeStr()
        if RUN and (self.theres_run_gif):
            self.logger("Creating gif for runs. This may take a while.",time_delta=True)
            imageio.mimsave("./Runs/Helicopter Rollout Run {} -- {}.gif".format(self.metadata_str, time_s), 
                self.frames_run_r, fps=fps)
            self.frames_run_r = []
            imageio.mimsave("./Runs/Helicopter Heuristic Run -- H_mode {} -- {}.gif".format(self.H_mode, time_s),
                self.frames_run_h, fps=fps)
            self.frames_run_h = []
            self.theres_run_gif = False
            self.logger("Run gif. Done!\n",time_delta=True)
        if TEST and self.theres_test_gif:
            self.logger("Creating gif for tests. This may take a while.",time_delta=True)
            imageio.mimsave("./Runs/Helicopter Rollout Test {} -- {}.gif".format(self.metadata_str, time_s),
                self.frames_test_r, fps=fps)
            self.frames_test_r = []
            self.theres_test_gif = False
            self.logger("Test gif. Done!\n",time_delta=True)

    @property
    def metadata_str(self):
        msg = "LH-{} K-{} H_mode-{} C_mode-{} N_SAMPLES-{} GAMMA-{} MCTS_P-{} SEED-{}".format(
            self.LOOKAHEAD, self.K, self.H_mode, self.env.reward_type, self.N_SAMPLES, self.gamma, 
            self.mcts_p, self.seed)
        return msg