"""
Forest fire cuda version

@author: Bobobert
"""

from CAFFR.utils.consts import *

@cuda.jit(device=True)
def helicopter_step(grid, 
                    updated_grid, 
                    crn,
                    pos_row, pos_col, 
                    steps_before_update, 
                    action, 
                    parameters, 
                    probs, 
                    costs,):
    """
    ForestFire Cuda version

    Function on to execute on device per thread to update
    and give a step of the agent in its environment. 
    
    This is meant
    to run inside the sample_trajectories kernel.
    """
    new_steps_before_updating = 0
    empty_, tree_, fire_ = parameters[10], parameters[11], parameters[12]

    # Check if the grid needs to be updated
    if steps_before_update == 0:
        # Generating the random values
        throws = crn
        # Begin the update of the grid.
        for i in range(parameters[8]):
            for j in range(parameters[9]):
                # From the cell padded_grid[i,j]
                # If it's a fire, spread on its neighborhood
                if grid[i,j] == fire_:
                    # Get the neighborhood
                    for i_n in range(i-1, i+2):
                        for j_n in range(j-1, j+2):
                            if (i_n < 0) or (i_n >= parameters[8]):
                                0.0 #Out of bounds
                            elif (j_n < 0) or (j_n >= parameters[9]):
                                0.0 # Out of bounds
                            elif grid[i_n,j_n] == tree_:
                                # Burn the tree
                                updated_grid[i_n,j_n] = fire_
                    # Extinguish the fire
                    updated_grid[i,j] = empty_
                # If it's a tree, throw a dice to a lighting to
                # hit it or not
                elif (grid[i,j] == tree_) and (throws[i,j] <= probs[0]):
                        # The tree is hitted by a ligthning
                        updated_grid[i,j] = fire_
                # If the cell it's empty, it has a chance to grow a tree
                elif (grid[i,j] == empty_) and (throws[i,j] <= probs[1]):
                        # A tree growns in this cell
                        updated_grid[i,j] = tree_
        new_steps_before_updating = parameters[3] # Restarting

    else:
        new_steps_before_updating = steps_before_update - 1
    # End of the grid update

    # Start of the agent movements
    # By design, if the grid changes happens at the same time that the
    # grids updates. So it it was to put down a fire, this will dissapear
    # and wont count.
    it_moved = 1.0
    delta_row, delta_col = 0, 0
    if action == 5:
        0.0
    elif (action == 1) or (action == 2) or (action == 3):
        delta_row = -1
    elif (action == 7) or (action == 8) or (action == 9):
        delta_row = 1
    if (action == 1) or (action == 4) or (action == 7):
        delta_col = -1
    elif (action == 3) or (action == 6) or (action == 9):
        delta_col = 1

    new_pos_row = pos_row + delta_row
    new_pos_col = pos_col + delta_col

    if (new_pos_row < 0) or (new_pos_row >= parameters[8]):
        # Invalid movement - out of bounds
        new_pos_row = pos_row
    if (new_pos_col < 0) or (new_pos_col >= parameters[9]):
        # Invalid movement - our of bounds
        new_pos_col = pos_col
    if (pos_row == new_pos_row) and (pos_col == new_pos_col):
        # There was no movement, the action was out of bounds
        it_moved = 0.0
    # End of agent movement

    # Start to check the hits
    hits = 0.
    if updated_grid[new_pos_row, new_pos_col] == fire_:
        updated_grid[new_pos_row, new_pos_col] = empty_
        hits += 1.0
    # End of hits
    
    # Start to counting the cells
    fires, empties, trees = 0.0 ,0.0 ,0.0
    f, t, ratio = 0.0, 0.0, 0.0
    for i in range(parameters[8]):
        for j in range(parameters[9]):
            if updated_grid[i,j] == fire_:
                fires += 1.
            elif updated_grid[i,j] == tree_:
                trees += 1. 
            elif updated_grid[i,j] == empty_:
                empties += 1.
            if grid[i,j] == tree_:
                t += 1
            elif grid[i,j] == fire_:
                f += 1
    if f > 0:
        ratio = t/f
    else:
        ratio = parameters[8]*parameters[9]*1.0
    ratio = -1.0*ratio
    # End of counting

    # Calculating cost.
    ### This is the cost shape for a given state ### 
    c_mode = parameters[7]
    cost = 0.0
    if c_mode == 0:
        ### This is the same as 'custom' type on Helicopter Env ###
        cost += costs[0]*trees
        cost += costs[1]*fires**2
        cost += costs[2]*empties
        cost += costs[3]*hits
        cost += costs[4]*it_moved
    elif c_mode == 1:
        #Quad mode
        diff = (trees-fires)
        if diff < 0:
            cost += -1.0 * diff**2 * costs[0]
        else:
            cost += diff**2 * costs[0]
        cost += costs[2]*empties
        cost += costs[3]*(2.0*hits-1.0*it_moved)
    elif c_mode == 2:
        #Ratio mode cost here
        #cost += costs[0]*math.exp(ratio)
        #cost += -1.0*costs[0]*ratio
        #cost += costs[2]*empties
        #cost += costs[3]*(2.0*hits-1.0*it_moved)
        if steps_before_update == 0:
            rr = parameters[8]*parameters[9]*1.0
            cost += 1.9 * trees / rr - 0.9
        cost += costs[4] * it_moved
    elif c_mode == 3:
        # hit w/ratio REWARD
        tot =  0.5 / (parameters[8] * parameters[9])
        cost += trees * tot
        cost += hits * costs[3]
    elif c_mode == 4:
        if steps_before_update == 0:
            rr = parameters[8]*parameters[9]*1.0
            cost += 1.8 * trees / rr - 0.9
        cost += costs[4] * it_moved
        cost += costs[3] * hits
    # End of cost

    return updated_grid, new_pos_row, new_pos_col, new_steps_before_updating, cost