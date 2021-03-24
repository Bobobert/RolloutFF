"""
    Heuristics programmed for CUDA calls

    @authors: Bobobert, MauMontenegro
"""

from CAFFR.utils.consts import *

###### PROGRAM HERE YOUR HEURISTICs ######
### ASING THEM INTO A SWITH MODE USE #####
@cuda.jit(device=True)
def Heuristic(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker):
    h_mode = parameters[6]
    if h_mode == 0:
        # dummy heuristic, by rob. It took around 5 hrs to write this one
        return 3
    elif h_mode == 11:
        return CONSERVATIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 1)
    elif h_mode == 12:
        return CONSERVATIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 2)
    elif h_mode == 13:
        return CONSERVATIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 3)
    elif h_mode == 21:
        return PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 1)
    elif h_mode == 22:
        return PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 2)
    elif h_mode == 23:
        return PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, 3)

@cuda.jit(device=True)
def CONSERVATIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, VISION):
    # Heuristic from Mau
    # Corrective, vision = 2
    action = 5
    NS = 3 + 2 * VISION
    cb = cuda.local.array(9, dtype=NBTINT)
    for i in range(9):
        cb[i] = 0
    # top_left top  top_right  left  --  rigth down_left  down  down_right
    #    0      1       2        3    4    5       6       7        8
    # Count the burned cells on the neighborhood
    for i in range(pos_row - VISION, pos_col + VISION + 1):
        for j in range(pos_col - VISION, pos_col + VISION + 1):
            if (i < 0) and (i >= GRID_SIZE[0]):
                0.0
            elif (j < 0) and (j >= GRID_SIZE[1]):
                0.0 # Out of boundaries
            elif local_grid[i,j] == FIRE:
                if (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < NS):
                    #Up zone
                    cb[1] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < 1 + VISION):
                    #Up left zone
                    cb[0] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j > 1 + VISION) and (j < NS):
                    #Up rigth zone
                    cb[2] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0 ) and (j < NS):
                    # Down zone
                    cb[7] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Down left zone
                    cb[6] += 1
                elif (i > 1 + VISION) and (i < NS) and (j > 1 + VISION) and (j < NS):
                    #Down right zone
                    cb[8] += 1
                elif (i >= 0) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Left zone
                    cb[3] += 1
                elif (i >= 0) and (i < NS) and (j > 1 + VISION ) and (j < NS):
                    #Right zone
                    cb[5] += 1
    non_zero = 0
    for i in range(9):
        if cb[i] > 0:
            non_zero += 1
    if non_zero == 0:
        for i in range(1, 10):
            if rdm_uniform_sample(random_states, worker) < 0.112:
                action = i
    else:
        p = 1 / non_zero
        max_yeet = 0        
        for i in range(9):
            if cb[i] > max_yeet:
                action = i + 1
                max_yeet = cb[i]
            elif cb[i] == max_yeet:
                if rdm_uniform_sample(random_states, worker) < p:
                    action = i + 1
    return action

@cuda.jit(device=True)
def PREVENTIVE(local_grid, pos_row, pos_col, steps_to_update, parameters, probs, costs, random_states, worker, VISION):
    # Heuristic preventive variable vision
    # based on Mau's
    action = 5
    VISION = 1
    NS = 3 + 2 * VISION
    cb = cuda.local.array(9, dtype=NBTFLOAT)
    cz = cuda.local.array(9, dtype=NBTINT)
    for i in range(9):
        cb[i] = 0.0
        cz[i] = 0
    # Function coefficients
    Fire_coef = 2.0
    Tree_coef = 0.5
    Empty_coef = 0.5
    # top_left top  top_right  left  --  rigth down_left  down  down_right
    #    0      1       2        3    4    5       6       7        8
    # Count the burned cells on the neighborhood
    for i in range(pos_row - VISION, pos_col + VISION + 1):
        for j in range(pos_col - VISION, pos_col + VISION + 1):
            if (i < 0) and (i >= GRID_SIZE[0]):
                0.0
            elif (j < 0) and (j >= GRID_SIZE[1]):
                0.0 # Out of boundaries
            else:
                if (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < NS):
                    #Up zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[1] += coef
                    cz[1] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j >= 0) and (j < 1 + VISION):
                    #Up left zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[0] += coef
                    cz[0] += 1
                elif (i >= 0 ) and (i < 1 + VISION) and (j > 1 + VISION) and (j < NS):
                    #Up rigth zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[2] += coef
                    cz[2] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0 ) and (j < NS):
                    # Down zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[7] += coef
                    cz[7] += 1
                elif (i > 1 + VISION) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Down left zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[6] += coef
                    cz[6] += 1
                elif (i > 1 + VISION) and (i < NS) and (j > 1 + VISION) and (j < NS):
                    #Down right zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[8] += coef
                    cz[8] += 1
                elif (i >= 0) and (i < NS) and (j >= 0) and (j < 1 + VISION):
                    #Left zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[3] += coef
                    cz[3] += 1
                elif (i >= 0) and (i < NS) and (j > 1 + VISION ) and (j < NS):
                    #Right zone
                    coef = 0.0
                    if local_grid[i, j] == FIRE:
                        coef += Fire_coef
                    elif local_grid[i, j] == TREE:
                        coef += Tree_coef
                    elif local_grid[i, j] == EMPTY:
                        coef -= Empty_coef
                    cb[5] += coef
                    cz[5] += 1
    for i in range(9):
        # Normalize the coefficients
        if cz[i] != 0:
            cb[i] = cb[i] / cz[i]
        else:
            cb[i] = 0
    non_zero = 0
    for i in range(9):
        if cz[i] > 0:
            non_zero += 1
    if non_zero == 0:
        for i in range(1, 10):
            if rdm_uniform_sample(random_states, worker) < 0.112:
                action = i
    else:
        p = 1 / non_zero
        max_yeet = 0.0
        for i in range(9):
            if cb[i] > max_yeet:
                action = i + 1
                max_yeet = cb[i]
            elif cb[i] == max_yeet:
                if rdm_uniform_sample(random_states, worker) < p:
                    action = i + 1
    return action
############## END OF IT ################