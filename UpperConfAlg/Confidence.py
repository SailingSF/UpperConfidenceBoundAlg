#Upper Confidence Bound Algorithm Main Script

# Importing Libraries
import numpy as np
import pandas as pd
import math




def UCB(dataset):
    '''
    Function for using an Upper Confidence Bound on a dataset
    Dataset can be any amount of instances with any amount of variables
    Dataset must be size of n observations by d variables
    Dataset shows success as 1 at [n][d] and failure as 0
    1 is reward 0 is nothing

    '''
    def delta_i_func(n, selections):
        '''
        Function for finding delta_i which is used in UCP computation
        '''
        return math.sqrt(3/2 * math.log(n + 1) / selections)

    # Constants for shapes
    N = len(dataset)
    d = len(dataset.columns)
    total_reward = 0
    numbers_of_selections = [0] * d # Total selections vector
    sums_of_rewards = [0] * d # Sum of rewards vector
    ads_selected = []
    # Implementing UCB
    for n in range(0, N):
        ad = 0
        max_upper_bound = 0
        for i in range(0, d):
            if (numbers_of_selections)[i] > 0:
                average_reward = sums_of_rewards[i]/numbers_of_selections[i]
                delta_i = delta_i_func(n, numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] = numbers_of_selections[ad] + 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward
        total_reward = total_reward + reward
        # Ending iterations after one variable has been found to be better than rest
        # Params for when to look and when value is much higher should be tuned
        if n > 0.25*N and max(numbers_of_selections) > 1.25*max(n for n in numbers_of_selections if n!=max(numbers_of_selections)):
            print(f'The best variable in the set is {ad}')
            break
    print(f"The total rewards found are {total_reward} in {n} iterations after stopping having found the {ad} is the best variable with an upper bound of {upper_bound}")
    return ad, upper_bound, numbers_of_selections


    
    