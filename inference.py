'''Collection of inference for decision-making algorithms'''
import numpy as np


def value_iteration(
    gridworld,
    U: dict,
    U_prime: dict,
    gamma,
    epsilon
):
    '''
    AI: A Modern Approach (pg. 653)
    '''
    iterations = 0
    delta = np.inf
    while delta > ((epsilon * (1 - gamma)) / gamma):
        delta = 0
        U.update(U_prime)
        
        for state in gridworld.all_valid_states:
            EU = []
            for action in gridworld.available_actions:
                sp_d = gridworld.transition_model(state, action)
                if len(sp_d) == 0:
                    EU.append(0)
                else:
                    s_prime, s_prime_prob = tuple(sp_d.keys())[0], list(sp_d.values())[0]
                    EU.append(s_prime_prob * U_prime[s_prime])
            MEU = max(EU)
            # print(MEU)
         
            U_prime[state] = gridworld.state_utilities[state] + gamma * MEU
            
            delta = max(abs(U_prime[state] - U[state]), delta)   
        
        iterations += 1
        
    print(f'Iterations: {iterations}, Delta: {delta}, Stopping Criterion: {((epsilon * (1 - gamma)) / gamma)}')    
    return U, U_prime