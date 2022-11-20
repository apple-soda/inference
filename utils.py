import numpy as np


def init_policy_mdp(gridworld):
    '''
    Initialize a random policy to test policy iteration for a MDP gridworld
    
    In a determinisitc MDP grid world, the iterative algorithm reduces to the reward 
    
    π∗(s) = argmax a∈A(s)sP(s | s, a)U(s)
    
    How to represent this policy?
        Initialize the estimated utility function to be random
    '''
    
    # random intialization of the utility function
    U, pi = {}, {}
    for state in gridworld.all_valid_states:
        U[state] = np.random.random()
    
    for state in gridworld.all_valid_states:
        # technically incorrect implementation for now since we need to sum over all possible states, but since our gridworld
        # of focus is determinisitc there is only one state
        EU_next_states = []
        for i in range(4):
            s_prime = gridworld.transition_model(state, i)
            
            if len(s_prime) == 0:
                EU_next_states.append(0.0)
            else:   
                s_prime, s_prime_p = tuple(s_prime.keys())[0], list(s_prime.values())[0]
                EU = s_prime_p * U[s_prime]
                EU_next_states.append(EU)
            
        MEU_action = np.argmax(EU_next_states)
        pi[state] = MEU_action
        
    return U, pi