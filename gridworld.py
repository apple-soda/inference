import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# make 'validity' function (to check all valid states from the current state)
# when at the bottom/corners, the transition model should return the same spot if you try to make a move that bounces into the wall
# make the transition matrices reflect this
# inaccurate utilitiy estimations rn because of no termination condition in the current gridworld implementation


class GridWorld:
    '''
    Toy grid-world for visualizing MDPs/POMDPs and other inference problems
    
    MDP:
        P(s'|s, a) - transition model, deterministic or noisy
        R(s) - reward function
        
    Actions:
        0: right
        1: left
        2: down
        3: up
    '''
    def __init__(
        self,
        h,
        w,
        terminal_squares: dict,
        barriers=None
    ):
        self.h, self.w, self.grid_size = h, w, h * w
        self.all_states = [(i, j) for i in range(h) for j in range(w)]
        self.all_valid_states = [i for i in self.all_states if i not in barriers]
        self.barriers = barriers
        self.terminal_squares = terminal_squares
        
        self.available_actions = [0, 1, 2, 3]
        
        ### GRID INIT  ###
        self.state_utilities = self._init_utilities(terminal_squares)
        self.valid_right, self.valid_left, self.valid_down, self.valid_up = self._get_valid_actions()
        self.state_to_idx, self.idx_to_state = self._state_tm_mappings()
        self.transition_matrix = self._create_transition_matrix()
        self.current_state = (0, 0)
        
        # VISUALS INIT ###
        self.fig, self.ax = self._create_plt_grid()
        
    def step(self, action):
        '''
        Transition matrix returns an array [s_0,...,s_n] of probabilities of transition from the current state to that 
        respective state
        
        In a deterministic grid world, the agent has complete control over its actions and the probabilitiy array is a sparse
        array that is 0 everywhere except the state it intends to go to (with probability = 1)
        '''
        transition_idx = self.state_to_idx[self.current_state]
        next_state = self.transition_matrix[action, transition_idx]
        
        # np.random.choice in the future
        next_state = np.where(next_state == 1)[0][0] if np.where(next_state == 1)[0].size != 0 else None
        next_state = self.idx_to_state.get(next_state, None)
        
        if next_state:
            # if the action is valid given the current state, move the agent to the next tile
            self.current_state = next_state
            self._plt_move_agent(next_state)
            reward = self.state_utilities[next_state]
            return next_state, reward
        else:
            # else keep the agent in the same tile
            reward = self.state_utilities[self.current_state]
            return self.current_state, reward
        
    
    # GRIDWORLD INITIALIZATION FUNCTIONS
    def _init_utilities(self, terminal_squares):
        state_utilities = {}
        for state in self.all_states:
            state_utilities[state] = -0.04
            
        # terminal "good" square for now, add flexibility later
        for ts, tsv in terminal_squares.items():
            state_utilities[ts] = tsv
            
        return state_utilities
        
    def _get_valid_actions(self):
        # fix these mayne
        valid_right, valid_left, valid_down, valid_up = [], [], [], []
        
        for i, j in zip(range(self.grid_size), range(self.grid_size)):
            transition = [i, j + 1] if (i + 1) % self.w != 0 else [i, j]
            valid_right.append(transition)
            
        for i, j in zip(range(self.grid_size), range(self.grid_size)):
            transition = [i, j - 1] if i % self.w != 0 else [i, j]
            valid_left.append(transition)
          
        for i, j in zip(range(self.grid_size), range(self.grid_size)):
            transition = [i, j - self.w] if i >= self.w else [i, j]
            valid_down.append(transition)
            
        for i, j in zip(range(self.grid_size), range(self.grid_size)):
            transition = [i, j + self.w] if i < self.grid_size - self.w else [i, j]
            valid_up.append(transition)
            
        return valid_right, valid_left, valid_down, valid_up
    
    def _state_tm_mappings(self):
        state_to_idx, idx_to_state = {}, {}
        for i, state in enumerate(self.all_states):
            state_to_idx[state] = i
            idx_to_state[i] = state
            
        return state_to_idx, idx_to_state
        
    def _create_transition_matrix(self):
        transition_matrix = []
        
        for v in [self.valid_right, self.valid_left, self.valid_down, self.valid_up]:
            transition_matrix_slice = np.zeros((self.grid_size, self.grid_size))
            for i in v:
                transition_matrix_slice[i[0], i[1]] = 1
            transition_matrix.append(transition_matrix_slice)
            
        transition_matrix = np.dstack(transition_matrix)
        transition_matrix = transition_matrix.transpose(2, 0, 1)
        
        self.transition_matrix = transition_matrix # so i can call _get_possible_next_states
        
        # modify transition_matrices according to barriers
        # find all adjacent states and modify the transition matrices in those particular positions to "bump" into the wall
        # and return the same state its currently in
        for barrier in self.barriers:
            barrier_tm_idx = self.state_to_idx[barrier]
            for state in self._get_possible_next_states(barrier):
                state_tm_idx = self.state_to_idx[state]
                
                try:
                    modified_tm_idx = int(np.where(transition_matrix[:, state_tm_idx, barrier_tm_idx] != 0)[0])
                except:
                    raise TypeError('Invalid Barrier Location')
                
                transition_matrix[modified_tm_idx, state_tm_idx, barrier_tm_idx] = 0
                transition_matrix[modified_tm_idx, state_tm_idx, state_tm_idx] = 1
                
        # set all transitions to 0 for the terminal states (since the world 'ends' when these states are reached)
        for ts in self.terminal_squares.keys():
            ts_tm_idx = self.state_to_idx[ts]
            transition_matrix[:, ts_tm_idx] = 0
                
        
        return transition_matrix
    
    def transition_model(self, state, action):
        '''
        P(s'|s, a)
        
        Returns: dictionary of all possible next states given a state and an action and their corresponding 
        transition probabilities
        '''
        assert state in self.all_states
        transition_matrix_index = self.state_to_idx[state]
        possible_transition = np.where(self.transition_matrix[action, transition_matrix_index] != 0)[0]
        
        res = {}
        for i in possible_transition:
            corresponding_probability = self.transition_matrix[action, transition_matrix_index, i]
            possible_state = self.idx_to_state[i]
            res[possible_state] = corresponding_probability
            
        return res
        
        
    def _get_possible_next_states(self, state):
        assert state in self.all_states
        transition_matrix_index = self.state_to_idx[state]
        possible_transitions = self.transition_matrix[:, transition_matrix_index]
        possible_transitions = np.where(possible_transitions != 0)[1]
        
        res = []
        for i in possible_transitions:
            possible_state = self.idx_to_state[i]
            res.append(possible_state)
            
        return list(set(res))
    
    # GRIDWORLD VISUAL INITIALIZATION FUNCTIONS (might move to utils.py file or something)
    def _create_plt_grid(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        
        xs, ys = np.linspace(0, self.w, self.w + 1), np.linspace(0, self.h, self.h + 1)
        w, h = xs[1] - xs[0], ys[1] - ys[0]
        for i, x in enumerate(xs[:-1]):
            for j, y in enumerate(ys[:-1]):
                if self.barriers and (int(x), int(y)) in self.barriers:
                    color, alpha = 'black', 1.0
                else:
                    color, alpha = 'white', 0.1
                ax.add_patch(Rectangle((x, y), w, h, fill=True, color=color, alpha=alpha))
                             
        for x in xs:
            ax.plot([x, x], [ys[0], ys[-1]], color='black', alpha=0.33, linestyle=':')
        for y in ys:
            ax.plot([xs[0], xs[-1]], [y, y], color='black', alpha=0.33, linestyle=':')
            
        return fig, ax
    
    def _plt_move_agent(self, next_state): # need to figure out how to remove single points at each step
        # self.ax.remove()
        self.ax.scatter(next_state[1] + 0.5, next_state[0] + 0.5)