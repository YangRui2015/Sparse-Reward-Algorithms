import numpy as np
from .normalize import Normalizer


class ExperienceBuffer():
    def __init__(self, max_buffer_size, batch_size, if_normalize=False):
        self.size = 0
        self.max_buffer_size = max_buffer_size
        self.experiences = []
        self.batch_size = batch_size
        self.if_normalize = if_normalize
        self.state_norm = None
        self.goal_norm = None

    def add(self, experience):
        assert len(experience) == 7, 'Experience must be of form (s, a, r, s, g, t, grip_info\')'
        assert type(experience[5]) == bool

        if self.if_normalize:
            if self.state_norm is None and self.goal_norm is None:
                self.state_norm = Normalizer(len(experience[0]))
                self.goal_norm = Normalizer(len(experience[4]))

            self.state_norm.update(np.vstack((experience[0], experience[3])))
            self.goal_norm.update(experience[4])

        self.experiences.append(experience)
        self.size += 1

        if self.size >= self.max_buffer_size:
            beg_index = int(np.floor(self.max_buffer_size/10))                     # 删除了1/6的buffer
            self.experiences = self.experiences[beg_index:]
            self.size -= beg_index

    def get_batch(self):
        states, actions, rewards, new_states, goals, is_terminals = [], [], [], [], [], []
        dist = np.random.randint(0, high=self.size, size=self.batch_size)
        
        for i in dist:
            if self.if_normalize and (self.state_norm is not None) and (self.goal_norm is not None):
                states.append(self.state_norm.normalize(self.experiences[i][0]))
                new_states.append(self.state_norm.normalize(self.experiences[i][3]))
                goals.append(self.goal_norm.normalize(self.experiences[i][4]))
            else:
                states.append(self.experiences[i][0])
                new_states.append(self.experiences[i][3])
                goals.append(self.experiences[i][4])

            actions.append(self.experiences[i][1])
            rewards.append(self.experiences[i][2])
            is_terminals.append(self.experiences[i][5])

        return states, actions, rewards, new_states, goals, is_terminals

    def clear(self):
        self.size = 0
        self.experiences = []
        if self.if_normalize and (self.state_norm is not None) and (self.goal_norm is not None):
            self.state_norm.reset()
            self.goal_norm.reset()
