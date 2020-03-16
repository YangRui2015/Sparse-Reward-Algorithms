import numpy as np
from .experience_buffer import ExperienceBuffer
from .actor import Actor
from .critic import Critic
from .normalize import Normalizer
from .Curiosity import ForwardDynamics

class Layer():
    def __init__(self, layer_number, FLAGS, env, sess, agent_params):
        self.layer_number = layer_number
        self.FLAGS = FLAGS
        self.sess = sess

        # time limit = max action if len(layers) = 1
        if FLAGS.layers > 1:
            self.time_limit = FLAGS.time_scale      
        else:
            self.time_limit = env.max_actions

        self.current_state = None
        self.goal = None
        self.buffer_size_ceiling = 10**6
        self.episodes_to_store = agent_params["episodes_to_store"]
        self.num_replay_goals = 4

        # Number of the transitions created 
        if self.layer_number == 0:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit
        else:
            self.trans_per_attempt = (1 + self.num_replay_goals) * self.time_limit + int(self.time_limit/3)

        # Buffer size 
        self.buffer_size = min(self.trans_per_attempt * self.time_limit**(self.FLAGS.layers-1 - self.layer_number) * self.episodes_to_store, self.buffer_size_ceiling)
        self.batch_size = agent_params["batch_size"]    # 太大了，以至于没有学习 1024,12
        self.replay_buffer = ExperienceBuffer(self.buffer_size, self.batch_size, FLAGS.normalize)
        
        # imitation buffer
        if FLAGS.imitation:
            self.imitation_buffer = ExperienceBuffer(self.buffer_size, agent_params['imit_batch_size'], FLAGS.normalize)

        self.temp_goal_replay_storage = []
        self.actor = Actor(sess, env, self.batch_size, self.layer_number, FLAGS, imit_batch_size=agent_params['imit_batch_size'],imit_ratio=agent_params['imit_ratio'])
        self.critic = Critic(sess, env, self.layer_number, FLAGS)
        self.epsilon = agent_params["epsilon"]
        if self.layer_number == 0:
            self.noise_perc = agent_params["atomic_noise"]
        else:
            self.noise_perc = agent_params["subgoal_noise"]

        self.maxed_out = False
        self.subgoal_penalty = agent_params["subgoal_penalty"]
        self.normalize_state = None
        self.normalize_goal = None
        if self.FLAGS.curiosity:
            self.cur_model = ForwardDynamics(env.state_dim, env.action_dim, name=str(self.layer_number))
        self.normalize_inreward = Normalizer(size=1)

    # Add noise to actions
    def add_noise(self,action, env):
        for i in range(len(action)):
            if self.layer_number == 0:
                action[i] += np.random.normal(0, self.noise_perc * env.action_bounds)
                action[i] = max(min(action[i], env.action_bounds + env.action_offset), -env.action_bounds + env.action_offset)
            else:
                action[i] += np.random.normal(0, self.noise_perc * env.subgoal_bounds_symmetric[i])
                action[i] = max(min(action[i], env.subgoal_bounds_symmetric[i] + env.subgoal_bounds_offset[i]), -env.subgoal_bounds_symmetric[i] + env.subgoal_bounds_offset[i])
        return action

    # randomly select action
    def get_random_action(self, env):
        if self.layer_number == 0:
            action = np.zeros((env.action_dim))
        else:
            action = np.zeros((env.subgoal_dim))

        for i in range(len(action)):
            if self.layer_number == 0:
                action[i] = np.random.uniform(-env.action_bounds + env.action_offset, env.action_bounds + env.action_offset)
            else:
                action[i] = np.random.uniform(env.subgoal_bounds_symmetric[i] - env.subgoal_bounds_offset[i],env.subgoal_bounds_symmetric[i] + env.subgoal_bounds_offset[i])
        return action


    # select action using an epsilon-greedy policy
    def choose_action(self,agent, env,test, subgoal_test):
        # If testing, no noise
        if test or subgoal_test:
            if self.FLAGS.normalize and (self.normalize_state is not None) and (self.normalize_goal is not None):
                return self.actor.get_action(np.reshape(self.normalize_state,(1,len(self.current_state))), np.reshape(self.normalize_goal,(1,len(self.goal))))[0], "Policy", subgoal_test
            return self.actor.get_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0], "Policy", subgoal_test
        else:
            if np.random.random_sample() > self.epsilon:
                if self.FLAGS.normalize and (self.normalize_goal is not None) and (self.normalize_state is not None):
                    action = self.actor.get_action(np.reshape(self.normalize_state,(1,len(self.current_state))), np.reshape(self.normalize_goal,(1,len(self.goal))))[0]
                else:
                    action = self.actor.get_action(np.reshape(self.current_state,(1,len(self.current_state))), np.reshape(self.goal,(1,len(self.goal))))[0]
                action = self.add_noise(action, env)
                action_type = "Noisy Policy"
            else:
                action = self.get_random_action(env)
                action_type = "Random"

            # Determine whether to test upcoming subgoal
            if np.random.random_sample() < agent.subgoal_test_perc:
                next_subgoal_test = True
            else:
                next_subgoal_test = False
            return action, action_type, next_subgoal_test


    def get_project_goal(self, state, env, total_layers):
        if self.layer_number == 0:                                  
            hindsight_goal = env.project_state_to_end_goal(state)
        else:
            hindsight_goal = env.project_state_to_subgoal(state)
        return hindsight_goal

    # add data to replay buffer
    def perform_action_replay(self, hindsight_action, next_state, goal_status, env, rtype='sparse'):           
        if goal_status[self.layer_number]:
            reward = 1
            finished = True
        else:
            reward = -1
            finished = False
        # reward shaping, only use shaping rewards
        if rtype == 'dense':
            hindsight_goal0 = self.get_project_goal(self.current_state, env, len(goal_status))
            hindsight_goal1 = self.get_project_goal(next_state, env, len(goal_status))
            reward0 = self.get_reward(self.goal, hindsight_goal0, rtype=rtype)
            reward1 = self.get_reward(self.goal, hindsight_goal1, rtype=rtype)
            reward = (reward1 - reward0) * 1000

        transition = [self.current_state, hindsight_action, reward, next_state, self.goal, finished, None]
        self.replay_buffer.add(np.copy(transition))


    # Create transitions for HER
    def create_prelim_goal_replay_trans(self, hindsight_action, next_state, env, total_layers):
        hindsight_goal = self.get_project_goal(next_state, env, total_layers)
        transition = [self.current_state, hindsight_action, None, next_state, None, None, hindsight_goal]
        self.temp_goal_replay_storage.append(np.copy(transition))


    # reward function
    def get_reward(self,new_goal, hindsight_goal, goal_thresholds=None, rtype='sparse'):
        assert len(new_goal) == len(hindsight_goal), "Goal, hindsight goal, and goal thresholds do not have same dimensions"
        if rtype == 'sparse':
            for i in range(len(new_goal)):
                if np.absolute(new_goal[i]-hindsight_goal[i]) > goal_thresholds:
                    return -1
            # Else goal is achieved
            return 0
        elif rtype == 'dense':
            return - np.square(new_goal - hindsight_goal).sum()
        else:
            raise NotImplementedError

    # finally add data to buffer for HER
    def finalize_goal_replay(self, goal_thresholds):
        num_trans = len(self.temp_goal_replay_storage)
        num_replay_goals = self.num_replay_goals                       
        if num_trans < self.num_replay_goals:
            num_replay_goals = num_trans

        indices = np.zeros((num_replay_goals))
        indices[:num_replay_goals-1] = np.random.randint(num_trans,size=num_replay_goals-1)     # future mode
        indices[num_replay_goals-1] = num_trans - 1
        indices = np.sort(indices)

        for i in range(len(indices)):
            trans_copy = np.copy(self.temp_goal_replay_storage)
            new_goal = trans_copy[int(indices[i])][6]
            for index in range(num_trans):
                # Update goal 
                trans_copy[index][4] = new_goal
                # Update reward
                trans_copy[index][2] = self.get_reward(new_goal, trans_copy[index][6], goal_thresholds, self.FLAGS.rtype)
                # Update finished boolean 
                if trans_copy[index][2] == 0:
                    trans_copy[index][5] = True
                else:
                    trans_copy[index][5] = False
                self.replay_buffer.add(trans_copy[index])
        # reset temp goal buffer
        self.temp_goal_replay_storage = []

    # penalize subgoal
    def penalize_subgoal(self, subgoal, next_state, high_level_goal_achieved):    
        transition = [self.current_state, subgoal, self.subgoal_penalty, next_state, self.goal, True, None]
        self.replay_buffer.add(np.copy(transition))

    # Determine whether training of the layer is finished 
    def return_to_higher_level(self, max_lay_achieved, agent, env, attempts_made):
        if max_lay_achieved is not None and max_lay_achieved >= self.layer_number:
            return True

        # Return when out of time
        elif agent.steps_taken >= env.max_actions:
            return True

        # Return when layer has maxed out attempts
        elif not agent.FLAGS.test and attempts_made >= self.time_limit:
            return True

        # NOTE: During testing, agent will have env.max_action attempts to achieve goal
        elif agent.FLAGS.test and self.layer_number < agent.FLAGS.layers-1 and attempts_made >= self.time_limit:
            return True
        else:
            return False

    def train(self, agent, env, test=False, subgoal_test=False, episode_num=None):
        self.goal = agent.goal_array[self.layer_number]
        self.current_state = agent.current_state
        self.maxed_out = False
        attempts_made = 0

        while True:
            if self.FLAGS.normalize and self.replay_buffer.state_norm and self.replay_buffer.goal_norm:
                self.normalize_state = self.replay_buffer.state_norm.normalize(self.current_state)
                self.normalize_goal = self.replay_buffer.goal_norm.normalize(self.goal)

            #Select action
            action, action_type, next_subgoal_test = self.choose_action(agent, env,test, subgoal_test)
            if self.layer_number > 0:             
                agent.goal_array[self.layer_number - 1] = action        
                goal_status, max_lay_achieved = agent.layers[self.layer_number - 1].train(agent, env, test=test, subgoal_test=next_subgoal_test, episode_num=episode_num)

            # If layer is bottom level, execute action
            else:
                next_state = env.execute_action(action)        
                agent.steps_taken += 1
                agent.current_state = next_state
                goal_status, max_lay_achieved = agent.check_goals(env)
            attempts_made += 1                                      

            if self.layer_number == 0:                       
                hindsight_action = action
            else:
                # If subgoal action was achieved by layer below, use this as hindsight action
                if goal_status[self.layer_number-1]:
                    hindsight_action = action
                else:
                    hindsight_action = env.project_state_to_subgoal(agent.current_state)

            # for imitation learning we need to save data when both testing and training
            self.perform_action_replay(hindsight_action, agent.current_state, goal_status, env, rtype=self.FLAGS.rtype)
            
            # no need to make her data
            if not agent.FLAGS.test and not test:
                if agent.FLAGS.her:    
                    self.create_prelim_goal_replay_trans(hindsight_action, agent.current_state, env, agent.FLAGS.layers)      
                # Penalize subgoals 
                if self.layer_number > 0 and next_subgoal_test and agent.layers[self.layer_number-1].maxed_out:
                    self.penalize_subgoal(action, agent.current_state, goal_status[self.layer_number])

            if agent.FLAGS.verbose:
                print("\nEpisode %d, Training Layer %d, Attempt %d" % (episode_num, self.layer_number,attempts_made))
                print("Old State: ", self.current_state)
                print("Hindsight Action: ", hindsight_action)
                print("Original Action: ", action)
                print("Next State: ", agent.current_state)
                print("Goal: ", self.goal)
                if self.layer_number == agent.FLAGS.layers - 1:
                    print("Hindsight Goal: ", env.project_state_to_end_goal(agent.current_state))
                else:
                    print("Hindsight Goal: ", env.project_state_to_subgoal(agent.current_state))
                print("Goal Status: ", goal_status, "\n")
                print("All Goals: ", agent.goal_array)

            # Update state of current layer
            self.current_state = agent.current_state
            # Return to previous level to receive next subgoal if applicable
            if (max_lay_achieved is not None and max_lay_achieved >= self.layer_number) or agent.steps_taken >= env.max_actions or attempts_made >= self.time_limit:
                if attempts_made >= self.time_limit and not goal_status[self.layer_number]:
                    self.maxed_out = True

                # revise data for HER
                if not test and agent.FLAGS.her:
                    if self.layer_number == agent.FLAGS.layers - 1:
                        goal_thresholds = env.end_goal_thresholds
                    else:
                        goal_thresholds = env.subgoal_thresholds
                    self.finalize_goal_replay(goal_thresholds)

                # Under certain circumstances, the highest layer will not seek a new end goal
                if self.return_to_higher_level(max_lay_achieved, agent, env, attempts_made):
                    return goal_status, max_lay_achieved


    # Update neural networks
    def learn(self, num_updates, episode_num):
        for _ in range(num_updates):
            if self.replay_buffer.size >= self.batch_size:
                old_states, actions, rewards, new_states, goals, is_terminals = self.replay_buffer.get_batch()
                # forward dynamic
                if self.FLAGS.curiosity:
                    intrinsic_reward = self.cur_model.get_intrinsic_reward(old_states, actions, new_states)
                    self.normalize_inreward.update(np.vstack(intrinsic_reward))
                    normed_intrinsic_reward = self.normalize_inreward.normalize(intrinsic_reward)   
                    self.cur_model.train(old_states, actions, new_states)
                    rewards = normed_intrinsic_reward

                self.critic.update(old_states, actions, rewards, new_states, goals, self.actor.get_action(new_states,goals), is_terminals)
                action_derivs = self.critic.get_gradients(old_states, goals, self.actor.get_action(old_states, goals))
                self.actor.update(old_states, goals, action_derivs)

            # Behavior Clone
            if self.FLAGS.imitation:
                self.actor.imit_ratio = self.actor.imit_init_ratio * (1 - episode_num / self.FLAGS.episodes)
                imit_states, imit_actions, _, _, imit_goals, _ = self.imitation_buffer.get_batch()
                imit_loss, _ = self.actor.imit_update(imit_states, imit_goals, imit_actions)

        if self.FLAGS.imitation:
            print('imitation loss :{}'.format(imit_loss))



                
        
  