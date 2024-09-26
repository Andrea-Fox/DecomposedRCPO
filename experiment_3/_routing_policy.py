# here we define the class that corresponds to the routing server

# each function will have a function called select_action that chooses the correct action
# in the constructor of each action object, it will be specified the information considered to choose the action
import numpy as np
import math

class StatelessRouting():


    def __init__(self, env, routing_policy = None):
        self.env = env

        # the routing policy is a matrix that contains the probability of routing to each server for videos from each origin area
        if routing_policy is None:
            self.routing_policy = np.zeros((env.origin_servers, env.origin_server))
        else:
            self.routing_policy = routing_policy
        
    def select_section(self, env, state = None):
        if state == None:
            state = env.current_state

        # we select the section according to the routing policy
        origin_area = state['last_origin_server']
        return np.random.choice(env.N_servers, p = self.routing_policy[origin_area, :])

    def print_probabilities(self, env, state = None):
        if state == None:
            state = env.current_state
        origin_area = state['last_origin_server']
        print(self.routing_policy[origin_area, :])



class ComponentOccupationRouting():

    def __init__(self, env, routing_policy = None):
        self.env = env
        if routing_policy is None:
            max_capacity = np.max([env.servers[i].memory_capacity for i in range(env.N_servers)])
            self.routing_policy = np.zeros(( max_capacity, max_capacity, max_capacity, env.origin_servers, env.origin_servers))
        else:
            self.routing_policy = routing_policy

    def select_section(self, env, state = None):
        if state is None:
            state = env.current_state

        # we select the section according to the routing policy
        origin_area = state['last_origin_server']
        occupation_vector = np.zeros((env.N_servers, ))
        for i in range(env.N_servers):
            occupation_vector[i] = state['server_'+str(i+1)+'_occupation'][origin_area]
        
        # print(server_1_occupation, server_2_occupation, server_3_occupation, origin_area)
        # print(self.routing_policy[server_1_occupation, server_2_occupation, server_3_occupation, origin_area, :])
        # print(server_1_occupation, server_2_occupation, server_3_occupation)
        # print(self.routing_policy[server_1_occupation, server_2_occupation, server_3_occupation, :])
        
        # we only allow it to send to servers that have interest in this video
        area_interested = []    
        for i in range(env.N_servers):
            if env.servers[i].areas_of_interest[origin_area]:
                area_interested.append(i)
        # print(origin_area, area_interested)
        

        weight_multiplier = 1
        # we need to compute the probabiities of routing and then sample from this distribution
        probabilities_area_interested = np.zeros((env.N_servers, ))

        for i in range(env.N_servers):
            # clearly we only consider the servers interested in videos form this area
            if i in area_interested:
                probabilities_area_interested[i]  = math.exp(- weight_multiplier * occupation_vector[i]/env.servers[i].access_capacity)
        
        probabilities_area_interested = probabilities_area_interested/np.sum(probabilities_area_interested)
        
        return np.random.choice(area_interested, p = probabilities_area_interested)
        
    def print_probabilities(self, env, state = None):
        if state == None:
            state = env.current_state
        origin_area = state['last_origin_server']
        server_1_occupation = state['server_1_occupation'][origin_area]
        server_2_occupation = state['server_2_occupation'][origin_area]
        server_3_occupation = state['server_3_occupation'][origin_area]
        print(self.routing_policy[server_1_occupation, server_2_occupation, server_3_occupation, origin_area, :])


class TotalOccupationRouting():
    
        def __init__(self, env, routing_policy = None):
            self.env = env
            if routing_policy is None:
                max_capacity = np.max([env.servers[i].memory_capacity for i in range(env.N_servers)]) + 1
                self.routing_policy = np.zeros(( max_capacity, max_capacity, max_capacity, env.N_servers, env.N_servers))
            else:
                self.routing_policy = routing_policy
              

        def select_section(self, env, state = None):
            if state is None:
                state = env.current_state
    
            # we select the section according to the routing policy
            

            # we select the section according to the routing policy
            origin_area = state['last_origin_server']
            occupation_vector = np.zeros((env.N_servers, ))
            for i in range(env.N_servers):
                occupation_vector[i] = np.sum(state['server_'+str(i+1)+'_occupation'])
        
       
            # we only allow it to send to servers that have interest in this video
            area_interested = []    
            for i in range(env.N_servers):
                if env.servers[i].areas_of_interest[origin_area]:
                    area_interested.append(i)
            # print(origin_area, area_interested)


            weight_multiplier = 1
            # we need to compute the probabiities of routing and then sample from this distribution
            probabilities_area_interested = np.zeros((env.N_servers, ))

            for i in range(env.N_servers):
                # clearly we only consider the servers interested in videos form this area
                if i in area_interested:
                    probabilities_area_interested[i]  = math.exp(- weight_multiplier * occupation_vector[i]/env.servers[i].access_capacity)

            probabilities_area_interested = probabilities_area_interested/np.sum(probabilities_area_interested)

            return np.random.choice(np.arange(env.N_servers), p = probabilities_area_interested)
            

class ComponentAndTotalOccupationRouting():

    def __init__(self, env, routing_policy = None):
        self.env = env
        if routing_policy is None:
            max_capacity = np.max([env.servers[i].memory_capacity for i in range(env.N_servers)])
            self.routing_policy = np.zeros(( max_capacity, max_capacity, max_capacity, max_capacity, max_capacity, max_capacity, env.origin_servers, env.origin_servers))
        else:
            self.routing_policy = routing_policy

    def select_section(self, env, state = None):
        if state is None:
            state = env.current_state

        # we select the section according to the routing policy
        origin_area = state['last_origin_server']
        server_1_occupation_component = state['server_1_occupation'][origin_area]
        server_1_total_occupation = np.sum(state['server_1_occupation'])
        server_2_occupation_component = state['server_2_occupation'][origin_area]
        server_2_total_occupation = np.sum(state['server_2_occupation'])
        server_3_occupation_component = state['server_3_occupation'][origin_area]
        server_3_total_occupation = np.sum(state['server_3_occupation'])
        
        
        origin_area = state['last_origin_server']
        # we only allow it to send to servers that have interest in this video
        area_interested = []    
        for i in range(env.N_servers):
            if env.servers[i].areas_of_interest[origin_area]:
                area_interested.append(i)
        
        # we need to normalize the probabilities
        if len(area_interested) < env.N_servers:
            probabilities_area_interested = np.zeros((len(area_interested), ))
            for i in range(len(area_interested)):
                probabilities_area_interested[i] = self.routing_policy[server_1_occupation_component, server_1_total_occupation, server_2_occupation_component, server_2_total_occupation, server_3_occupation_component, server_3_total_occupation, origin_area, area_interested[i]]
                probabilities_area_interested = probabilities_area_interested/np.sum(probabilities_area_interested)
                if np.max(probabilities_area_interested)==0.5:
                    print('equal probabilities')
            return np.random.choice(area_interested, p = probabilities_area_interested)
        else:
            return np.random.choice(range(env.N_servers), p = self.routing_policy[server_1_occupation_component, server_1_total_occupation, server_2_occupation_component, server_2_total_occupation, server_3_occupation_component, server_3_total_occupation, origin_area, :])

    def print_probabilities(self, env, state = None):
        raise ValueError('Not implemented yet')
        if state == None:
            state = env.current_state
        origin_area = state['last_origin_server']
        server_1_occupation = state['server_1_occupation'][origin_area]
        server_2_occupation = state['server_2_occupation'][origin_area]
        server_3_occupation = state['server_3_occupation'][origin_area]
        print(self.routing_policy[server_1_occupation, server_2_occupation, server_3_occupation, origin_area, :])
