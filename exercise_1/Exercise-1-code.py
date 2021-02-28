from collections import defaultdict

import numpy as np



class Variable:
    def __init__(self, name, no_states, table, parents=[], no_parent_states=[]):
        """
        name (string): Name of the variable
        no_states (int): Number of states this variable can take
        table (list or Array of reals): Conditional probability table (see below)
        parents (list of strings): Name for each parent variable.
        no_parent_states (list of ints): Number of states that each parent variable can take.

        The table is a 2d array of size #events * #number_of_conditions.
        #number_of_conditions is the number of possible conditions (prod(no_parent_states))
        If the distribution is unconditional #number_of_conditions is 1.
        Each column represents a conditional distribution and sum to 1.

        Here is an example of a variable with 3 states and two parents cond0 and cond1,
        with 3 and 2 possible states respectively.
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond0   | cond0(0) | cond0(1) | cond0(2) | cond0(0) | cond0(1) | cond0(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        |  cond1   | cond1(0) | cond1(0) | cond1(0) | cond1(1) | cond1(1) | cond1(1) |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(0) |  0.2000  |  0.2000  |  0.7000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(1) |  0.3000  |  0.8000  |  0.2000  |  0.0000  |  0.2000  |  0.4000  |
        +----------+----------+----------+----------+----------+----------+----------+
        | event(2) |  0.5000  |  0.0000  |  0.1000  |  1.0000  |  0.6000  |  0.2000  |
        +----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

        Variable('event', 3, [[0.2, 0.2, 0.7, 0.0, 0.2, 0.4],
                              [0.3, 0.8, 0.2, 0.0, 0.2, 0.4],
                              [0.5, 0.0, 0.1, 1.0, 0.6, 0.2]],
                 parents=['cond0', 'cond1'],
                 no_parent_states=[3, 2])
        """
       
        self.name = name #name of variable 
        self.no_states = no_states #number of states the variable can take 
        self.table = np.array(table) #CPT
        self.parents = parents #Name for each parent variable
        self.no_parent_states = no_parent_states #number of states each parent variable can take 

        if self.table.shape[0] != self.no_states:
            raise ValueError(f"Number of states and number of rows in table must be equal. "
                             f"Recieved {self.no_states} number of states, but table has "
                             f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError("Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError("Number of parents must match number of length of list no_parent_states.")

    def __str__(self):
        """
        Pretty string for the table distribution
        For printing to display properly, don't use variable names with more than 7 characters
        """
        width = int(np.prod(self.no_parent_states))
        grid = np.meshgrid(*[range(i) for i in self.no_parent_states])
        s = ""
        for (i, e) in enumerate(self.parents):
            s += '+----------+' + '----------+' * width + '\n'
            gi = grid[i].reshape(-1)
            s += f'|{e:^10}|' + '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + '|'.join([f'{p:^10.4f}' for p in self.table[i]])
            s += '|\n'

        s += '+----------+' + '----------+' * width + '\n'

        return s

    def probability(self, state, parentstates):
        """
        Returns probability of variable taking on a "state" given "parentstates"
        This method is a simple lookup in the conditional probability table, it does not calculate anything.

        Input:
            state: integer between 0 and no_states
            parentstates: dictionary of {'parent': state}
        Output:
            float with value between 0 and 1
        """
        if not isinstance(state, int):
            raise TypeError(f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable in self.parents:
            if variable not in parentstates:
                raise ValueError(f"Variable {variable} does not have a defined value in parentstates.")

            var_index = self.parents.index(variable)
            table_index += parentstates[variable] * np.prod(self.no_parent_states[:var_index])

        return self.table[state, int(table_index)]


class BayesianNetwork:
    """
    Class representing a Bayesian network.
    Nodes can be accessed through self.variables['variable_name'].
    Each node is a Variable.

    Edges are stored in a dictionary. A node's children can be accessed by
    self.edges[variable]. Both the key and value in this dictionary is a Variable.
    """
    def __init__(self):
        self.edges = defaultdict(lambda: [])  # All nodes start out with 0 edges
        self.variables = {}                   # Dictionary of "name":TabularDistribution

    def add_variable(self, variable):
        """
        Adds a variable to the network.
        """
        if not isinstance(variable, Variable):
            raise TypeError(f"Expected {Variable}; got {type(variable)}.")
        self.variables[variable.name] = variable

    def add_edge(self, from_variable, to_variable):
        """
        Adds an edge from one variable to another in the network. Both variables must have
        been added to the network before calling this method.
        """
        if from_variable not in self.variables.values():
            raise ValueError("Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError("Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def sorted_nodes(self):
        """
        Kahns algorithm, with an extra "dummy"-list, temp. 
        I first iterate through all of the variables given, and for each of them 
        I add their children to a list called temp. The list allows for duplicates. 
        I then iterate through all the variables again, and if they are not in 
        temp I add them to S. Then I consider the first element of S, and add it to L 
        while removing it from S. I consider every child of that variable, and removes
        it from my dummy list. Since list allows for duplicates, I can see if the 
        children are children of other nodes as well by looking through temp. 
        If they are not, I add them to S. 
        Finally, I return L. 
        Returns: dictionary of sorted variable names.
        
        """
        S = []
        L = {}
        temp = []
        
        for key in self.variables:
            for child in self.edges[key]:
                temp.append(child) 
                
        for key in self.variables:
            if(key not in temp): 
                S.append(key)
        while(len(S)):
            temp_var = S[0]
            new_dictionary_element = {temp_var: self.edges[temp_var]}
            L.update(new_dictionary_element)
            S.remove(temp_var)
            for child in self.edges[temp_var]:
                temp.remove(child) 
                if(child not in temp):
                    S.append(child)
        if  temp: 
            raise Exception ("Graph has at least one cycle")           
        else: 
            return (L)
    


class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()
        

    def _enumeration_ask(self, X, evidence):
        """
        self: instance of inferenceByEnumeration-object
        (which contains a bayesion network and a dictionary of the bayesian network in topological sorted order. 
        X: Its type is string, but it is the name of a variable. We want to find its marginal distribution. 
        Evidence: a dictionary that is supposed to have all parent states of X (the node being enumerated). Key: name of parent, value: state
        
        I make an empty list, and find out which variable X is. Then I add the new evidence and 
        adds enumerate all here. 
        I normalize the elements of Q and return the list of the likelihood. 
     
        Returns: Normalized list Q
        
        """
       
        
        Q = []
        var_X = self.bayesian_network.variables[X]                
        for x_i in range(0, var_X.no_states): 
            new_to_evidence ={X: x_i}
            x_evidence = evidence.copy()
            x_evidence.update(new_to_evidence)
            Q.append(self._enumerate_all(self.topo_order.copy(), x_evidence))
            
        normalization_int = sum(Q)
     
        for element in Q: 
            index = Q.index(element)
            if(normalization_int != 0): 
                new_element = element/normalization_int
                Q[index] = new_element
            
        return(np.array(Q))
    
        
     
    
    def _enumerate_all(self, vars, evidence): 
        
        """
        Vars: A dictionary of variables names and their children, topologically sorted.
        evidence: A dictionary of parents and their states
        
        I find the evidence for the first element in vars, this is why it needs to be topologically sorted
        so I can go through it in its entirety in the right direction. 
        
        Then it actually finds the probability. Finds the probability of all states if Y is not in evidence
        and if it is, it finds only the probability for the state. 
        
        # Reminder:
        # When mutable types (lists, dictionaries, etc.) are passed to functions in python
        # it is actually passing a pointer to that variable. This means that if you want
        # to make sure that a function doesn't change the variable, you should pass a copy.
        # You can make a copy of a variable by calling variable.copy()
        
        """
        if(not vars): 
                return 1.0
        Y_evidence = evidence.copy()
        Y  = next(iter(vars)) 
        var_Y = self.bayesian_network.variables[Y]
        rest_vars = vars.copy()
        del rest_vars[Y]
        for key in evidence: 
            if (var_Y.name == key):
                #y_prob = evidence.get(key) I dont think this is correct, as it only gives me the state. 
                y_prob = var_Y.probability(Y_evidence.get(key), Y_evidence)
                return( y_prob* self._enumerate_all(rest_vars, Y_evidence))
        
        prob_y = []
        for increment in range(0, var_Y.no_states):
            Y_evidence[Y] = increment
            prob_y.append((var_Y.probability(increment, Y_evidence)*self._enumerate_all(rest_vars, Y_evidence)))
        return(sum(prob_y))
            
            
           
        for increment in range(0,var_Y.no_states):
            Y_evidence[Y] = increment
            y_prob = var_Y.probability(increment, Y_evidence)
            return (y_prob * self._enumerate_all(rest_vars,Y_evidence))
            
        
        

        

        
    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = self._enumeration_ask(var, evidence).reshape(-1, 1) #had to turn Q into a numpy array 
        return Variable(var, self.bayesian_network.variables[var].no_states, q) 
        #name of variable,  number of states of the variable, numpy vector with probability

    


def problem3c():
    d1 = Variable('A', 2, [[0.8], [0.2]])
    d2 = Variable('B', 2, [[0.5, 0.2],
                           [0.5, 0.8]],
                  parents=['A'],
                  no_parent_states=[2])
    d3 = Variable('C', 2, [[0.1, 0.3],
                           [0.9, 0.7]],
                  parents=['B'],
                  no_parent_states=[2])
    d4 = Variable('D', 2, [[0.6, 0.8],
                           [0.4, 0.2]],
                  parents=['B'],
                  no_parent_states=[2])
    
    print("This is for me because I forget it: 0 = true, 1 = false!")

    print(f"Probability distribution, P({d1.name})")
    print(d1)

    print(f"Probability distribution, P({d2.name} | {d1.name})")
    print(d2)

    print(f"Probability distribution, P({d3.name} | {d2.name})")
    print(d3)

    print(f"Probability distribution, P({d4.name} | {d2.name})")
    print(d4)

    bn = BayesianNetwork()

    bn.add_variable(d1)
    bn.add_variable(d2)
    bn.add_variable(d3)
    bn.add_variable(d4)
    bn.add_edge(d1, d2)
    bn.add_edge(d2, d3)
    bn.add_edge(d2, d4)
    

    inference = InferenceByEnumeration(bn)
    posterior = inference.query('C', {'D': 1})
    posterior2 = inference.query('A', {'C': 1, 'D': 0})

    print(f"Probability distribution, P({d3.name} | !{d4.name})")
    print(posterior)
    print(f"Probability distribution, P({d1.name} | !{d3.name}, {d4.name})")
    print(posterior2)
    


def monty_hall():
    print("Prize = prize")
    print("Chosen by guest = CBG")
    print("Opened by host = CBG")
    print("Door 1 = 0, door 2 = 1, door 3 = 2")
    print("")
    
    prize = Variable('Prize', 3, [[0.33333], [0.33333], [0.33333]])
    #chosenbyguest = CBG
    cbg = Variable('CBG', 3, [[0.33333], [0.33333], [0.33333]])
    #chosenbyhost = CBH
    cbh = Variable('CBH', 3, [[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 1.0, 0.5], 
                                                  [0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5], 
                                                  [0.5, 1.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0]],
                            parents = ['Prize', 'CBG'], 
                            no_parent_states =[3, 3])
    print(f"Probability distribution, P({prize.name})")
    print(prize)

    print(f"Probability distribution, P({cbg.name})")
    print(cbg)

    print(f"Probability distribution, P({cbh.name} | {prize.name, cbg.name})")
    print(cbh)

    bn = BayesianNetwork()

    bn.add_variable(cbg)
    bn.add_variable(prize)
    bn.add_variable(cbh)
    bn.add_edge(cbg, cbh)
    bn.add_edge(prize, cbh)
    inference = InferenceByEnumeration(bn)
    posterior = inference.query('Prize', {'CBG': 0, 'CBH': 2})
    print(f"Probability distribution, P({prize.name} | {cbg.name} = 0, {cbh.name} = 2)")
    print(posterior)

    """
    Explenation: each door is a different state the variables can be in. So my CPT for chosenByHost basically looks like this prior: 

        ---------+----------+----------+----------+----------+----------+----------+----------+----------+----
        |  chosenByGuest      | door 1 | door 1| door 1 | door 2 | door 2 | door 2 | door 3 | door 3 | door 3|
        +----------+----------+----------+----------+----------+----------+----------+----------+----------+
        |  Prize             | door 1 | door 2| door 3 | door 1 | door 2 | door 3 | door 1 | door 2 | door 3|
        +----------+----------+----------+----------+----------+----------+----------+----------+----------+
        | Host chooses door 1|  0.0  |  0.0  |  0.0   |  0.0   |  0.5   |  1     |  0.0  |   1     |  0.5  |
        +----------+----------+----------+----------+----------+----------+----------+----------+----------+
        | Host chooses door 2|  0.5  |  0.0  |  1    |  0.0   |   0.0  |  0.0   |  1    |   0.0   |  0.5   |
        +----------+----------+----------+----------+----------+----------+----------+----------+----------+
        | Host chooses door 3|  0.5  |  1    |  0.0  |  1     |  0.5  |  0.0    |  0.0   |  0.0  |  0.0    |
        +----------+----------+----------+----------+----------+----------+----------+----------+----------+

        To create this table you would use the following parameters:

    """


if __name__ == '__main__':
    problem3c()
    monty_hall()
    
    
    

