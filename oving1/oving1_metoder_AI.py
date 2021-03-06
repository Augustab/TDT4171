from collections import defaultdict

import numpy as np
import copy


class Variable:
    def __init__(
            self,
            name,
            no_states,
            table,
            parents=[],
            no_parent_states=[]):
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
        self.name = name
        self.no_states = no_states
        self.table = np.array(table)
        self.parents = parents
        self.no_parent_states = no_parent_states

        if self.table.shape[0] != self.no_states:
            raise ValueError(
                f"Number of states and number of rows in table must be equal. "
                f"Recieved {self.no_states} number of states, but table has "
                f"{self.table.shape[0]} number of rows.")

        if self.table.shape[1] != np.prod(no_parent_states):
            raise ValueError(
                "Number of table columns does not match number of parent states combinations.")

        if not np.allclose(self.table.sum(axis=0), 1):
            raise ValueError("All columns in table must sum to 1.")

        if len(parents) != len(no_parent_states):
            raise ValueError(
                "Number of parents must match number of length of list no_parent_states.")

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
            s += f'|{e:^10}|' + \
                '|'.join([f'{e + "("+str(j)+")":^10}' for j in gi])
            s += '|\n'

        for i in range(self.no_states):
            s += '+----------+' + '----------+' * width + '\n'
            state_name = self.name + f'({i})'
            s += f'|{state_name:^10}|' + \
                '|'.join([f'{p:^10.4f}' for p in self.table[i]])
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
            raise TypeError(
                f"Expected state to be of type int; got type {type(state)}.")
        if not isinstance(parentstates, dict):
            raise TypeError(
                f"Expected parentstates to be of type dict; got type {type(parentstates)}.")
        if state >= self.no_states:
            raise ValueError(
                f"Recieved state={state}; this variable's last state is {self.no_states - 1}.")
        if state < 0:
            raise ValueError(
                f"Recieved state={state}; state cannot be negative.")

        table_index = 0
        for variable_name in self.parents:
            #print("Variable", variable_name, " + self", self.name, "self.parretns", self.parents)
            if variable_name not in parentstates:
                raise ValueError(
                    f"Variable {variable_name} does not have a defined value in parentstates.")
            var_index = self.parents.index(variable_name)
            table_index += parentstates[variable_name] * \
                np.prod(self.no_parent_states[:var_index])

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
        # All nodes start out with 0 edges
        self.edges = defaultdict(lambda: [])
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
            raise ValueError(
                "Parent variable is not added to list of variables.")
        if to_variable not in self.variables.values():
            raise ValueError(
                "Child variable is not added to list of variables.")
        self.edges[from_variable].append(to_variable)

    def parents(self, node):
        parents = []
        for variable_parent_name, variable_parent in self.variables.items():
            if node in self.edges[variable_parent]:
                parents.append(variable_parent)
        return parents

    def sorted_nodes(self):
        """
        TODO: Implement Kahn's algorithm (or some equivalent algorithm) for putting
              variables in lexicographical topological order.
        Returns: List of sorted variable names.

        Her er det tatt utgangspunkt i algoritmen til Kahn: https://en.wikipedia.org/wiki/Topological_sorting
        """
        sorted = []
        startnodes = []
        bayes_copy = copy.deepcopy(self)
        # MERK: Jeg tar en deepcopy av hele det bayesianske nettverket her for ?? ikke slette kantene i det originale
        # nettverket som jeg senere f??r brukt for!!

        for node in bayes_copy.variables.values():
            if not bayes_copy.parents(node):
                # if the code comes here we have confirmed that the node has no
                # parents (since the parents-list is empty)
                startnodes.append(node)

        while startnodes:
            current_node = startnodes.pop(0)
            sorted.append(current_node.name) #jeg appender navnet og ikke noden fordi det er det beskrivelsen sier.
            edges = bayes_copy.edges[current_node].copy()
            for node in edges:
                bayes_copy.edges[current_node].remove(node)
                if not bayes_copy.parents(node):
                    startnodes.append(node)

        # koden under s??rger bare for at det kastes en feilmelding dersom ikke
        # sorteringen var vellykket.
        empty = True
        for variable_name in self.variables.keys():
            if bayes_copy.edges[variable_name]:
                empty = False
        if not empty:
            raise Exception(
                'The sorting was not successful -- not all edges removed')
        else:
            return sorted


class InferenceByEnumeration:
    def __init__(self, bayesian_network):
        self.bayesian_network = bayesian_network
        self.topo_order = bayesian_network.sorted_nodes()

    def _enumeration_ask(self, X, evidence):
        var_x = self.bayesian_network.variables[X]
        Q = np.zeros((var_x.no_states, 1)) ##viktig ?? bruke np.array siden det er dette utgitt kode antar.
        vars_topo = []
        for name in self.topo_order:
            vars_topo.append(self.bayesian_network.variables[name])
            # her legger jeg nodene i nettverket inn i en liste i riktig rekkef??lge!
        for state_x in range(var_x.no_states):
            evidence_copy = copy.deepcopy(evidence)
            evidence_copy[var_x.name] = state_x
            prob = self._enumerate_all(vars_topo, evidence_copy)
            Q[state_x] = prob
        return Q * 1 / sum(Q) #her ganger jeg med normaliseringskonstanten!!

    def _enumerate_all(self, vars_topo, evidence):
        if not vars_topo:
            return 1
        # her kan jeg ikke bruke deep-copy da dette vil lage helt nye node-objekter
        vars_topo_copy = vars_topo.copy()
        # her er ok ?? bruke deepcopy da det ikke er objekter men strings og tall
        evidence_copy = copy.deepcopy(evidence)
        y = vars_topo_copy.pop(0)
        evidence_shortened = {}
        # for-l??kken under brukes for ?? lage et sett med evidence som kun inneholder foreldrene til y
        for variable_name in evidence_copy.keys():
            if variable_name in y.parents:
                evidence_shortened[variable_name] = evidence_copy[variable_name]
        if y.name in evidence.keys():
            return y.probability(evidence[y.name], evidence_shortened) * self._enumerate_all(
                vars_topo_copy.copy(), copy.deepcopy(evidence))
        else:
            sum_over_y = 0
            for state_y in range(y.no_states):
                evidence_copy[y.name] = state_y
                sum_over_y += y.probability(state_y, evidence_shortened) * self._enumerate_all(
                    vars_topo_copy.copy(), copy.deepcopy(evidence_copy))
            return sum_over_y

    def query(self, var, evidence={}):
        """
        Wrapper around "_enumeration_ask" that returns a
        Tabular variable instead of a vector
        """
        q = self._enumeration_ask(var, evidence).reshape(-1, 1)
        return Variable(var, self.bayesian_network.variables[var].no_states, q)


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

    print(bn.sorted_nodes())
    inference = InferenceByEnumeration(bn)
    posterior = inference.query('C', {'D':1})

    #print(f"Probability distribution, P({d3.name} | !{d4.name})")
    print(posterior)


def monty_hall():
    # TODO: Implement the monty hall problem as described in Problem 4c)
    Prize = Variable('Prize', 3, [[1/3], [1/3], [1/3]])
    ChosenByGuest = Variable('Guest', 3, [[1/3], [1/3], [1/3]],
                             parents=[],
                             no_parent_states=[])
    OpenedByHost = Variable(
        'Host', 3, [[0, 0, 0, 0, 0.5, 1, 0, 1, 0.5],
                    [0.5, 0, 1, 0, 0, 0, 1, 0, 0.5],
                    [0.5, 1, 0, 1, 0.5, 0, 0, 0, 0]], parents=['Prize', 'Guest'], no_parent_states=[3,3])

    bn_mothy = BayesianNetwork()

    bn_mothy.add_variable(Prize)
    bn_mothy.add_variable(ChosenByGuest)
    bn_mothy.add_variable(OpenedByHost)
    bn_mothy.add_edge(ChosenByGuest, OpenedByHost)
    bn_mothy.add_edge(Prize, OpenedByHost)

    print(f"Probability distribution, P({ChosenByGuest.name})")
    print(ChosenByGuest)

    print(f"Probability distribution, P({Prize.name})")
    print(Prize)

    print(f"Probability distribution, P({OpenedByHost.name} | {ChosenByGuest.name}, {Prize.name})")
    print(OpenedByHost)

    inference = InferenceByEnumeration(bn_mothy)
    posterior = inference.query('Prize', {'Guest': 0, 'Host': 2})
    print(posterior)


if __name__ == '__main__':
    problem3c()
    monty_hall()
    ##by the result vi get the answer that it is to our advantage to switch choice!
