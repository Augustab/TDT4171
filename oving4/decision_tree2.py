import copy
import pandas as pd
import numpy as np
import graphviz as gv
import os
# this line was needed for graphwiz to work
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'


class Tree():
    """The Tree-class used to store the decision-tree-structure"""
    def __init__(self, attribute):
        self.children = {}
        self.attribute = attribute
        self.split_value = None

    def add_child(self, edge, child):
        self.children[edge] = child

    def add_split_value(self, split_value):
        self.split_value = split_value


def plurality_value(examples):
    """Method to calculate the plurality value used in the recursive method"""
    count = [0, 0]
    for value in examples["Survived"]:
        count[value] += 1
    return count.index(max(count))


def all_the_same(examples):
    """Checks if all examples have the same classification"""
    same = True
    first_val = examples["Survived"].iloc[0]
    for value in examples["Survived"]:
        if value != first_val:
            same = False
    return same


def B(p):
    """The B-function from the book, notice that i have added support in the edge-cases!"""
    if p < 0:
        raise ValueError("Something wrong has happened")
    elif p == 1:
        return 0
    elif p == 0:
        return 1
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def remainder(A, examples):
    """The function used to calculate remainder for categorical attributes"""
    remainder = 0
    # her henter jeg ut p=overlevde og n=døde fra eksemplene
    p = examples["Survived"].value_counts()[1]
    n = examples["Survived"].value_counts()[0]
    for value in examples[A].unique():
        # her henter jeg ut p=overlevde og n=døde fra eksemplene for gjeldende verdi value av attributten
        p_k = examples.groupby("Survived")[A].value_counts().unstack(
            fill_value=0).stack()[1, value]
        n_k = examples.groupby("Survived")[A].value_counts().unstack(
            fill_value=0).stack()[0, value]
        b_value = B(p_k / (p_k + n_k))
        remainder += ((p_k + n_k) / (p + n)) * b_value
    return remainder


def remainder_each(examples):
    """Function used to find remainder for each of the splitted example-sets"""
    # this is just another way to find p and n
    p_k = examples[examples["Survived"] == 1].shape[0]
    n_k = examples[examples["Survived"] == 0].shape[0]
    b_value = B(p_k / (p_k + n_k))
    return (p_k + n_k) * b_value


def remainder_split(examples, split, attribute):
    """The function that splits the examples and calculates the importance"""
    # Here i split the examples on the splitting-point
    head = examples.loc[examples[attribute] < split]
    tail = examples.loc[examples[attribute] > split]
    p = examples[examples["Survived"] == 1].shape[0]
    n = examples[examples["Survived"] == 0].shape[0]
    # i have to multiply in (1 / (p + n)) here since i dont do it in remainder_each()
    head_remainder = (1 / (p + n)) * remainder_each(head)
    tail_remainder = (1 / (p + n)) * remainder_each(tail)
    return 1 - head_remainder - tail_remainder


def find_splitting_point(attribute, examples):
    """The function that iterates through all changes in survived-values to find the best splitting point."""
    lastValue = None
    remainder = None
    new_median = None
    sorted = examples.sort_values(by=[attribute])
    for row in sorted.iterrows():
        if lastValue is None:
            lastValue = row[1][attribute]
            continue
        elif lastValue == row[1][attribute]:
            #we skip situations where current and last attribute-value is equal (only consider changes)
            continue
        else:
            median = round((lastValue + row[1][attribute]) / 2, 3)
            lastValue = row[1][attribute]
            new_remainder = remainder_split(sorted, median, attribute)
            if remainder is None:
                # first time we come here
                remainder = new_remainder
                new_median = median
            elif new_remainder > remainder:
                remainder = new_remainder
                new_median = median
    if new_median is None:
        # this is just to make sure that we dont return None as new_median.
        new_median = lastValue
    return remainder, new_median


def importance(attributes, categorical, examples):
    """The function that calculates the importance of all the possible attributes, used to choose next attribute"""
    best_attribute_value = None
    best_attribute = None
    splittingpoint = None
    split = None
    for attribute in attributes:
        if attribute in categorical:
            #Attribute is categorical (not continous)
            entropy = 1 - remainder(attribute, examples)
        else:
            # Attribute is continous, need to find splitting-point first
            entropy, splittingpoint = find_splitting_point(attribute, examples)
        if best_attribute_value is None:
            # the first time we come here
            best_attribute_value = entropy
            best_attribute = attribute
            if not (attribute in categorical):
                split = splittingpoint
            else:
                split = None
        elif best_attribute_value < entropy:
            #if the new attribute was better we come here
            best_attribute_value = entropy
            best_attribute = attribute
            if not (attribute in categorical):
                # if it is continuous we have to update the split-value
                split = splittingpoint
            else:
                split = None
    return best_attribute, split


def decision_tree_learning(
        examples,
        attributes,
        categorical,
        unique,
        parent_examples):
    """The DTL-algorithm as explained in the textbook. Supports both categorical and continous attributes!"""
    if examples.empty:
        return plurality_value(parent_examples)
    elif all_the_same(examples):
        return examples["Survived"].iloc[0]
    elif attributes.empty:
        return plurality_value(examples)
    else:
        A, split = importance(attributes, categorical, examples)
        root = Tree(A)
        new_attributes = copy.deepcopy(attributes)
        if split is None:
            # this means we have chosen a categorical attribute
            new_attributes = new_attributes.drop(A)
            for value in unique[A]:
                exs = examples.loc[examples[A] == value]
                # i call the function recursicely on the new shortened examples.
                subtree = decision_tree_learning(
                    exs, new_attributes, categorical, unique, examples)
                root.add_child(value, subtree)
        else:
            # this means we have chosen a continuous
            examples = examples.sort_values(by=[A])
            head = examples.loc[examples[A] < split]
            tail = examples.loc[examples[A] > split]
            new_attributes = new_attributes.drop(A)
            # her kaller jeg funksjonen rekursivt på hver av split-eksemplene.
            subtree = decision_tree_learning(
                head, new_attributes, categorical, unique, examples)
            subtree2 = decision_tree_learning(
                tail, new_attributes, categorical, unique, examples)
            #denne split-verdien får jeg bruk for ved senere andledninger.
            root.split_value = split
            root.add_child("Less than", subtree)
            root.add_child("Greater than", subtree2)
        return root


def illustrate_tree(tree, lasttree):
    """The function used to print the DT with Graphviz"""
    # The first if is necessary to stop the recursive algorithm when we reach bottom.
    if isinstance(tree, Tree):
        for edge, value in tree.children.items():
            # Dette er en litt jalla løsning som tar høyde for at første kall ikke har noe last tree
            if lasttree is None:
                # Inn her kommer man i første iterasjon.
                old_name = value.attribute + str(edge)
                G.node(name=old_name, label=str(value.attribute))
                G.edge(tree.attribute, old_name, label=str(edge))
                illustrate_tree(value, old_name)
            elif isinstance(value, Tree):
                # kommer koden inn her er vi i et tre (node) midt i det store treet.
                old_name = lasttree + value.attribute + str(edge)
                if tree.split_value is not None:
                    label = str(edge) + " " + str(tree.split_value)
                    G.node(name=old_name, label=str(value.attribute))
                    G.edge(lasttree, old_name, label=label)
                else:
                    G.node(name=old_name, label=str(value.attribute))
                    G.edge(lasttree, old_name, label=str(edge))
                illustrate_tree(value, old_name)
            else:
                # kommer koden inn her er man i bunn av treet.
                if tree.split_value is not None:
                    G.node(lasttree + str(edge) + str(value), label=str(value))
                    label = str(edge) + " " + str(tree.split_value)
                    G.edge(
                        lasttree,
                        lasttree +
                        str(edge) +
                        str(value),
                        label=label)
                else:
                    G.node(lasttree + str(edge) + str(value), label=str(value))
                    G.edge(
                        lasttree,
                        lasttree +
                        str(edge) +
                        str(value),
                        label=str(edge))


def test_tree(testset):
    """The funtion used to test the tree on a testset"""
    result = [0, 0]
    for index, row in testset.iterrows():
        # tar en deepcopy av treet i tilfelle jeg skulle ende opp med å redigere det
        tre = copy.deepcopy(tree)
        # while-løkke som itererer hele veien ned til bunn av treet
        while isinstance(tre, Tree):
            if tre.split_value is not None:
                if row[tre.attribute] > tre.split_value:
                    tre = tre.children["Greater than"]
                else:
                    tre = tre.children["Less than"]
            else:
                tre = tre.children[row[tre.attribute]]
        value = tre
        # når jeg kommer hit har jeg fått verdien treet mitt foreslår gitt eksempelet. Tester om det stemmer.
        if value == row["Survived"]:
            result[1] += 1
        else:
            result[0] += 1
    return result


df = pd.read_csv("titanic/train.csv")
#TASK A:::::::::: usable_df = df[["Survived", "Pclass", "Sex", "Embarked"]]
#TASK B::::::::::
usable_df = df[["Survived", "Pclass", "Sex", "Fare", "Parch", "SibSp", "Embarked"]]
categorical = df[["Pclass", "Sex", "Embarked"]].columns
unique = {}
for cat in categorical:
    unique[cat] = usable_df[cat].unique()
#TASK A::::::::::: attributes = df[["Pclass", "Sex", "Embarked"]].columns
#TASK B:::::::::::
attributes = df[["Pclass", "Sex", "Fare", "Parch", "SibSp", "Embarked"]].columns
tree = decision_tree = decision_tree_learning(
    usable_df, attributes, categorical, unique, usable_df)

######### ILLUSTRATING THE TREE #############
G = gv.Digraph("G")
G.node(tree.attribute)
illustrate_tree(tree, None)
G.view()

######### TESTING THE TREE ###########
df_test = pd.read_csv("titanic/test.csv")
result = test_tree(df_test)
print("RESULT", result, "Gives an accuracy of",
      round((result[1] / sum(result)) * 100, 1), "%")
