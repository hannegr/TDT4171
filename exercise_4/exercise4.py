# -*- coding: utf-8 -*-
"""
Created on Sun Mar 5 14:30:04 2021

@author: Hanne-Grete Alvheim 
@assignment: 4 in Artificial Intelligence methods 

Implementation of a decision tree classifier 


"""

import pandas as pd
from math import log2



class Tree:
    def __init__(self, parents, children): 
        self.parents = parents
        self.children = children #children is a list  of attributes in parents maybe 
    
    def make_child_parent(self):
        child_key = next(iter(self.children))
        child_key_items = self.children.get(child_key)
        self.parents[child_key] = child_key_items
        del self.children[child_key]
    
    def add_child_attributes(self, attributes): #Not used anymore
        for A in attributes: 
            self.children[A] = None
    
    def add_plurality_value(self, examples): #not used anymore
        if(len(self.children)):
            child_key = list(self.children.keys())[0] 
        #child_key = next(iter(self.children))
            if(self.children.get(child_key) == None): 
                self.children[child_key] = [["Survived", plurality_value(examples)]]
            else: 
                self.children[child_key].append([["Survived", plurality_value(examples)]])    
        else:
            self.children["Survived"] = plurality_value(examples)
            
    def add_value_in_subtree(self,attribute, attribute_value, value):
        if(self.children.get(attribute_value) == None):
            self.children[attribute, attribute_value] = [value]
        else:
            self.children[attribute, attribute_value].append([value])
            
"""
@problem a) 
Implement the Decision-Tree-Learning alogrithm from Artificial Intelligence -
a modern approach with informational gain as the IMPORTANCE function. Use it 
to train a decision tree on the Titanic dataset. 

"""
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 12)

def convert_to_pandas(excel_file):
    
    """
    Parameters
    ----------
    excel_file : excel file 
    Returns
    -------
    pandas object read_titanic

    """
    read_titanic = pd.read_csv(excel_file)

    if(excel_file == 'train.csv'):
        split_info = read_titanic["Survived"].str.split(",", n = 11, expand = True)
        read_titanic["Survived"] = split_info[0]
        read_titanic["Pclass"] = split_info[1]
        read_titanic["Name"] = split_info[2] + "," + split_info[3]
        read_titanic["Sex"] = split_info[4]
        read_titanic["Age"] = split_info[5]
        read_titanic["SibSp"] = split_info[6]
        read_titanic["Parch"] = split_info[7]
        read_titanic["Ticket"] = split_info[8]
        read_titanic["Fare"] = split_info[9]
        read_titanic["Cabin"] = split_info[10]
        read_titanic["Embarked"] = split_info[11]
        
    return(read_titanic)

def get_examples(pandacsv, rows, columns): 
    """
    Parameters
    ----------
    pandacsv : datasheet in Pandas-format
    rows : number of rows we do not want to use 
    columns : name of columns we do not want to use 

    Returns
    -------
    datasheet in Pandas-format without unneccesary stuff 
    
    """
    panda_ex = pandacsv.copy()
    if len(columns): 
        panda_ex.drop(columns, axis = 1, inplace=True)
    if len(rows): 
        dropped = panda_ex.index[rows]
        panda_ex.drop(dropped, inplace = True)
    panda_ex.reset_index(inplace = True, drop = True)
    return panda_ex


def same_classification(examples): 
    """
    Parameters
    ----------
    examples : list with examples in Pandas-format

    Returns
    -------
    the classification if all classifications are the same, 0 if they are not  
    """
    pn_dict = total_p_n(examples)
    if(pn_dict.get("p") == 0 or pn_dict.get("n") == 0): 
        return True
    return False


def B(q): 
    if(q == 0 or q == 1): 
        return 0 
    #    return -(1-q)*log2(1-q)
    #elif(q == 1): 
    #    return-q*log2(q)
    else:
        return(-(q*log2(q) +(1-q)*log2(1-q)))

def find_attribute_states_and_amount(A, examples):
    """
    Parameters
    ----------
    A : Attribute (string)
    examples : pandas data format
    
    Returns
    -------
    vk_dict : dictionary containing all of the states of the attribute  A and the amounts of each of these states in the key-value pairs. 
    
    
    """
    vk_dict = {} 
    vk_list = examples[A].tolist()
    vk_elements_list = list(set(vk_list)) #to get only the different values out
    iter = 0
    for elem in vk_elements_list: 
        vk_dict.update({elem: vk_list.count(vk_elements_list[iter])}) #counts existence of different items in the vk_elements_list. 
        iter+=1     
    return vk_dict

def total_p_n(pandacsv):  
    """
    Parameters
    ----------
    pandacsv : pandas data format
    Returns
    -------
    survived : dictionary with the total amount that survived and the total amount that did not survive. 
    """
    
    survived = find_attribute_states_and_amount("Survived", pandacsv)
    survdict = {"p": survived.get('1'), "n": survived.get('0')}
    for a in survdict: 
        if (survdict.get(a) == None): 
            survdict[a] = 0
    return survdict
    
def partial_p_n(A, state, pandacsv):
    """

    Parameters
    ----------
    A : Attribute (string)
    state : One state of the attribute (string)
    pandacsv : pandas data format
    Returns
    -------
    pk_nk_dict : dictionary containing partial positive and 
    partial negative values
    """
    p_n_rows = pandacsv.copy()  
    rowlist = [] #rowlist = the indicies without the state
    for a in range(0, p_n_rows.index.size): 
        if(p_n_rows.at[a, A] != state): 
            rowlist.append(a) 
    p_n_rows = get_examples(pandacsv, rowlist, [])
    p_k_list = p_n_rows.values.tolist()
    n_k_p_k = len(p_k_list)
    p_k_index = []
    survived_index = titanic_data.columns.get_loc("Survived") #can change this if survived is no longer what we want to classify by
    for i in range(0, n_k_p_k): 
        if(p_k_list[i][survived_index] == '0'): #THIS IS CHANGED
            p_k_index.append(i)
        i+=1
    p_k_index = p_k_index[::-1]
    for i in p_k_index: 
        p_k_list.remove(p_k_list[i])
    n_k = n_k_p_k - len(p_k_list)
    p_k = n_k_p_k - n_k
    
    pk_nk_dict = {"p_k": p_k, "n_k": n_k}
    return pk_nk_dict

def Importance(attributes, examples): 
    """
    Parameters
    ----------
    attributes : list with relevant attributes, given in string-format
    examples : list with examples in pandas-format

    Returns
    -------
    a list with the importancy of different attributes based on functions blah blah
    """

    p = 340 #found using total_p_n(titanic_data). 
    n = 559
    Bppn = B(p/(p+n))
    correct_attribute = ""
    remainders = []
    for A in attributes: 
        no_states = list(find_attribute_states_and_amount(A, examples).keys()) #k        
        remainder = 0
        for i in no_states:
            pk_nk_dict_values = list(partial_p_n(A, i, examples).values()) #pnkndictvalues er feil. 
            pk = int(pk_nk_dict_values[0])#pk_nk_dict_values[0]#pk_nk_dict_values(0)
            nk = int(pk_nk_dict_values[1])#pk_nk_dict_values[1]#pk_nk_dict_values(1)'    
            remainder = remainder + ((pk+nk)/(p+n))*B((pk)/(pk+nk))
        remainders.append(Bppn - remainder)
        #print(remainders)
    for i in remainders: 
        if i == max(remainders): 
            attribute_index = remainders.index(i)
            correct_attribute = attributes[attribute_index]
    return correct_attribute
    
def plurality_value(examples): 
    """
    Will "select the most common output value among a set of examples, breaking ties randomly" -page 702 AIAMA
    
    Parameters
    ----------
    examples : panda dataframe.

    Returns
    -------
    1 or 0 for live or die 
    """
    survived = total_p_n(examples)
    if(survived.get("p") > survived.get("n")): 
        return 1
    return 0

def contImportance(attribute, titanic_file): 
    """
    Parameters
    ----------
    attributes : list with relevant attributes, given in string-format
    examples : list with examples in pandas-format

    Returns
    -------
    a list with the importancy of different attributes based on functions blah blah
    """
    p = 340 #found using total_p_n(titanic_data). 
    n = 559
    Bppn = B(p/(p+n))
    states = ["Under", "Over"]
    remainder = 0
    for i in states: 
        pk_nk_dict_values = list(partial_p_n(attribute, i, titanic_file).values())
        pk = int(pk_nk_dict_values[0])
        nk = int(pk_nk_dict_values[1])
        remainder = remainder + ((pk+nk)/(p+n))*B((pk)/(pk+nk))
    return(Bppn - remainder)

    """
    if(attribute == "Age"): 
        states = ["", "Under", "Over"]
        remainder = 0
        for i in states: 
            pk_nk_dict_values = list(partial_p_n("Age", i, titanic_file).values())
            pk = int(pk_nk_dict_values[0])
            nk = int(pk_nk_dict_values[1])
            if(pk == 0):
                remainder = remainder
            else: 
                remainder = remainder + ((pk+nk)/(p+n))*B((pk)/(pk+nk))
        return(Bppn - remainder)
    """

def contsplit (attribute, titanic_file):
    list_data = titanic_file.values.tolist()
    attribute_index = titanic_data.columns.get_loc(attribute)
    list_data = sorted(list_data, key = lambda x: float(x[attribute_index]))
    
    sorteddata =  pd.DataFrame(list_data,columns=list(titanic_file.columns))
    importance = []
    survived_index = titanic_data.columns.get_loc("Survived") #can change this if survived is no longer what we want to classify by
    
    for a in range (1, len(list_data)): 
        if(list_data[a-1][survived_index] == '0' and list_data[a][survived_index] == '1'): #THIS IS CHANGED
            copydata = sorteddata.copy()
            for b in range(0,a):      
                 copydata.loc[b, attribute]= "Under"
            for c in range(a, len(list_data)): 
                copydata.loc[c, attribute] = "Over"
            importance.append([a, contImportance(attribute,copydata)])
    amax = 0
    for i in range(1, len(importance)): 
        if(importance[i][survived_index] >= importance[i-1][survived_index]): 
            amax = i           
    print(attribute, list_data[amax][attribute_index])
    for b in range(0,amax):      
        sorteddata.loc[b, attribute]= "Under"
    for c in range(amax, len(list_data)): 
           sorteddata.loc[c, attribute] = "Over"
    return (sorteddata) 



    
def decision_tree_learning(examples, attributes, parent_examples): 
    #lage png med graphviz: Se på demo, slik setter du opp variablene. Om du skal lage png med demo går du
    #inn på anaconda prompt og skriver "dot -Tpng demo.dot -o demo_dot.png"
    """
    Parameters
    ----------
    examples : list with examples in pandas-format
    attributes : list with relevant attributes, given in string-format
    parent_examples : earlier lines in the training set? 
    tree : map with attribute as key and states of attributes as items 

    Returns
    -------
    a decision tree.
    """
    if(examples.index.size == 0): 
        return plurality_value(parent_examples)
    elif(same_classification(examples)): 
        return examples.loc[0,"Survived"]
    elif(len(attributes)== 0): 
        return plurality_value(examples)
    else: 
        A = Importance(attributes, examples)
        tree = Tree(parents = {}, children = {})
        vk = set(examples[A].tolist())
        
        for value in vk: 
            exs = examples.copy() #exs er good nå
            for b in range(0, exs.index.size):
                if (exs.loc[b, A] != value): 
                    exs.drop([b], inplace = True)
            exs.reset_index(inplace = True, drop = True)
            new_attributes = attributes.copy()
            new_attributes.remove(A) 
            subtree = decision_tree_learning(exs, new_attributes, examples)
            tree.add_value_in_subtree(A, value, subtree)
            tree.make_child_parent()
        return tree.parents
        

def disc_test(titanic_testset): 
    list_testset = titanic_testset.values.tolist()
    sex_index = titanic_data.columns.get_loc("Sex")
    survive_index = titanic_data.columns.get_loc("Survived")
    correct = 0 
    total = len(list_testset)
    for i in range(0, len(list_testset)): 
        if(list_testset[i][sex_index]== 'female'): 
            if(list_testset[i][survive_index] == 1): 
                correct += 1
        else: 
            if(list_testset[i][survive_index] == 0):
                correct += 1
    return (float(correct/total)*100)




if __name__ == '__main__':
    print(B(6/10)-(5/10)*B(4/5) - (5/10)*B(2/5))
    
    """
    
    titanic_data = convert_to_pandas('train.csv')
    titanic_test = convert_to_pandas('test.csv')
    print("The accuracy of the decision tree supporting discrete variables")
    print(disc_test(titanic_test))
    
    
    discrete_tree = decision_tree_learning(titanic_data, ["Sex", "Pclass"], {})
    for key in discrete_tree:
        print(key, '\n', discrete_tree[key],"\n")

    
    cont = contsplit("SibSp", titanic_data)
    cont = contsplit("Parch", cont)
    cont = contsplit("Fare", cont)

    continous_tree = decision_tree_learning(cont, ["Sex", "Pclass","Parch", "SibSp", "Fare"],{}) 
    for key in continous_tree:
        print(key, '\n', continous_tree[key],"\n")
        
    """
  
    
    
    
    

    
    
    
    
    
    

"""
def contsplit_age (titanic_file):
    list_data = titanic_file.values.tolist()
    unknowns = []
    for item in list_data: 
        if(item[4] != ''):
            unknowns.append(item)
            list_data.remove(item)     
    unknowns = sorted(unknowns, key = lambda x: float(x[4]))
    for i in range(0, len(unknowns)): 
        list_data.append(unknowns[i])
    sorteddata =  pd.DataFrame(list_data,columns=list(titanic_file.columns))
    importance = []
    for a in range (1, len(list_data)): 
        if(list_data[a-1][0] == '0' and list_data[a][0] == '1'): 
            copydata = sorteddata.copy()
            for b in range(0,a):      
                 copydata.loc[b, "Age"]= "Under"
            for c in range(a, len(list_data)): 
                copydata.loc[c, "Age"] = "Over"
            importance.append([a, contImportance("Age", copydata)])
        elif(list_data[a-1][0] == '1' and list_data[a][0] == '0'):
             copydata = sorteddata.copy()
             for b in range(0,a):      
                copydata.loc[b, "Age"]= "Over"
             for c in range(a, len(list_data)): 
                 copydata.loc[c, "Age"] = "Under"
             importance.append([a, contImportance("Age", copydata)])
    amax = 0
    for i in range(1, len(importance)): 
        if(importance[i][0] >= importance[i-1][0]): 
            amax = i           
    for b in range(0,amax):      
        sorteddata.loc[b, "Age"]= "Under"
    for c in range(amax, len(list_data)): 
        sorteddata.loc[c, "Age"] = "Over"
    return (sorteddata) 

"""
    
    
    
    
    
    
    
  