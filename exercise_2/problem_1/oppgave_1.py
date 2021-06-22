# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 13:50:19 2021

@author: Hanne-Grete 



Use programming to solve all exercises in this section involving computation. The results need to
be extracted from the program and well documented in a human-readable format that is easy to
understand in a PDF file. Additionally, write a few sentences to give the results some context. The
results can, for example, be plotted using Matplotlib [1] to give a more straightforward overview.
The code must be runnable without any modifications after delivery. Moreover, the code must be
readable and contain comments explaining it. We recommend that Python with the package NumPy
[2] be used for the programming exercises. It is not allowed to use libraries, such as Scikit-learn [3]
to solve the tasks.

"""
"""
I will use forward and backward as explained in chapter 15 in AIMA, to filter, predict and so on. 
"""

import matplotlib.pyplot as plt
import numpy as np


def forward_eq(O, T, f):
    """
    Parameters
    ----------
    O : np.matrix for evidence, Ot+1
    T : np.matrix for X_t given X_t-1
    f : f_1:t

    Returns
    -------
    f1:t+1
    
    
    This is just my forward algorithm, as it is written in page 579, eq. (15.12)
    in the book "Artificial Intelligence A modern approach"
    """
    T_T = T.transpose()
    f_vector = O@T_T@f #matrix-multiplication is done like this
    norm_const = f_vector[0] + f_vector[1]
    return(f_vector/norm_const)
    
def backward_eq(O, T, b):
    """
    

    Parameters
    ----------
    O : np.matrix for evidence, Ok+1
    T : np.matrix for X_t given X_t-1
    b : b_k+2:t

    Returns
    -------
    b_k+1:t
    
    This is just my backward algorithm, as it is written in page 579, eq. (15.13)
    in the book "Artificial Intelligence A modern approach"

    """
    return (T@O@b)

def forward_eq_no(O, T, f):
    """
    Parameters
    ----------
    O : np.matrix for evidence, Ot+1
    T : np.matrix for X_t given X_t-1
    f : max number 

    Returns
    -------
    f1:t+1
    
    This I will use in the mls-problem. It basically does the same as 
    forward_eq. 

    """
    
    #T_T = T.transpose()
    f_mat = (O@T) #matrix-multiplication is done like this
    f_vector = f_mat.dot(f)
    return(f_vector)

def forward_eq_no_norm(f):
    """
    Parameters
    ----------
    O : np.matrix for evidence, Ot+1
    T : np.matrix for X_t given X_t-1
    f : max number 

    Returns
    -------
    normalized f
    
    This I will use in the mls-problem. It basically does the same as 
    forward_eq. 

    """
    
    #T_T = T.transpose()
    norm_const = f[0] + f[1]
    return(f/norm_const)
        

    

def problem_1b():
    """
    This is a filteringproblem, since it should compute the belief state. This means that I don't know what will happen,
    and I only have evidence for that day and the ones before, but the most likely scenario based on the evidence is what I 
    get out. Hence, the solution can be provided by the forward algorithm. 
     
    """
    print("\n Problem 1b: filtering")
    
    #write out the probabilities and evidences I know
    forward_list = []
    O_found =[1,1,0,1,0,1]
    x_0 = np.matrix("0.5; 0.5")
    T = np.matrix("0.8, 0.3; 0.2, 0.7")
    O_birds = np.matrix("0.75, 0; 0, 0.2")
    O_no_birds = np.matrix("0.25, 0; 0, 0.8")
    f1 = forward_eq(O_birds, T, x_0)
    forward_list.append(f1)
    
    #compute the beliefstate for he x's 
    for i in range(1, len(O_found)): 
        if(O_found[i] == 1): 
            f = forward_eq(O_birds, T, forward_list[i-1])
            forward_list.append(f)
        else: 
            f = forward_eq(O_no_birds, T, forward_list[i-1])
            forward_list.append(f)
            
    #plot a nice figure and print out the values         
    fig_f, ax = plt.subplots(2,1)
    ax[0].set_xlabel("time")
    ax[1].set_xlabel("time")
    ax[0].set_ylabel("probability of X_t")
    ax[1].set_ylabel("probability of not X_t")
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(0,1)
    fig_f.suptitle("Filtering probabilities")
    for i in range(0, len(O_found)): 
        #print(forward_list[i][0])
        ax[0].scatter([i+1], [forward_list[i][0]])
        ax[1].scatter([i+1], [forward_list[i][1]])
        print("\n P(X" + str(i+1) + "|e" + str(1) + ":" + str(i+1) + ") = " + str(np.transpose(forward_list[i])))
    fig_f.show()          
        
    
        
def problem_1c(): 
    """
    Here I want to predict forward the future. 
    I calculate T with the previous predition to do that. 
    This operation gives me a look of what will happen in the future when I only have the evidence I have.  
    As can be seen, the solution after a while converges to the value [0.6, 0.4].
    """
    print("\n \n Problem 1c: prediction")
    
    T = np.matrix("0.8, 0.3; 0.2, 0.7")
    P6_ev = np.matrix("0.72719639; 0.27280361") #Here I assume that the answers I got in b) were right, and I take the last answer.
    P5_ev = np.matrix("0.33650851; 0.66349149")
    prediction_table = []
    
    fig_p, ax = plt.subplots(2,1)
    ax[0].set_xlabel("time")
    ax[1].set_xlabel("time")
    ax[0].set_ylabel("probability of X_t")
    ax[1].set_ylabel("probability of not X_t")
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(0,1)
    fig_p.suptitle("Prediction probabilities")
    
    
    
    prediction_table.append(P6_ev)
    for i in range(7, 31): 
        prediction_table.append(T@prediction_table[i-7])
        
    for i in range(1, len(prediction_table)):
        ax[0].scatter([i+6], [prediction_table[i-1][0]])
        ax[1].scatter([i+6], [prediction_table[i][1]])
        print("\n P(X" + str(i+6) + "|e" + str(1) + ":" + str(6) + ") = " + str(np.transpose(prediction_table[i])))
    


def problem_1d(): 
    """
    Here I am already given all evidence, and I use all of it to calculate my states of x_t, so
    this is smoothing. I use the formula given in lecture 4 for smoothing with forward and backward.
    To make it easy for myself, I just reuse the code from 1b, then I find all
    the b's using the backward equation and multiply them together. I then normalize them.
    
    This process gives me a more accurate estimate of previous states, since I now can use all of 
    the evidence. 
    """
    print("\n \n problem 1d: smoothing")
    
    #from 1b
    forward_list = []
    O_found =[1,1,0,1,0,1]
    x_0 = np.matrix("0.5; 0.5")
    
    T = np.matrix("0.8, 0.3; 0.2, 0.7")
    O_birds = np.matrix("0.75, 0; 0, 0.2")
    O_no_birds = np.matrix("0.25, 0; 0, 0.8")
    #t = 5
    f1 = forward_eq(O_birds, T, x_0)
    forward_list.append(f1)
    
    for i in range(1, len(O_found)): 
        if(O_found[i] == 1): 
            f = forward_eq(O_birds, T, forward_list[i-1])
            forward_list.append(f)
        else: 
            f = forward_eq(O_no_birds, T, forward_list[i-1])
            forward_list.append(f)
            
    #now for backward
    backward_list = [0, 0, 0, 0, 0, 0]
    b6 = np.matrix("1; 1") #The last b is always 1,1
    backward_list[5] = b6
    for i in range(4, -1, -1):
        if(O_found[i] == 1): 
            b = backward_eq(O_birds, T, backward_list[i+1])
            backward_list[i] = b
        else: 
            b = backward_eq(O_no_birds, T, backward_list[i+1])
            backward_list[i] = b
    
    normalized_table =[]  
    for i in range(0,6):
        f_b_vec_1 = forward_list[i]@np.transpose(backward_list[i])   
        f_b_vec = np.array([f_b_vec_1[0,0], f_b_vec_1[0,1]])
        norm_const = f_b_vec[0] + f_b_vec[1] 
        normalized_table.append(f_b_vec/norm_const)
        
        
    fig_s, ax = plt.subplots(2,1)
    ax[0].set_xlabel("time")
    ax[1].set_xlabel("time")
    ax[0].set_ylabel("probability of X_t")
    ax[1].set_ylabel("probability of not X_t")
    ax[0].set_ylim(0,1)
    ax[1].set_ylim(0,1)
    fig_s.suptitle("Smoothing probabilities")
    
    
    for i in range(0, len(normalized_table)):
        ax[0].scatter([i+1], [normalized_table[i][0]])
        ax[1].scatter([i+1], [normalized_table[i][1]])
        print("\n P(X" + str(i) + "|e" + str(1) + ":" + str(6) + ") = " + str(normalized_table[i]))
    


    
    

def problem_1e():
    """
    This problem was much trying and failing. I ended up using the formula 
    on page 577 in the book "Artificial Intelligence A Modern Approach", 
    and I figured out which matrices I could use
    to end up with the same multiplication as they did there. 
    As one can see, the most likely sequence is actually that all states are 1
    = true, and this is probably due to the probability of fish on day t given 
    fish and no fish on day t-1 being higher than not getting fish given fish 
    and no fish on day t-1, respectively, as well as the evidence mostly being true. 
    
    Below I tried using it on the problem on page 577, to see if I got the correct results. 
    
    """
    """
    Tried doing it with 
    O_found = [1, 1, 0, 1, 1]
    x0 = np.matrix("0.5; 0.5")
    T = np.matrix("0.7, 0.3; 0.3, 0.7")
    T_fish = ("0.7, 0; 0, 0.3")
    T_no_fish = ("0.3, 0; 0, 0.7")
    mls_table = []
    mls_true_false = []
    
    O_birds = np.matrix("0.9, 0; 0, 0.2")
    O_no_birds = np.matrix("0.1, 0; 0, 0.8")
    x1 = forward_eq(O_birds, T, x0)
    mls_table.append(x1)
    print(x1)
    print("\n")
    
    if(x1[0] > x1[1]): 
        mls_true_false.append(1)
    else: 
        mls_true_false.append(0)
    
    
    for i in range(1, 5): 
        x_before = mls_table[i-1]
        if(x_before[0] > x_before[1]): 
            x_before[1] = x_before[0]
            T = np.matrix(T_fish)
        else: 
            T = np.matrix(T_no_fish)
            x_before[0] = x_before[1]

        if(O_found[i] == 1): 
            O = np.matrix(O_birds)
        else: 
            O = np.matrix(O_no_birds)
        f = forward_eq_no(O, T, x_before)
        mls_table.append(f)
        print(str(f) + "\n")
        if(f[0] > f[1]): 
            mls_true_false.append(1)
            
        else: 
            mls_true_false.append(0)
            
    
    print("\n")
    print(mls_true_false)
    print("\n")
    #print(str(mls_table) + "\n")
    """
    print("\n \n problem 1e: most likely sequence")
    O_found =[1,1,0,1,0,1]
    mls_table = []
    mls_true_false = []
    mls_norm = []
    x0 = np.matrix("0.5; 0.5")
    T = np.matrix("0.8, 0.3; 0.2, 0.7")
    O_birds = np.matrix("0.75, 0; 0, 0.2")
    O_no_birds = np.matrix("0.25, 0; 0, 0.8")
    T_fish = np.matrix("0.8, 0; 0, 0.2") #If max prob for last day was fish
    T_no_fish = np.matrix("0.3, 0; 0, 0.7")#if max prob for last day was no fish
    
    x1 = forward_eq_no(O_birds, T, x0)
    x1_norm = forward_eq_no_norm(x1)
    mls_norm.append(x1_norm)
    mls_table.append(x1) #based on formula on page 577 
    #print(str(x1)+ "\n")
    #print (str(x1_norm)+"\n")
    if(x1[0] > x1[1]): 
        mls_true_false.append(1)
    else: 
        mls_true_false.append(0)
    
    
    for i in range(1, 6): 
        x_before = mls_table[i-1]
        if(x_before[0] > x_before[1]): 
            x_before[1] = x_before[0]
            T = np.matrix(T_fish)
        else: 
            T = np.matrix(T_no_fish)
            x_before[0] = x_before[1]

        if(O_found[i] == 1): 
            O = np.matrix(O_birds)
        else: 
            O = np.matrix(O_no_birds)
        f = forward_eq_no(O, T, x_before)
        f2 = forward_eq_no_norm(f)
        mls_norm.append(forward_eq_no_norm(f))
        mls_table.append(f)
        #print(str(f) + "\n")
        #print(str(f2) + "\n")
        if(f[0] > f[1]): 
            mls_true_false.append(1)
            
        else: 
            mls_true_false.append(0)
            

            
    
    print("\nMost likely truth values:\n")
    print(mls_true_false)
    print("\n")
    
    fig_mls, ax = plt.subplots(2,1)
    ax[0].set_xlabel("time")
    ax[1].set_xlabel("time")
    ax[0].set_ylabel("Max probability not normalized")
    ax[1].set_ylabel("Normalized max probability")
    ax[0].set_ylim(-0.04,0.5)
    ax[1].set_ylim(0.4,1)
    fig_mls.suptitle("Max values in the most likely sequence")
    print("Most likely sequence, not normalized:\n")
    
    for i in range(0, len(mls_table)):
        ax[0].scatter([i+1], [mls_table[i][0]])
        ax[1].scatter([i+1], [mls_norm[i][0]])
        print("X" + str(i+1) + "=" +str(mls_table[i][0]) + "\n")
    print("Most likely sequence, normalized \n")
    for i in range(0, len(mls_table)):
        print("X" + str(i+1) + "=" +str(mls_norm[i][0]) + "\n")
        
    
    
    


if __name__ == '__main__':
    problem_1b()
    problem_1c()
    problem_1d()
    problem_1e()