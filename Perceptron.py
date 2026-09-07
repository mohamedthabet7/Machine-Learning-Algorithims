import numpy as np

def perceptron(inputs , weights):
    s = []
    base_weight = weights[-1]
    #Go over the every combination
    for combination_of_inputs in inputs:
        summation = base_weight
        
        #Itterate through each combinatoin to get sum
        for  i, element in enumerate(combination_of_inputs):
            summation += element*weights[i]
            
        #Add Sum to the list
        s.append(summation)
    
    activiation_list = thresh_tau(s)
    return activiation_list

def thresh_tau(sums):
    activation = []
    for s in sums:
        if s >= 0:
            activation.append(1)
        else:
            activation.append(-1)
    return activation

        
#Define Truth Tables
or_table = [[0 , 0 ],
            [0 , 1 ],
            [1 , 0 ],
            [1 , 1 ]]

#Weights
weights = [1 , 1 , -0.5]

sumations_list = perceptron(inputs = or_table , weights = weights)
activation_list = thresh_tau(sumations_list)

print(f"Or Table \nX1  X2  Tau(X)")

for i in range(len(or_table)):
    print(f"{or_table[i][0]}   {or_table[i][1]}   {activation_list[i]}")