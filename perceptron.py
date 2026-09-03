import numpy as np

def perceptron(inputs , weights):
    s = []
    #Bias Weight is last in list
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
            activation.append(0)
    return activation

def map_predictions(outputs):
    new = []
    for output in outputs :
        if output == 0 :
            new.append(-1)
        else :
            new.append(1)
    return new
    
#Self Learning Algo
def self_learning(input , output, weights , learning_rate):
    og_answers_tau_outputs = map_predictions(output)
    
    final_tau_outputs = perceptron(inputs = or_table , weights = initial_weights)
    total_errors = [x - y for x, y in zip(og_answers_tau_outputs , final_tau_outputs)]
    
    while sum(total_errors) != 0:
        pass
    
    pass

#Define Truth Tables
or_table = [[0 , 0 ],
            [0 , 1 ],
            [1 , 0 ],
            [1 , 1 ]]

or_table_outputs = [0,
                    1,
                    1,
                    1,]

and_table = [[0 , 0 ],
             [0 , 1 ],
             [1 , 0 ],
             [1 , 1 ]]

and_table_output = [0,
                    0,
                    0,
                    1]

#Initialize random weights between -5 and 5
# Bais = Index -1
initial_weights = np.random.uniform(-5, 5, 3)

self_learning(input= or_table , output= or_table_outputs , weights= initial_weights)

sumations_list = perceptron(inputs = or_table , weights = initial_weights)