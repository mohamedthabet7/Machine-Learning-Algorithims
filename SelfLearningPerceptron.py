import numpy as np

#Self Learning perceptron, Return Final Prections, weights and epochs
def perceptron(inputs , outputs , init_weights , learning_rate):
    
    #Create Epoch
    epoch = 0
    
    #Bias Weight is last in list
    base_weight = init_weights[-1]
    weights = init_weights[:-1]
    
    #Convert Correct Outputs with same thresh function
    y = list(map(lambda x: -1 if x == 0 else 1, outputs))
    #Learning loop
    run = True
    while run :
        
        #Total Errors For loop exit check
        total_errors = []
        
        #Final Predictions to store incase of loop exit
        final_predictions = []
        
        #Itterate Over Each combination of inputs
        
        for i, xs in enumerate(inputs):
            
            #Add the base weight
            summation = base_weight
             
            #Summation of whole Row
            for  k, x in enumerate(xs):
                summation += x*weights[k]
                
            #Activation Function For each single combination
            yhat = thresh_tau(summation)
            
            final_predictions.append(yhat)
            #Calculate Error and Store it For loop exit Check
            error = y[i] - yhat
            total_errors.append(error)
            
            #Check Error at each Combination, Then update weights before next Combination
            if error != 0:
                base_weight = base_weight + (learning_rate * error)
                for j , weight in enumerate(weights):
                    weights[j] = weights[j] + (learning_rate * error * xs[j])
        
        for i ,x in enumerate(weights):
            print(f"weight{i} : {x}")
            
        #Update Epoch Count
        epoch += 1
        
        #Check If there is any errors left after
        if all(error == 0 for error in total_errors):
            run = False
        print(f"Current Epoch {epoch}")
        
    #Add the base Weight to the weigths list        
    weights = np.append(weights, base_weight)    
    
    return final_predictions , weights , epoch
                    
def thresh_tau(s):
        if s >= 0:
            return 1
        else:
            return -1

#Define Truth Tables with correct output
table = [[0 , 0 ],
            [0 , 1 ],
            [1 , 0 ],
            [1 , 1 ]]

or_output = [0,
                    1,
                    1,
                    1,]

and_output = [0,
                    0,
                    0,
                    1]

#Initialize random weights between -5 and 5
# Bais = Index -1
initial_weights = np.random.uniform(-5, 5, 3)

#Learning Rate
alpha = 0.1

#Change to correct weights and output return
predicted_y , weights , epochs= perceptron(inputs = table , outputs = or_output, init_weights = initial_weights , learning_rate = 0.1)

#Display Final Weights
print(f"Base weight: {weights[-1]:.2f}")
print(f"w1: {weights[0]:.2f}")
print(f"w2: {weights[1]:.2f}")

#Display Final Table
print(f"Table \nX1  X2  Tau(X)")

for i in range(len(table)):
    print(f"{table[i][0]}   {table[i][1]}   {predicted_y[i]}")