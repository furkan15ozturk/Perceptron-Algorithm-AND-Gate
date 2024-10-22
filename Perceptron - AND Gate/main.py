from random import random

#initializes a random value to weights for x0, x1, x2 between 0 and 1
def initialize_weights(training_data, _weights):
    for i in range(len(training_data) - 1):
        _weights.append(random())
    _weights[0] = 0 # this will ensure the bias value by making w0 = 0
    return _weights

# adds integer 1 to each data row
def transform_training_data(training_data):
    for i in range(len(training_data)):
        training_data[i].insert(0, 1)
    return training_data

def sgn(dot_product):
    if dot_product > 0:
        return 1 # class 1 y = 0
    elif dot_product <= 0:
        return -1 # class 2 y = 1

# calculates x0*w0 + x1*w1 + x2*w2 for given row of data, the dot product of w(n) and x(n)
# this is called Z score
def calculate_dot_product(x, _weights):
    dot_product = 0
    for i in range(len(x) - 1):
        dot_product += x[i] * _weights[i]
    return dot_product

def calculate_d(x):
    if x == 0: # class 1 y = 0
        return 1
    elif x == 1: # class 2 y = 1
        return -1

def adjust_weight(training_data, _weights, _learning_rate = 0.5, epoch_count = 10):
    for _ in range(epoch_count):
        for i in range(len(training_data)):
            dot_product = calculate_dot_product(training_data[i], _weights) # dot product of  ith row
            y_n = sgn(dot_product)
            d_n = calculate_d(training_data[i][-1])
            print(f"dot_product: {dot_product}, y_n: {y_n}, d_n: {d_n}")
            if d_n is None or y_n is None:
                raise ValueError(f"Fonksiyonlar None değeri döndürüyor: y_n: {y_n}, d_n: {d_n}")
            for j in range(len(training_data[0]) - 1): # parameter count
                _weights[j] += learning_rate * (d_n - y_n) * training_data[i][j]
    return _weights

def predict(user_input, _weights):
    dot_product = 0
    for i in range(len(_weights)):
        dot_product += user_input[i] * _weights[i]
    return dot_product


def perceptron(training_data, _learning_rate, epoch_count):
    weights = []
    training_data = transform_training_data(training_data)
    weights = initialize_weights(training_data, weights)
    adjusted_weights = adjust_weight(training_data, weights, _learning_rate, epoch_count)
    prediction_list = []
    for i in range(len(training_data)):
        prediction = sgn(predict(training_data[i], adjusted_weights))
        if prediction == 1:
            prediction_list.append(0) # class 1
        elif prediction == -1:
            prediction_list.append(1) # class 1
    return prediction_list, adjusted_weights




if __name__ == "__main__":
    #input        x_1     x_2    Y
    inputList = [[0,      0,     0],
                 [0,      1,     0],
                 [1,      0,     0],
                 [1,      1,     1]]
    learning_rate = 0.1
    epoch = 1000
    pred_list, weights = perceptron(inputList, learning_rate, epoch)
    print(f"Predictions: {pred_list}")
    print(f"The formula: {weights[0]} + {weights[1]} * X1 + {weights[2]} * X2")