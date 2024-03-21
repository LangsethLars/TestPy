"""
Just some testing of what Python is capable of.
The code is not optimized and is just for testing purposes.
I wanted to make a comparison between a simple neural network
and a Keras neural network on the MNIST dataset.
The simple neural network is a shallow neural network with
one hidden layer and binary classification for digit 5.
"""
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense



# Function to display the first 9 images of the training set
def display_9_images(train_X, train_y) -> None:
    for i in range(9):
        plt.subplot(330 + 1 + i) # 3 rows, 3 columns, i+1
        plt.ylabel(train_y[i])
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    plt.show()



# Training parameters on set of handwritten digits, binary classification for digit 5
def train_shallow_nn_binary_classification(x_train, y_train, nh, alpha, num_iterations):

    m_train = x_train.shape[0]  # Number of training examples
    nx = x_train.shape[1]       # Number of features
    ny = 1                      # Number of output units

    w0 = np.zeros((nx,nh)) # Weights matrix for input layer to hidden layer
    b0 = np.zeros((1,nh))  # Bias vector for input layer to hidden layer

    w1 = np.zeros((nh,ny)) # Weights matrix for hidden layer to output layer
    b1 = np.zeros((1,ny))  # Bias vector for hidden layer to output layer

    for i in range(num_iterations):

        # Forward propagation
        z0 = np.dot(x_train,w0) + b0
        h = 1 / (1 + np.exp(-z0))
        z1 = np.dot(h,w1) + b1
        y_pred = (1 / (1 + np.exp(-z1))).reshape(m_train)

        # Cost function
        J = -np.mean(y_train * np.log(y_pred) + (1-y_train) * np.log(1-y_pred))
        if i % 100 == 0:
            print(f'J{i}: {str(J)}')

        # Back propagation

        dz1 = (y_pred - y_train).reshape(m_train,ny)
        dw1 = (np.dot(h.T,dz1) / m_train).reshape(nh,ny)
        db1 = (np.sum(dz1, axis=0) / m_train).reshape(ny,ny)

        w1 -= alpha * dw1
        b1 -= alpha * db1

        dz0 = np.dot(dz1,w1.T) * h * (1 - h)
        dw0 = np.dot(x_train.T,dz0) / m_train
        db0 = (np.sum(dz0, axis=0) / m_train).reshape(ny,nh)

        w0 -= alpha * dw0
        b0 -= alpha * db0

    return w0, b0, w1, b1



def count_correct_predictions(y_test, y_pred) -> None:
    # Display the number of correct and failed predictions for digit 5 and not digit 5
    is_5 = 0
    is_5_correct = 0
    is_5_failed = 0
    is_not_5 = 0
    is_not_5_correct = 0
    is_not_5_failed = 0
    for i in range(y_test.shape[0]):
        if y_test[i] == 1.0:
            is_5 += 1
            if y_pred[i] >= 0.5:
                is_5_correct += 1
            else:
                is_5_failed += 1
        else:
            is_not_5 += 1
            if y_pred[i] < 0.5:
                is_not_5_correct += 1
            else:
                is_not_5_failed += 1
    print(f"There are {is_5} digit 5 and {int(is_5_correct*100/is_5)}% was predicted correct")
    print(f"There are {is_not_5} not digit 5 and {int(is_not_5_correct*100/is_not_5)}% was predicted correct")



def test_shallow_nn_multi_classification(x_test, y_test, w0, b0, w1, b1) -> None:

    m_test = x_test.shape[0]  # Number of test examples

    # Forward propagation
    z0 = np.dot(x_test,w0) + b0
    h = 1 / (1 + np.exp(-z0))
    z1 = np.dot(h,w1) + b1
    y_pred = (1 / (1 + np.exp(-z1))).reshape(m_test)

    # Cost function
    J = -np.mean(y_test * np.log(y_pred) + (1-y_test) * np.log(1-y_pred))
    print(f'J test for {m_test} examples: {str(J)}')

    count_correct_predictions(y_test, y_pred)



# Main function
def main() -> None:

    print("Start:")

    # Read training set of handwritten digits
    (x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()
    print('x_train_full: ' + str(x_train_full.dtype) + ' ' + str(x_train_full.shape)) # x_train_full: uint8 (60000, 28, 28)
    print('y_train_full: ' + str(y_train_full.dtype) + ' ' + str(y_train_full.shape)) # y_train_full: uint8 (60000,)
    print('x_test_full:  ' + str(x_test_full.dtype)  + ' ' + str(x_test_full.shape))  # x_test_full:  uint8 (10000, 28, 28)
    print('y_test_full:  ' + str(y_test_full.dtype)  + ' ' + str(y_test_full.shape))  # y_test_full:  uint8 (10000,)

    # Display the 9 first handwritten digits
    display_9_images(x_train_full, y_train_full)

    # Shallow neural network with binary classification for digit 5
    nx = x_train_full.shape[1] * x_train_full.shape[2] # Number of features, 784 = 28 * 28
    nh = 20 # Number of hidden units
    alpha, num_iterations = 0.1, 10000
 
    # Reshape the training set to a 2D array and limit it to 2000 examples for speed
    m_train = 2000 # Number of training examples
    x_train = x_train_full[0:m_train,:,:].reshape(m_train,nx) / 255 # x:  float64 (2000, 784)
    y_train = y_train_full[0:m_train] # 0 to 9, y_train:  uint8 (2000,)
    y_train = np.where(y_train == 5, 1.0, 0.0) # Binary classification for digit 5, y:  float64 (2000,)

    w0, b0, w1, b1 = train_shallow_nn_binary_classification(x_train, y_train, nh, alpha, num_iterations)

    # Reshape the test set to a 2D array
    m_test = x_test_full.shape[0] # Number of test examples
    x_test = x_test_full[0:m_test,:,:].reshape(m_test,nx) / 255 # x_test:  float64 (1000, 784)
    y_test = y_test_full[0:m_test] # 0 to 9, y_test:  uint8 (1000,)
    y_test = np.where(y_test == 5, 1.0, 0.0) # Binary classification for digit 5, y:  float64 (1000,)

    test_shallow_nn_multi_classification(x_test, y_test, w0, b0, w1, b1)
    # J test for 10000 examples: 0.12188542193303845
    # There are 892 digit 5 and 76% was predicted correct
    # There are 9108 not digit 5 and 98% was predicted correct

    # Now let us do the same with Keras
    model = Sequential()
    model.add(Dense(nh, input_dim=nx, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    model.fit(x_train, y_train, epochs=100, verbose=1)
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')
    y_pred = model.predict(x_test)
    count_correct_predictions(y_test, y_pred)
    # There are 892 digit 5 and 83% was predicted correct
    # There are 9108 not digit 5 and 98% was predicted correct

    print("Finished!")



if __name__ == '__main__':
    main()
