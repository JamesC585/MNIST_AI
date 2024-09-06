import numpy as np
import matplotlib.pyplot as plt
import struct

class MLP:
    def __init__(self, label, img, test_label, test_img):
        #Training and Test data initialization
        self.label = label
        self.img = img
        self.test_label = test_label
        self.test_img = test_img

        #Weight and Bias initialization
        self.W1 = np.random.rand(10, 784)
        self.W2 = np.random.rand(10, 10)
        self.B1 = np.random.rand(10, 1)
        self.B2 = np.random.rand(10, 1)



    def relu(self, input):
        """Relu activation, if input < 0, then return 0, else return input"""
        return np.maximum(0, input)
    
    def softmax(self, input):
        """Softmax activation, returns 1-D array of values between (0, 1]"""
        input = input.T
        for i in range(len(input)):
            input[i] = input[i] - input[i].max()
            input[i] = np.exp(input[i])/np.sum(np.exp(input[i]))

        return input.T

    def forwardfeed(self, img):
        """"Forward propagation of weights and bias"""
        input_1 = img/255.0                                                    #(60000, 28, 28)
        output_1 = self.W1.dot(input_1.reshape(-1, 784).T) + self.B1           #(10, 784) * (784, 60000) = (10, 60000) + (10, 1) = (10, 60000)
        activate_1 = self.relu(output_1)
        

        output_2 = self.W2.dot(activate_1) + self.B1                           #(10, 10) * (10, 60000)  = (10, 60000) + (10, 1) = (10, 60000)
        activate_2 = self.softmax(output_2)                                    #(10, 60000)

        return input_1, output_1, activate_1, output_2, activate_2
        
    def one_hot(self):
        """Returns an array of 0s except for one position that represents true label"""
        one_hot_array = np.zeros((len(self.label), np.max(self.label) + 1))
        one_hot_array[np.arange(len(self.label)), self.label] = 1
        return one_hot_array.T


    def cross_entropy(self, output_layer):
        """Subtract output layer(Softmax) with One_hot to flip a value of output layer to negative for backpropagation"""
        one_hot_array = self.one_hot()
        return (output_layer - one_hot_array)
    
    def costFunction(self, output_layer):
        """Cost function using (Softmax - One_hot)"""
        return self.cross_entropy(output_layer)
    
    def deriv_relu(self, input):
        """Derivaion of Relu for Backprop, If input < 0 then return 0, else 1"""
        return input > 0

    def backpropagation(self, input_1, output_1, activate_1, output_2, activate_2, cost_func):
        """Backpropagation using derivatives of formulas in self.forwardfeed() backwards"""
        #(10, 60000) * (60000, 10) = (10, 10)
        dW2 = 1/len(self.img) * np.dot(cost_func, activate_1.T)      
        dB1 = 1/len(self.img) * np.sum(cost_func, 1).reshape(10, 1)


        #(10, 10) * (10, 60000) = (10, 60000) * (10, 10) = (10, 60000)
        dW1 = np.dot(self.W2, cost_func)
        dW1 = dW1 * self.deriv_relu(activate_1)
        #(10, 10) = (10)
        dB2 = 1/len(self.img) * np.sum(dW1, 1).reshape(10, 1)

        #(10, 60000) * (60000, 784)
        dW1 = 1/len(self.img) * np.dot(dW1, self.img.reshape(-1, 784))

        

        self.B1 = np.subtract(self.B1, .01*  dB1) # Learning rate .01
        self.B2 = np.subtract(self.B2, .01*  dB2)
        self.W1 = np.subtract(self.W1, .01 * dW1)
        self.W2 = np.subtract(self.W2, .01 * dW2)


    def get_accuracy(self, activate_2, label):
        """Calculate Accuracy of predicted labels with true labels"""
        return (np.sum(np.argmax(activate_2, 0) == label)/ label.size)
    
    
    def train(self):
        """Training using Gradient descent over i iterations using Forwardpropagation, cost estimation, and Backpropagation"""
        for i in range(2001):
            input_1, output_1, activate_1, output_2, activate_2 = self.forwardfeed(self.img)
            # if(i%50 == 0):
                # print("Iteration", i)
                # print("Accuracy", self.get_accuracy(activate_2, self.label))
                # plt.imshow(tr_img_data[2,:,:], cmap='gray')
                # plt.show()
            cost_func = self.costFunction(activate_2)                       #(10, 60000)
            self.backpropagation(input_1, output_1, activate_1, output_2, activate_2, cost_func)

        return (self.W1, self.W2, self.B1, self.B2)
    
    def predict(self):
        """Predict using trained weights and bias on test data"""
        input_1, output_1, activate_1, output_2, activate_2 = self.forwardfeed(self.img)
        tr_accuracy = self.get_accuracy(activate_2, self.label)
        input_1, output_1, activate_1, output_2, activate_2 = self.forwardfeed(self.test_img)
        val_accuracy = self.get_accuracy(activate_2, self.test_label)
        return (tr_accuracy, val_accuracy)




class CNN:
    def __init__(self, tr_x, tr_y, test_x, test_y):
        #Training and Test data initialization
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.test_x = test_x
        self.test_y = test_y

        #Weight and Bias initialization
        np.random.seed(5)
        self.F1 = np.random.rand(10, 3, 3)
        self.F2 = np.random.rand(10, 10, 3, 3)
        self.W1 = np.random.rand(10, 196)
        self.B1 = np.random.rand(10, 1)
        self.B2 = np.random.rand(10, 1)
        self.B3 = np.random.rand(10, 1)



    def relu(self, input):
        """Relu activation, if input < 0, then return 0, else return input"""
        return np.maximum(0.01*input, input)
    
    def softmax(self, input):
        """Softmax activation, returns 1-D array of values between (0, 1]"""
        input = input.T
        for i in range(10):
            input[i] = input[i] - input[i].max()
            input[i] = np.exp(input[i])/np.sum(np.exp(input[i]))
        return input.T
    
    def one_hot(self, index):
        """Returns an array of 0s except for one position that represents true label"""
        one_hot_array = np.zeros((1, np.max(self.tr_y) + 1))
        one_hot_array[np.arange(1), self.tr_y[index]] = 1

        return one_hot_array
    
    def cross_entropy(self, output_layer, index):
        """Subtract output layer(Softmax) with One_hot to flip a value of output layer to negative for backpropagation"""
        one_hot_array = self.one_hot(index)
        return (output_layer - one_hot_array)
    
    def costFunction(self, output_layer, index):
        """Cost function using (Softmax - One_hot)"""
        return self.cross_entropy(output_layer, index)
    
    def unpool(self, img, size):
        """Reverse average pooling, expanding the img by size*size"""
        img = img.reshape(-1, 14, 14)
        x = np.repeat(img, size, axis = 2)
        x = np.repeat(x, size, axis = 1)
        return x/.25
    
    def deriv_relu(self, input):
        """Derivaion of Relu for Backprop, If input < 0 then return 0, else 1"""
        one_like = np.ones_like(input)  # Create an array of ones with the same shape as x
        one_like[input < 0] = .01
        return one_like
    
  

    
    def im2col(self, filter_len, filter_num, img_size, stride = 1):
        """Transform image into columns for easier matrix multiplication"""
        #filter_len represents filter size 3 for (3, 3), filter_num represents number of filters which is 10

        #Generate filter [0, ..., filter_len] and repeat for a (filter_len, filter_len) x column
        x_vector = np.tile(np.repeat(np.arange(filter_len) , filter_len), filter_num)
        #Generate filter [0, ..., filter_len] and tile for a (filter_len, filter_len) y column
        y_vector = np.tile(np.tile(np.arange(filter_len) , filter_len), filter_num)

        #Tile x column img_size*img_size times for whole image
        x_vector = np.tile(x_vector, (int((img_size/stride)*(img_size/stride)), 1))
        #Tile y column img_size*img_size times for whole image
        y_vector = np.tile(y_vector, (int((img_size/stride)*(img_size/stride)), 1))

        #Generate increasing numbers by 1 to add onto each of our x and y column
        #(676,)
        every_x_col = np.repeat(np.arange(int(img_size/stride)) * stride, int(img_size/stride)).reshape(-1, 1)
        #(676,)
        every_y_col = np.tile(np.arange(int(img_size/stride)) * stride, int(img_size/stride)).reshape(-1, 1)

        x_vector = x_vector + every_x_col
        y_vector = y_vector + every_y_col

        
        return (x_vector, y_vector)
    
    def col2im(self, input):
        """Outputting N x N error onto gradient"""
        x_pad = np.zeros((10, 30, 30))
        input = input.T.reshape(-1, 10, 9)
        for i in range(28):
            for j in range(28):
                np.add.at(x_pad, (slice(None), np.repeat(np.arange(3), 3)+i, np.tile(np.arange(3), 3)+j), input[(i*28)+j])

        x_pad = x_pad[:, 1:29, 1:29].reshape(-1, 784)/(np.max(np.abs(x_pad)))
        return x_pad

    def clip_gradients(self, gradients, threshold):
        """Gradient Clipping"""
        total = 0

        # Calculate Norm of Gradients
        for grad in gradients:
            total += np.sum(np.square(grad))
        total = np.sqrt(total)
        
        # Clip Gradients with threshold
        if total > threshold:
            for i in range(len(gradients)):
                gradients[i] *= threshold / total

        return gradients


    def forwardfeed(self, tr_img):
        """"Forward propagation of weights and bias"""
        #Total 2(Conv + Relu) -> Pooling -> Flattening -> Softmax
        
        #Padding (28, 28) -> (30, 30)
        X = tr_img.reshape(784)
        tr_img = np.pad(tr_img, (1, 1), 'constant')
        
        #2(Conv + Relu)
        x_vector, y_vector = self.im2col(3, 1, 28)
        #(10, 9) * (9, 784) = (10, 784)
        F1 = self.F1.reshape(10, 9)
        cur_img = np.dot(F1, (tr_img[x_vector, y_vector].T)) + self.B1
        Z1 = cur_img
        cur_img = (self.relu(cur_img))#/np.max(np.abs(cur_img))*255
        A1 = cur_img
        cur_img = cur_img.reshape(-1, 28, 28)
        
        #Padding (10, 28, 28) -> (10, 30, 30)
        cur_img = np.pad(cur_img, ((0, 0), (1, 1), (1, 1)), 'constant')
        x_vector, y_vector = self.im2col(3, 1, 28)
        #(10, 10, 9) * (10, 9, 784) -> Im2col -> (10, 90) * (90, 784) = (10, 784)
        #10 (10 x 3 x 3) filters applied on (10 x 28 x 28) images
        F2 = self.F2.reshape(-1, 90)
        cur_img = (np.dot(F2, np.transpose(cur_img[:, x_vector, y_vector], (0, 2, 1)).reshape(90, -1)) + self.B2)
        Z2 = cur_img
        cur_img = (self.relu(cur_img))
        cur_img = cur_img.reshape(-1, 28, 28)
        A2 = cur_img


        #Pooling, 2x2 stride 2 
        x_vector, y_vector = self.im2col(2, 1, 28, 2)
        #(10, 196)
        cur_img = np.sum(cur_img[:, x_vector, y_vector], axis = 2)/4.0


        #Flattened layer
        #(10, 196) * (196, 10) = (10, 10)
        input_1 = cur_img
        output_1 = self.W1.dot(input_1.T) + self.B3
        activate_1 = self.softmax(output_1)


        return X, F1, Z1, A1, F2, Z2, A2, input_1, output_1, activate_1


    def backpropagate(self, costFunc, X, F1, Z1, A1, F2, Z2, A2, input_1, output_1, activate_1):
        """Backpropagation using derivatives of formulas in self.forwardfeed() backwards"""
        #(10, 10) * (10, 196) = (10, 196)
        dW1 = np.dot(costFunc.T, input_1)
        #sum(10, 10) = (10, 1)
        dB3 = np.sum(costFunc.T, axis=1).reshape(10, 1)

        #(10, 10) * (10, 196) = (10, 196)
        dA2 = np.dot(costFunc.T, self.W1)
        dA2 = dA2
        #(10, 196) -> (10, 784)
        dA2 = self.unpool(dA2, 2).reshape(-1, 784)


        #(10, 784) * (10, 784) = (10, 784)
        dZ2 = dA2 * self.deriv_relu(Z2)
        #sum (10, 784) -> (10, 1)
        dB2 = np.sum(dZ2, axis=1).reshape(10, 1)
        #(10, 784) * (784, 10) -> im2col -> (90, 784) * (784, 10) = (90, 10)
        temp_dZ2 = np.pad(dZ2.reshape(-1, 28, 28), ((0, 0), (1, 1), (1, 1)), 'constant')
        x_vector, y_vector = self.im2col(3, 1, 28)
        dF2 = np.dot(np.transpose(temp_dZ2[:, x_vector, y_vector], (0, 2, 1)).reshape(90, -1), A1.T).T


        #(90, 10) * (10, 784) = (90, 784)
        dA1 = F2.T.dot(dZ2)
        dA1 = dA1
        #(90, 784) -> (10, 784) by redistributing the 3x3 convolution
        dA1 = self.col2im(dA1)


        #(10, 784) * (10, 784) = (10, 784)
        dZ1 = dA1 * self.deriv_relu(Z1)
        #sum (10, 784) -> (10, 1)
        dB1 = np.sum(dZ1, axis=1).reshape(10, 1)
        #(10, 784) * (784) -> im2col -> (10, 784) * (784, 9) = (10, 9)
        temp_X = np.pad(X.reshape(28,28), (1, 1), 'constant')
        x_vector, y_vector = self.im2col(3, 1, 28)
        dF1 = dZ1.dot(temp_X[x_vector, y_vector])
        
        #Clipping the Gradient
        dW1, dF1, dF2, dB1, dB2, dB3 = self.clip_gradients([dW1, dF1, dF2, dB1, dB2, dB3], 5)

        self.W1 = np.subtract(self.W1, .7*dW1)
        self.F2 = np.subtract(self.F2, .7*dF2.reshape(-1, 10, 9).reshape(10, -1, 3, 3))
        self.F1 = np.subtract(self.F1, .7*dF1.reshape(10, 9).reshape(-1, 3, 3))

        self.B3 = np.subtract(self.B3, .7*dB3)
        self.B2 = np.subtract(self.B2, .7*dB2)
        self.B1 = np.subtract(self.B1, .7*dB1)

        
    def get_accuracy(self): 
        """Calculate Accuracy of predicted labels with true labels"""
        counter = 0
        for i in range(60000):
            X, F1, Z1, A1, F2, Z2, A2, input_1, output_1, activate_1  = self.forwardfeed(self.tr_x[i])
            if np.argmax(np.sum(activate_1, axis=1)) == self.tr_y[i]:
                counter += 1

        counter_2 = 0
        for i in range(10000):
            X, F1, Z1, A1, F2, Z2, A2, input_1, output_1, activate_1  = self.forwardfeed(self.test_x[i])
            if np.argmax(np.sum(activate_1, axis=1)) == self.test_y[i]:
                counter_2 += 1

        # print("Training Accuracy", counter/self.tr_y.size)
        # print("Validation Accuracy", counter_2/self.test_y.size)
        return (counter/self.tr_y.size, counter_2/self.test_y.size)

    def train(self):
        """Training using stochastic gradient descent using 10 epochs with 500 steps"""
        for j in range(10):
            np.random.seed(j*50)
            for i in np.random.choice(60000, 500, replace=False):
                X, F1, Z1, A1, F2, Z2, A2, input_1, output_1, activate_1  = self.forwardfeed(self.tr_x[i])
                costFunc = self.costFunction(activate_1, i)
                self.backpropagate(costFunc, X, F1, Z1, A1, F2, Z2, A2, input_1, output_1, activate_1)
        train_accuracy, valid_accuracy =  self.get_accuracy()
        return (self.W1, self.F1, self.F2, self.B1, self.B2, self.B3,  train_accuracy, valid_accuracy)


def loadFiles(tr_label, tr_img, test_label, test_img):
    """Unpacks MNIST dataset files"""

    tr_label_data, tr_img_data, test_label_data, test_img_data = 0, 0, 0, 0
    with open(tr_label, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        tr_label_data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))

    with open(tr_img, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        nrows, ncols = struct.unpack(">II", file.read(8))
        data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        tr_img_data = data.reshape((size, nrows, ncols))

    with open(test_label, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        test_label_data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))

    with open(test_img, "rb") as file:
        magic, size = struct.unpack(">II", file.read(8))
        nrows, ncols = struct.unpack(">II", file.read(8))
        data = np.fromfile(file, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_img_data = data.reshape((size, nrows, ncols))

    return(tr_label_data, tr_img_data, test_label_data, test_img_data)


if __name__ == "__main__":

    tr_label_data, tr_img_data, test_label_data, test_img_data = loadFiles("train-labels.idx1-ubyte",
                                                                           "train-images.idx3-ubyte",
                                                                           "t10k-labels.idx1-ubyte",
                                                                           "t10k-images.idx3-ubyte",
                                                                           )


    #No hyperparameter tuning due to time constraint from running on python, seriously, it takes several minutes per run

    #.94 Training and Validation Accuracy
    neural_network = MLP(tr_label_data, tr_img_data, test_label_data, test_img_data)
    W1, W2, B1, B2 = neural_network.train()
    train_accuracy, valid_accuracy = neural_network.predict()
    print("Training Accuracy:", train_accuracy)
    print("Valid Accuracy:", valid_accuracy)

    #.65 to .67 Training and Validation Accuracy(2 Convolution layers with only 10 Channels each)
    cnn = CNN(tr_img_data, tr_label_data, test_img_data, test_label_data)
    W1, F1, F2, B1, B2, B3, train_accuracy, valid_accuracy = cnn.train()
    print("Training Accuracy:", train_accuracy)
    print("Valid Accuracy", valid_accuracy)




