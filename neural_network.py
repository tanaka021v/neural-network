import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'] / 255.0, mnist['target']

# Y = (np.arange(10) == y).astype(int).reshape(-1, 1)
 
class Neural_Network():
    def __init__(self):
        n, m = 784, 16

        self.weights = [
                    None, 
                    np.random.randn(m, n) * np.sqrt(2.0 / n),    # Layer 1 (ReLU): He-Init mit 784 Inputs
                    np.random.randn(m, m) * np.sqrt(2.0 / m),    # Layer 2 (ReLU): He-Init mit 16 Inputs
                    np.random.randn(10, m) * np.sqrt(1.0 / m)    # Output (Softmax): Xavier-Init mit 16 Inputs
                    ]
        self.bias = [
                    None, 
                    np.zeros((m, 1)), 
                    np.zeros((m, 1)), 
                    np.zeros((10, 1))
                ]
        self.z = [None]
        self.Kw = []
        self.Kb = []
        self.size = len(self.bias)
        self.learning_rate = 0.01
        
    def relu(self, x):
    # Vergleicht jedes Element in x mit 0 und nimmt das Maximum
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
    # 1 für alle Werte > 0, sonst 0
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        # Numerische Stabilität: Maximum abziehen verhindert Überlauf (Overflow) bei np.exp()
        # Mathematisch ändert das das Ergebnis der Softmax-Funktion nicht.
        
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def activation_function_end(self, x):
        return self.softmax(x)
    def activation_function_hidden(self, x):
        return self.relu(x)

    def categorical_crossentropy(self, y_pred, y_true):
        # Numerische Stabilität: np.clip verhindert den Fehler log(0), 
        # indem exakte 0-Werte auf eine extrem kleine Zahl (1e-15) gesetzt werden.
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
        # Berechnet: -Summe(y_true * log(y_pred))
        return -np.sum(y_true * np.log(y_pred))
    
    def get_max_index(self, vector):
        return np.argmax(vector)
    
    def lost_function(self, input, label): 
        return self.categorical_crossentropy(input, label)

    def forward_pass(self, input): 
        self.z = [None]
        # 1. Liste in einen 3D-Array konvertieren
        arr = np.array(input)
        # 2. Matrizen flachdrücken und direkt transponieren
        self.activation = [arr.reshape(arr.shape[0], -1).T]

        # Hiddenlayers verwende ReLU Funktion
        for L in range(1, self.size-1): 
            self.z.append(self.weights[L] @ self.activation[L-1] + self.bias[L])
            self.activation.append(
                self.activation_function_hidden(self.z[L]))
        #End layer mit Softmax Funktion
        self.z.append(self.weights[-1] @ self.activation[-1] + self.bias[-1])
        self.activation.append(
                self.activation_function_end(self.z[-1]))

    def backpropagation(self, y, split): 
        batch_size = len(y) # Dynamisch statt fest 10
        zeilen = np.array(y, dtype=int)
        
        # 10 Klassen (Zeilen), batch_size Bilder (Spalten)
        Y = np.zeros((10, batch_size), dtype=int)
        spalten = np.arange(batch_size)
        Y[zeilen, spalten] = 1

        # b_s = self.sigmoid_derivative(self.z[-1]) * self.lost_function(self.activation[-1], Y)
        # Als Aktivierungsfunktion wird die Softmax FUnktion verwendet mit der Lost Funktion Categorical Cross-Entropy (CCE)
        c_b_s = 1/split * (self.activation[-1] - Y)
        weight_s = c_b_s @ self.activation[-2].T
        Gb_s = [np.sum(c_b_s, axis=1, keepdims=True)]
        Gw_s = [weight_s]
        for L in range(self.size - 2, 0, -1):
            c_b_s = self.relu_derivative(self.z[L]) * (self.weights[L+1].T @ c_b_s ) 
            weight_s = c_b_s @ self.activation[L-1].T
            Gb_s.insert(0,  np.sum(c_b_s, axis=1, keepdims=True))
            Gw_s.insert(0, weight_s)
        return Gb_s, Gw_s

    def process(self, input, y): 
        split = 10
        epochs = 50
        for epoch in range(epochs):
            accuracy = 0
            count = 0
            for size in range(0, 10000, 10): 
                self.forward_pass(input[size:size+10])
                count +=10
                vorhersagen = np.argmax(self.activation[-1], axis=0)
                accuracy += np.sum(vorhersagen == np.array(y[size:size+10], dtype=int))
                Gb_s, Gw_s = self.backpropagation(y[size:size+10], split)
                for L in range(1, self.size):
                    self.weights[L] -= self.learning_rate * Gw_s[L-1]
                    self.bias[L] -= self.learning_rate * Gb_s[L-1]

        print("Gewichte:", self.weights)
        print("Biases:", self.bias)
        print("Letzt aktivierung:", self.activation[-1])
        print(accuracy/count)
            

# Dimensionen: n=784 (Inputs), m=16 (Hidden)


neuronal = Neural_Network()
neuronal.process(X, y)




        