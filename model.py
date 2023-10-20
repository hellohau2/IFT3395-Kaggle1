import numpy as np

class LogisticRegression :
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self,z):
        return 1/(1 + (np.exp(-z)))
    
    def train(self,X,Y):

        n_samples, n_features = X.shape
        Y = np.array(Y.tolist())
        print(Y.shape)
        self.weights = np.random.randn(n_features, Y.shape[1])
        self.bias = 0

        #Grad Descent
        for epoch in range(self.epochs):
            sums = np.dot(X,self.weights) + self.bias
            predictions = self.sigmoid(sums)
            
            Y = np.array(Y.tolist())

            #gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - Y))
            db = (1/n_samples) * np.sum(predictions - Y)
            
            loss = (-1 / n_samples) * np.sum(Y * np.log(predictions) + (1 - Y) * np.log(1 - predictions))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            print(f"Epoch {epoch} : Loss {loss}")

    def predict(self, X):
        sum = np.dot(X,self.weights) + self.bias
        prediction = self.sigmoid(sum)

        return prediction

class Scaler:

    @staticmethod
    def minmax(df):
        scaled = df.copy()

        for col in df.columns:
            min = df[col].min()
            max = df[col].max()
            scaled[col] = (df[col] - min) / (max - min)
        
        return scaled