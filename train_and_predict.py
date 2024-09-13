from forward_prop import ForwardProp
from back_prop import BackProp
import numpy as np

class FitAndPredict(ForwardProp,BackProp):
    def fit(self, X, y, max_epochs = 1000, threshold = 1e-9):
        if len(X) != len(y):
            raise ValueError("len(X) != len(y)")

        old_loss = np.inf
        epoch = 0

        while epoch < max_epochs:
            epoch += 1
            print(f"Epoch {epoch}")
            
            predictions = []
            
            for i in range(len(X)):
                y_pred = self.forward_prop(X[i])
                self.back_prop(input_ = X[i], output= y_pred, target=y[i])
                predictions.append(y_pred)
                
            total_loss = self.loss_func(y, np.array(predictions))

            print(f"Loss: {total_loss}")

            if np.isnan(total_loss):
                raise ValueError("Loss is NaN")
            
            if old_loss - total_loss < threshold:
                break
            old_loss = total_loss
        
        self.save_params()
        print("Updated values saved\n")

    def predict(self, X):
        if isinstance(X[0], np.ndarray):
            if len(X[0]) != len(self.input_weights):
                print(len(X[0]),len(self.input_weights))
                raise ValueError("Incorrect dimensions between input data and number of input neurons")
            return np.array([self.forward_prop(x) for x in X])
        if len(X)!=len(self.input_weights):
            raise ValueError("Incorrect dimensions between input data and number of input neurons")
        return self.forward_prop(X)
