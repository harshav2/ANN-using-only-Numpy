from forward_prop import ForwardProp
from back_prop import BackProp
import numpy as np

class GradientDescent(ForwardProp,BackProp):
    def fit(self, X, y, max_epochs=10, threshold=1e-5):
        if len(X) != len(y) or len(X[0]) != len(self.input_bias):
            raise ValueError("Incorrect dimensions between X and y or input_bias")
    
        total_loss = np.inf
        epoch=0

        while total_loss >= threshold and epoch<=max_epochs:
            epoch+=1
            print(f"Epoch {epoch}")
            total_loss = 0
            predictions = []
            
            for i in range(len(X)):
                y_pred = self.forward_prop(X[i])
                predictions.append(y_pred)
                
                loss = (y_pred - y[i]) ** 2
                total_loss += loss
         
            total_loss /= (2 * len(X))
            total_loss=np.sum(total_loss)
            
            for index, y_pred, y_true in zip(range(len(predictions)), predictions, y):
                self.back_prop(output=y_pred, target=y_true, inp=X[index])
            
            if np.isnan(total_loss): 
                raise ValueError("Loss is NaN")
        
        self.save_params()
        print("Updated values saved\n")