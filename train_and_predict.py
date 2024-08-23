from forward_prop import ForwardProp
from back_prop import BackProp
import numpy as np

class FitAndPredict(ForwardProp,BackProp):
    def fit(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 10000, threshold: float = 1e-9):
        if len(X) != len(y) or len(X[0]) != len(self.input_bias):
            raise ValueError("Incorrect dimensions between X and y or no. of input neurons")

        old_loss = np.inf
        epoch = 0

        while epoch < max_epochs:
            epoch += 1
            print(f"Epoch {epoch}")
            
            predictions = []
            total_loss = 0
            
            # Forward pass and loss calculation
            for i in range(len(X)):
                y_pred = self.forward_prop(X[i])
                predictions.append(y_pred)
                
                loss = (y_pred - y[i]) ** 2
                total_loss += loss
            
            total_loss /= (2 * len(X))  # Mean Squared Error
            total_loss = np.sum(total_loss)  # Ensure it's scalar

            print(f"Loss: {total_loss}")


            # Check for convergence
            if np.abs(total_loss - old_loss) < threshold:
                break
            old_loss = total_loss
            
            # Backward pass
            for index, (y_pred, y_true) in enumerate(zip(predictions, y)):
                self.back_prop(output=y_pred, target=y_true, inp=X[index])
            
            if np.isnan(total_loss):
                raise ValueError("Loss is NaN")
        
        self.save_params()
        print("Updated values saved\n")

    def predict(self, X):
        if len(X[0]) != len(self.input_bias):
            raise ValueError("Incorrect dimensions between input data and number of input neurons")
        
        output = [self.forward_prop(x) for x in X]

        return np.array(output)
