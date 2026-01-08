import numpy as np
from tqdm import tqdm

class RollAverage: #rolling average of n = lookback last values

    def __init__(self, lookback: int):
        if lookback <= 0:
            raise ValueError("Look back window parameter must be greater than 0.")
            
        self.lookback = lookback
        
        self._reset()

    def _reset(self):
        self.t = 0
        self.predictions = []
        self.y_history = []
        self.loss = []

    def run(self,
        y_sequence: list | np.ndarray, 
        theta: list | np.ndarray
    ) -> np.ndarray:

        self._reset()
        n = len(y_sequence)

        for t_idx in tqdm(range(n)):
            self.t += 1
            t = self.t
            y_t = y_sequence[t_idx]

            #predictions are only possible for t > 1
            if t > 1:
                mu_hat_t = sum(self.y_history[-self.lookback:]) / len(self.y_history[-self.lookback:])
                self.predictions.append(mu_hat_t)
                
            self.y_history.append(y_t)

            loss = np.nan

            if t > 1:
                loss = (mu_hat_t - theta[t_idx]) ** 2

            self.loss.append(loss)
            
        return np.array(self.predictions), np.array(self.loss)