import numpy as np
from tqdm import tqdm

class FLH:
    
    class _DefaultBlackBox:
        """
        An efficient implementation for calculating the running average.
        """
        def __init__(self):
            #initialize sum and count for average calculation.
            self.sum_of_observations = 0.0
            self.num_observations = 0

        def predict(self) -> float:
            """
            Predicts the mean of observations in time.
            """
            if self.num_observations == 0:
                return 0.0
            
            return self.sum_of_observations / self.num_observations

        def observe(self, y: float):
            """
            Updates the running sum and count in time.
            """
            self.sum_of_observations += y
            self.num_observations += 1

    def __init__(self, alpha: float, black_box_class = None):
        if alpha <= 0:
            raise ValueError("Learning parameter alpha must be greater than 0.")
            
        self.alpha = alpha
        #use the improved _DefaultBlackBox by default
        self.black_box_class = black_box_class if black_box_class else self._DefaultBlackBox
        
        self._reset()

    def _reset(self):
        self.t = 0
        self.v = {}  #stores weights v_{t+1}^{(i)} for the next step
        self.v_hat_init = {}  #stores initial weights \widehat{v}_t^{(t)}
        self.experts = []
        self.predictions = []
        self.y_history = []
        self.loss = []

    def run(self, y_sequence: list | np.ndarray, theta: list | np.ndarray) -> np.ndarray:
        """
        Runs the algorithm on a sequence of observations.

        Args:
            y_sequence (list or np.ndarray): The sequence of observed outcomes y_t.

        Returns:
            np.ndarray: The sequence of predictions widehat{mu}_t made by the algorithm.
        """
        self._reset()
        n = len(y_sequence)

        for t_idx in tqdm(range(n)):
            self.t += 1
            t = self.t
            y_t = y_sequence[t_idx]

            #at the start of time step t, we use weights v_t.
            #in our implementation, self.v stores v_t from the previous step.
            v_t = self.v.copy()
            
            #step 1: Start new instance A_t (Line 4)
            v_t[t] = 1.0 / (t ** 2)
            self.v_hat_init[t] = 1.0 / (n)
            self.experts.append(self.black_box_class())

            #step 2: Generate prediction (Lines 5-8)
            mu_hat_t = np.nan
            expert_predictions = {}
            
            #predictions are only possible for t > 1
            if t > 1:
                #get predictions from experts A_1, ..., A_{t-1} (Line 7)
                for i in range(1, t):
                    expert_predictions[i] = self.experts[i-1].predict()

                #calculate prediction weights \widehat{v}_t^{(i)}
                v_hat_pred = {}
                
                #weight for expert A_{t-1} (Line 6, second part)
                v_hat_pred[t-1] = self.v_hat_init[t-1]

                #normalize weights for experts A_1, ..., A_{t-2} (Line 5)
                if t > 2:
                    #sum of weights for "old" experts {1, ..., t-2}
                    sum_v_old = sum(v_t.get(j, 0) for j in range(1, t - 1))
                    
                    if sum_v_old > 1e-12: #check for non-zero sum
                        norm_factor = (1.0 - 1.0 / (t - 1)) / sum_v_old
                        for i in range(1, t - 1):
                            v_hat_pred[i] = norm_factor * v_t.get(i, 0)
                    else: #fallback for zero sum: uniform distribution
                        num_old_experts = t - 2
                        if num_old_experts > 0:
                            uniform_weight = (1.0 - 1.0 / ((t - 1) ** 2)) / num_old_experts
                            for i in range(1, t - 1):
                                v_hat_pred[i] = uniform_weight
                    self.v_hat_pred = v_hat_pred

                #predict \widehat{\mu}_t (Line 8)
                mu_hat_t = sum(v_hat_pred.get(i, 0) * expert_predictions.get(i, 0) for i in range(1, t))
            
            self.predictions.append(mu_hat_t)
            self.y_history.append(y_t)

            #step 3: all experts observe y_t
            #experts A_1, ..., A_t have been created and observe y_t
            for expert in self.experts:
                expert.observe(y_t)

            #step 4: update weights for next step, v_{t+1} (Lines 10-12)
            v_next = {}
            #update for expert t (Line 10)
            v_next[t] = v_t[t]

            #update for experts 1, ..., t-1 (Line 12)
            loss = np.nan

            if t > 1:
                for i in range(1, t):
                    squared_loss = (y_t - expert_predictions[i])**2
                    v_next[i] = v_hat_pred.get(i, 0) * np.exp(-self.alpha * squared_loss)
                loss = (mu_hat_t - theta[t_idx]) ** 2
                
            #store v_{t+1} for the next iteration
            self.v = v_next
            self.loss.append(loss)
            
            
        return np.array(self.predictions), np.array(self.loss)