import numpy as np
from tqdm import tqdm

class IFLH: #FLH code with stopping time functionality

    class _DefaultBlackBox:
        """
        An efficient O(1) implementation for calculating the running average.
        """
        def __init__(self):
            #initialize sum and count for O(1) average calculation.
            self.sum_of_observations = 0.0
            self.num_observations = 0

        def predict(self) -> float:
            """
            Predicts the mean of observations in O(1) time.
            """
            if self.num_observations == 0:
                return 0.0
            
            return self.sum_of_observations / self.num_observations

        def observe(self, y: float):
            """
            Updates the running sum and count in O(1) time.
            """
            self.sum_of_observations += y
            self.num_observations += 1

    def __init__(self, alpha: float, K_base=2, black_box_class=None):
        if alpha <= 0:
            raise ValueError("Learning parameter alpha must be greater than 0.")
            
        self.alpha = alpha
        self.K = K_base
        #use the improved _DefaultBlackBox by default
        self.black_box_class = black_box_class if black_box_class else self._DefaultBlackBox
        
        self._reset()

    def _reset(self):
        """Resets the internal state of the algorithm."""
        self.t = 0
        self.v = {}  #stores weights v_{t+1}^{(i)} for the next step
        self.v_hat_init = {}  #stores initial weights \widehat{v}_t^{(t)}
        self.tau = {} #stores lifetime tau_i for each expert i
        self.experts = []
        self.predictions = []
        self.y_history = []
        self.len_experts = []
        self.loss = []

    def _calculate_tau(self, t: int) -> int:
        """
        Calculates tau_t = t + 2^k, where 2^k is the least significant bit of t.
        This can be computed efficiently using the bitwise operation: t + (t & -t).
        """
        if t == 0:
            return 1  # t = 0 has no 1s in binary; by convention we return 1
        
        #find least significant bit (LSB) position that is set to 1
        k = (t & -t).bit_length() - 1  #trick to find index of LSB 1
        result = t + 2 ** k
        return result
    
    def _calculate_tau_K(self, t: int) -> int:
        if t == 0:
            return self.K #default for t=0

        beta = []
        temp_t = t
        while temp_t > 0:
            beta.append(temp_t % self.K)
            temp_t //= self.K
        
        #find smallest k such that beta[k] > 0
        k = -1
        for i in range(len(beta)):
            if beta[i] > 0:
                k = i
                break
        
        if k == -1: #should not happen if t > 0
            return self.K
        
        #zero out beta[k]
        beta[k] = 0

        #ensure beta has at least k+2 digits to accommodate K^{k+1}
        while len(beta) <= k + 1:
            beta.append(0)

        #add 1 to beta[k+1]
        beta[k + 1] += 1

        #handle carry-over if necessary
        for i in range(k + 1, len(beta)):
            if beta[i] >= self.K:
                beta[i] -= self.K
                if i + 1 >= len(beta):
                    beta.append(1)
                else:
                    beta[i + 1] += 1
            else:
                break

        #convert back to integer from base-K
        result = sum(d * (self.K ** i) for i, d in enumerate(beta))
        return result

    def run(self, y_sequence: list | np.ndarray, theta: list | np.ndarray) -> np.ndarray:
        self._reset()
        n = len(y_sequence)

        for t_idx in tqdm(range(n)):
            self.t += 1
            t = self.t
            y_t = y_sequence[t_idx]

            #at the start of time step t, we use weights v_t.
            v_t = self.v.copy()
            
            #step 1: start new instance A_t and assign weights (Line 4)
            v_t[t] = 1.0 / t
            self.v_hat_init[t] = 1.0 / t
            self.experts.append(self.black_box_class())
            
            #step 2: define tau_t and the active expert set S_t (Lines 5-6)
            self.tau[t] = self._calculate_tau_K(t)
            active_experts_S_t = {i for i in range(1, t) if self.tau.get(i, 0) > t - 1}
            self.len_experts.append(len(active_experts_S_t))

            #step 3: generate prediction
            mu_hat_t = np.nan
            expert_predictions = {}
            
            if t > 1 and active_experts_S_t:
                #get predictions from active experts A_i, i in S_t (Line 9)
                for i in active_experts_S_t:
                    expert_predictions[i] = self.experts[i-1].predict()

                #calculate prediction weights \widehat{v}_t^{(i)} for i in S_t
                v_hat_pred = {}
                sum_v_active = sum(v_t.get(j, 0) for j in active_experts_S_t)

                if sum_v_active > 1e-12:
                    for i in active_experts_S_t:
                        v_hat_pred[i] = v_t.get(i, 0) / sum_v_active
                else:  #fallback for zero sum: uniform distribution
                    uniform_weight = 1.0 / len(active_experts_S_t)
                    for i in active_experts_S_t:
                        v_hat_pred[i] = uniform_weight
                
                #predict \widehat{\mu}_t by summing over active experts S_t
                mu_hat_t = sum(v_hat_pred.get(i, 0) * expert_predictions.get(i, 0) for i in active_experts_S_t)

            self.predictions.append(mu_hat_t)
            self.y_history.append(y_t)

            #step 4: all experts (1..t) observe y_t
            for expert in self.experts:
                expert.observe(y_t)

            #step 5: update weights for next step, v_{t+1} (Lines 12-13)
            v_next = {}
            #set weight for the new expert t (Line 12)
            v_next[t] = v_t.get(t, 0)

            loss = np.nan
            if t > 1:
                #update weights for active experts based on loss
                for i in active_experts_S_t:
                    squared_loss = (y_t - expert_predictions[i]) ** 2
                    v_next[i] = v_hat_pred.get(i, 0) * np.exp(-self.alpha * squared_loss)
                if not np.isnan(mu_hat_t):
                    loss = (mu_hat_t - theta[t_idx]) ** 2
            
            self.loss.append(loss)
            
            #store v_{t+1} for the next iteration
            self.v = v_next
            
        return np.array(self.predictions), np.array(np.array(self.loss))