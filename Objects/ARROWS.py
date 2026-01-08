import numpy as np
import pywt
from tqdm import tqdm

class ARROWS:
    """
    A class-based implementation of the ARROWS algorithm for online change point detection.

    This class maintains the state of the algorithm, allowing it to be used for both
    batch processing of an entire dataset or for step-by-step online detection.

    Args:
        n (int): The total expected length of the time series (time horizon).
        sigma (float): The standard deviation of the noise.
        beta (float): The learning parameter, must be > 24.
    """

    def __init__(self, n: int, sigma: float, beta: float):
        if beta <= 24:
            raise ValueError("Parameter 'beta' must be greater than 24.")
        if n <= 0:
            raise ValueError("Time horizon 'n' must be positive.")

        self.n = n
        self.sigma = sigma
        self.beta = beta

        #pre-calculate the threshold tau for efficiency
        self.tau = self.sigma * np.sqrt(self.beta * np.log(self.n))

        #initialize the algorithm's state
        self.reset()

    def reset(self):
        """Resets the algorithm's state to its initial values."""
        self.t_B = 0  #start time of the current bin (0-based index)
        self.Bin = 1  #flag: 1 if a new bin just started, 0 otherwise
        self.y_prev = np.nan  #previous observation y_{t-1}
        self.y_history = [] #stores all observed y so far

        #results
        self.detected_cps = []
        self.predictions = []
        self.total_loss = 0
        self.current_time = 0

    @staticmethod
    def _soft_threshold(x, threshold):
        """Performs element-wise soft thresholding on a numpy array."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)

    def _run_detection_test(self):
        """
        Performs the core change point detection test for the current window.
        
        Returns:
            bool: True if a change point is detected, False otherwise.
        """
        window_data = np.array(self.y_history[self.t_B:])
        window_len = len(window_data)

        if window_len <= 1:
            return False

        #pad the window to the next power of 2 (k) for the wavelet transform
        k = 1 << (window_len - 1).bit_length()
        
        #create the zero-padded, mean-centered vector y_tilde
        win_mean = np.mean(window_data)
        y_tilde_unpadded = window_data - win_mean
        y_tilde = np.pad(y_tilde_unpadded, (0, k - window_len), 'constant')

        #wavelet transform requires at least 2 points
        max_level = int(np.log2(k))
        if max_level < 1:
            return False

        #H * y_tilde: Perform the discrete wavelet transform
        coeffs = pywt.wavedec(y_tilde, 'haar', level=max_level)
        
        #T(H * y_tilde): Apply soft thresholding to all coefficients
        thresholded_coeffs = [ARROWS._soft_threshold(c, self.tau) for c in coeffs]
        
        #calculate the test statistic
        test_statistic = 0
        detail_coeffs_thresh = thresholded_coeffs[1:]
        
        for l, cD_thresh in enumerate(detail_coeffs_thresh):
            test_statistic += (2**(l / 2)) * np.linalg.norm(cD_thresh, ord=1)
        
        return test_statistic / np.sqrt(len(thresholded_coeffs)) > self.sigma / np.sqrt(self.n)

    def process_step(self, y_t: float):
        """
        Processes a single data point y_t, updates the state, and runs the test.

        Args:
            y_t (float): The new data point observed at the current time step.
        """
        if self.Bin == 1:
            prediction = self.y_prev
        else:
            prediction = np.mean(self.y_history[self.t_B:])
        
        self.predictions.append(prediction)


        self.total_loss += (prediction - y_t)**2
        self.y_history.append(y_t)
        self.y_prev = y_t
        self.Bin = 0

        if self._run_detection_test():
            self.Bin = 1
            self.t_B = self.current_time + 1
            self.detected_cps.append(self.t_B)

        self.current_time += 1

    def run(self, y: np.ndarray, theta: np.ndarray):
        """
        Runs the ARROWS algorithm on an entire dataset and returns the predictions.

        Args:
            y (np.ndarray): The full time series data.

        Returns:
            np.ndarray: An array containing the sequence of predictions.
        """
        if len(y) != self.n:
            raise ValueError(f"Input data length {len(y)} does not match "
                             f"the expected time horizon n={self.n}")
        
        self.reset()
        for y_t in tqdm(y):
            self.process_step(y_t)

        predictions_arr = np.array(self.predictions)
        
        #calculate the loss between prediction and the true parameter theta
        loss_vs_theta = (predictions_arr - theta)**2
        
        #calculate the cumulative sum, treating the initial nan as zero in the sum
        cumulative_loss_vs_theta = np.nancumsum(loss_vs_theta)
            
        return np.array(self.predictions), cumulative_loss_vs_theta