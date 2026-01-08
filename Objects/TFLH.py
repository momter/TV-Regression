import numpy as np
from Objects.FLH import FLH
from Objects.IFLH import IFLH

class TFLH: #thinned flh combining many flh classes

    def __init__(
        self, 
        m: int, 
        alpha: float, 
        black_box_class = None
    ):
        if m <= 0:
            raise ValueError("m must be positive")

        self.m = m
        self.alpha = alpha
        self.black_box_class = black_box_class

        
        self.streams = [
            FLH(alpha=alpha, black_box_class=black_box_class) #change to IFLH for faster performance
            for _ in range(m)
        ]

    def moving_average(self, a):
        b = []
        for i in range(len(a)):
            start = max(0, i - self.m)
            window = a[start:i+1]
            b.append(sum(window) / len(window))
        return b

    def run(self, y_sequence: np.ndarray, theta: np.ndarray):
        n = len(y_sequence)
        m = self.m

        stream_indices = [[] for _ in range(m)]
        for t in range(n):
            stream_indices[t % m].append(t)

        y_streams = [y_sequence[idxs] for idxs in stream_indices]
        theta_streams = [theta[idxs] for idxs in stream_indices]


        stream_preds = []
        stream_losses = []

        for k in range(m):
            preds_k, loss_k = self.streams[k].run(y_streams[k], theta_streams[k])
            stream_preds.append(preds_k)
            stream_losses.append(loss_k)

        preds = np.full(n, np.nan)
        losses = np.full(n, np.nan)

        for k in range(m):
            idxs = stream_indices[k]
            preds[idxs] = stream_preds[k]
            losses[idxs] = stream_losses[k]

        cum_loss = np.nancumsum(losses)

        avg_preds = self.moving_average(preds) #average last m predictions to reduce variance

        return preds, avg_preds, cum_loss