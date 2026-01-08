import numpy as np
from scipy.stats import norm

class SyntheticData:
    def __init__(self, 
                 n : int, 
                 parameter_scale = 1,
                 sample_mean = 0, 
                 sample_var = 1,
                 theta_type : str = 'hard_shift_random',
                 segment_size : int = 100,
                 segment_func : bool = False,
                 rho : float = 2,
                 eta : float = 0.9,
                 decay_factor = 0.5
    ):

        self.length = n
        self.sample_mean = sample_mean
        self.sample_var = sample_var
        self.theta_type = theta_type

        self.segment_size = segment_size
        self.func = segment_func
        self.rho = rho
        self.eta = eta
        self.decay_factor = decay_factor

        self.errors = np.zeros(self.length)
        self.theta = np.zeros(self.length)

        self.scale = parameter_scale
        

    def normal(self):
        norm_list = np.array(norm.rvs(loc = self.sample_mean, 
                                      scale = self.sample_var, 
                                      size = self.length
                        )
                    )
        self.norm_list = norm_list

    
    def dependent_errors_ARMA22( #generates dependent errors according to an ARMA(2,2) model
        self,
        phi1 = 0.5, 
        phi2 = -0.2,
        theta1 = 0.4, 
        theta2 = 0.3,
        innovation_var = 1.0
        ):

        
        self.errors = np.zeros(self.length)
        innovations = np.random.normal(
            loc = 0.0,
            scale = np.sqrt(innovation_var),
            size = self.length
        )

        
        e_tm1 = 0.0
        e_tm2 = 0.0
        eps_tm1 = 0.0
        eps_tm2 = 0.0

        for t in range(self.length):

            #ARMA(2,2) recursion
            eps_t = (phi1 * eps_tm1 +
                    phi2 * eps_tm2 +
                    theta1 * e_tm1 +
                    theta2 * e_tm2 +
                    innovations[t])

            self.errors[t] = eps_t

            #shift lags
            eps_tm2 = eps_tm1
            eps_tm1 = eps_t

            e_tm2 = e_tm1
            e_tm1 = innovations[t]

    def hard_shift( #generates parameter with hard shifts
        self, 
        segment_growth_func = None
    ):
        self.theta = np.zeros(self.length)
        boundaries = []

        if self.func:
            segment_growth_func = lambda i: 2**i

        if callable(segment_growth_func):
            current_pos = 0
            i = 0
            boundaries.append(current_pos)
            while current_pos < self.length:
                segment_len = int(segment_growth_func(i))
                
                if segment_len < 1:
                    raise ValueError("segment_growth_func must return values >= 1.")
                
                current_pos += segment_len
                boundaries.append(min(current_pos, self.length))
                i += 1
        else:
            boundaries = list(range(0, self.length, self.segment_size))
            if not boundaries or boundaries[-1] < self.length:
                boundaries.append(self.length)
        
        self.phi = []

        current_val = 0.0
        for i, start_idx in enumerate(sorted(list(set(boundaries[:-1])))):
            end_idx = sorted(list(set(boundaries)))[i+1]
            
            if start_idx >= end_idx:
                continue
                
            self.theta[start_idx:end_idx] = current_val
            
            phi = np.random.normal(loc=0, scale=self.rho)
            self.phi.append(phi)
            current_val += phi

    
    def hard_shift_plot( #same as previous, however the first shift always occurs at n = first_segment
        self, 
        segment_growth_func = None,
        first_segment : int = 100
    ):
        self.theta = np.zeros(self.length)
        boundaries = []

        if self.func:
            segment_growth_func = lambda i: 2**i

        if callable(segment_growth_func):
            current_pos = 0
            i = 0
            boundaries.append(current_pos)
            while current_pos < self.length:
                segment_len = int(segment_growth_func(i))
                
                if segment_len < 1:
                    raise ValueError("segment_growth_func must return values >= 1.")
                
                current_pos += segment_len
                boundaries.append(min(current_pos, self.length))
                i += 1
        else:
            boundaries = list(range(0, self.length, self.segment_size))
            
            # FIX: If segment_size > 100, ensure a boundary at index 100.
            if self.segment_size > first_segment:
                if first_segment < self.length:
                    boundaries.append(first_segment)
                # Sort and remove duplicates to maintain order.
                boundaries = sorted(list(set(boundaries)))

            # Ensure the last boundary is the total length.
            if not boundaries or boundaries[-1] < self.length:
                boundaries.append(self.length)
        
        self.phi = []

        current_val = 0.0
        # Use sorted list of unique boundaries to create segments.
        unique_boundaries = sorted(list(set(boundaries)))
        for i, start_idx in enumerate(unique_boundaries[:-1]):
            end_idx = unique_boundaries[i+1]
            
            if start_idx >= end_idx:
                continue
                
            self.theta[start_idx:end_idx] = current_val
            
            phi = np.random.normal(loc=0, scale=self.rho)
            self.phi.append(phi)
            current_val += phi


    def soft_shift(self): #generate parameter according to soft shifts
        self.theta = np.zeros(self.length)
        self.zeta = []
        
        for t in range(1, self.length):
            variance = (t + 1)**(-self.eta)
            zeta = np.random.normal(loc=0, scale=np.sqrt(variance))
            self.zeta.append(zeta)
            self.theta[t] = self.theta[t-1] + zeta


    def run(self):
        self.normal()
        self.dependent_errors_ARMA22()

        if self.theta_type == 'hard_shift':
            self.hard_shift()
        elif self.theta_type == 'hard_shift_2':
            self.hard_shift_plot()
        else:
            self.soft_shift()
        
        abs_diffs = np.abs(np.diff(self.theta))        
        self.cumulative_TV = np.insert(np.cumsum(abs_diffs), 0, 0)
        self.bound = (self.cumulative_TV ** (2/3)) * (np.arange(len(self.cumulative_TV)) + 1) ** (1/3)

        self.theta = self.scale * self.theta

        self.ind = self.theta + self.norm_list
        self.dep = self.theta + self.errors