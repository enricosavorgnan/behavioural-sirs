# Several Implementations for Behavioural-dependent Beta Functions
import numpy as np

class BetaFunction:
    """
    Several implementations of β() functions
    """
    def __init__(self,
                 model_type : str = 'sirs' ,
                 beta_zero : float = 0.,
                 alpha1 : float = 0.,
                 alpha2 : float = 0.,
                 delta : float = 0.,
                 omega : float = 0.
                 ):

        self.model_type : str = model_type
        self.beta_zero = beta_zero
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.delta = delta
        self.omega = omega


    def set_beta_function(self):
        if self.model_type in ['sirs'] :
            return self.beta_const
        elif self.model_type in ['sirs_zero_layer', 'sirs_zero_layer_incidence', 'sirs_one_layer', 'sirs_one_layer_incidence', 'sirs_two_layer_one_memory', 'sirs_two_layer_incidence_one_memory']:
            return self.beta_one
        elif self.model_type in ['sirs_two_layer', 'sirs_two_layer_incidence']:
            return self.beta_two
        else:
            raise ValueError(f"Specified model type {self.model_type} is not valid.")


    def beta_const(self, t : float | None = None , *args):
        if self.delta != 0:
            dumping = 1 + self.delta * np.cos(self.omega * t)
            return dumping * self.beta_zero
        return self.beta_zero


    def beta_one(self, t, X, *args):
        infection_rate = self.beta_zero / (1 + self.alpha1 * X)
        if self.delta != 0:
            dumping = 1 + self.delta * np.cos(self.omega * t)
            return infection_rate * dumping
        return infection_rate


    def beta_two(self, t, X1, X2):
        infection_rate = self.beta_zero / (1 + self.alpha1 * X1) / (1 + self.alpha2 * X2)
        if self.delta != 0:
            dumping = 1 + self.delta * np.cos(self.omega * t)
            return infection_rate * dumping
        return infection_rate