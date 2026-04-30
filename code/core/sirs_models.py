from __future__ import annotations

import numpy as np
from code.core.beta_functions import BetaFunction

class SIRSModels:

    def __init__(self,
                 model_params : dict | None = None,
                 model_type: str | None = 'sirs',
                 r0: float | None = 0.,
                 gamma: float | None = 0.,
                 mu: float | None = 0.,
                 theta: float | None = 0.,
                 a1: float | None = 0.,
                 a2: float | None = 0.,
                 a3: float | None = 0.,
                 k1: float | None = 1.,
                 k2: float | None = 1.,
                 k3: float | None = 1.,
                 alpha1 : float | None = 1.,
                 alpha2 : float | None = 1.,
                 alpha3 : float | None = 1.,
                 delta: float | None = 0.,
                 omega: float | None = 2 * np.pi / 365,
                 T : int | None = 14):

        self.model_type = model_type if model_params is None else model_params.get('model_type', 'sirs')
        # Model must be valid
        assert self.model_type in ['sirs',
                                   'sirs_zero_layer',
                                   'sirs_zero_layer_incidence',
                                   'sirs_one_layer',
                                   'sirs_one_layer_incidence',
                                   'sirs_two_layer',
                                   'sirs_two_layer_incidence',
                                   'sirs_two_layer_one_memory',
                                   'sirs_two_layer_incidence_one_memory',
                                   'sirs_three_layer',
                                   'sirs_three_layer_incidence',
                                   'sirs_three_layer_one_memory',
                                   'sirs_three_layer_incidence_one_memory',
                                   'sirs_three_layer_two_memory',
                                   'sirs_three_layer_incidence_two_memory',
                                   'sirs_delay',
                                   'sirs_delay_incidence'
                                   ], f"Specified model type {self.model_type} is not valid."

        self.r0 = r0 if model_params is None else model_params.get('r0', 1.)
        self.gamma = gamma if model_params is None else model_params.get('gamma', 0.)
        self.mu = mu if model_params is None else model_params.get('mu', 0.)
        self.theta = theta if model_params is None else model_params.get('theta', 0.)
        self.a1 = a1 if model_params is None else model_params.get('a1', 0.)
        self.a2 = a2 if model_params is None else model_params.get('a2', 0.)
        self.a3 = a3 if model_params is None else model_params.get('a3', 0.)
        self.k1 = k1 if model_params is None else model_params.get('k1', 1.)
        self.k2 = k2 if model_params is None else model_params.get('k2', 1.)
        self.k3 = k3 if model_params is None else model_params.get('k3', 1.)
        self.alpha1 = alpha1 if model_params is None else model_params.get('alpha1', 1.)
        self.alpha2 = alpha2 if model_params is None else model_params.get('alpha2', 1.)
        self.alpha3 = alpha3 if model_params is None else model_params.get('alpha3', 1.)
        self.delta = delta if model_params is None else model_params.get('delta', 0.)
        self.omega = omega if model_params is None else model_params.get('omega', 2 * np.pi / 365)

        # Model must not have None parameters
        assert isinstance(self.r0, (int, float)), "R0 provided should be set for SIRS model."
        assert isinstance(self.gamma, (int, float)), "Gamma provided should be set for SIRS model."
        assert isinstance(self.mu, (int, float)), "Mu provided should be set for SIRS model."
        assert isinstance(self.theta, (int, float)), "Theta provided should be set for SIRS model."
        assert isinstance(self.a1, (int, float)), "A1 provided should be set for SIRS model."
        assert isinstance(self.a2, (int, float)), "A2 provided should be set for SIRS model."
        assert isinstance(self.a3, (int, float)), "A2 provided should be set for SIRS model."
        assert isinstance(self.k1, (int, float)), "K1 provided should be set for SIRS model."
        assert isinstance(self.k2, (int, float)), "K2 provided should be set for SIRS model."
        assert isinstance(self.k3, (int, float)), "K2 provided should be set for SIRS model."
        assert isinstance(self.alpha1, (int, float)), "Alpha1 provided should be set for SIRS model."
        assert isinstance(self.alpha2, (int, float)), "Alpha2 provided should be set for SIRS model."
        assert isinstance(self.alpha3, (int, float)), "Alpha2 provided should be set for SIRS model."
        assert isinstance(self.delta, (int, float)), "Delta provided should be set for SIRS model."
        assert isinstance(self.omega, (int, float)), "Omega provided should be set for SIRS model."

        if self.model_type in ['sirs_delay', 'sirs_delay_incidence']:
            self.T = int(T) if model_params is None else int(model_params.get('T', 14))
            assert isinstance(self.T, int), "T provided should be set for SIRS model."
            self.Is = []
            self.incidences = []

        self.beta_zero = self.r0 * (self.mu + self.gamma)
        self.beta = BetaFunction(model_type=self.model_type,
                                      beta_zero=self.beta_zero,
                                      alpha1=self.alpha1,
                                      alpha2= self.alpha2,
                                      alpha3=self.alpha3,
                                      delta=self.delta,
                                      omega=self.omega).set_beta_function()
        self.model = self.set_model()


    def set_model(self):
        models = {
            'sirs':                                 self.sirs,
            'sirs_zero_layer':                      self.sirs_zero_layer,
            'sirs_zero_layer_incidence':            self.sirs_zero_layer_incidence,
            'sirs_one_layer':                       self.sirs_one_layer,
            'sirs_one_layer_incidence':             self.sirs_one_layer_incidence,
            'sirs_two_layer':                       self.sirs_two_layer,
            'sirs_two_layer_incidence':             self.sirs_two_layer_incidence,
            'sirs_two_layer_one_memory':            self.sirs_two_layer_one_memory,
            'sirs_two_layer_incidence_one_memory':  self.sirs_two_layer_incidence_one_memory,
            'sirs_three_layer':                     self.sirs_three_layer,
            'sirs_three_layer_incidence':           self.sirs_three_layer_incidence,
            'sirs_three_layer_one_memory':          self.sirs_three_layer_one_memory,
            'sirs_three_layer_incidence_one_memory':self.sirs_three_layer_incidence_one_memory,
            'sirs_three_layer_two_memory':          self.sirs_three_layer_two_memory,
            'sirs_three_layer_incidence_two_memory':self.sirs_three_layer_incidence_two_memory,
            'sirs_delay':                           self.sirs_delay,
            'sirs_delay_incidence':                 self.sirs_delay_incidence
        }
        return models[self.model_type]


    def _manage_delay(self):
        if len(self.Is) > self.T:
            self.Is.pop(0)
        if len(self.incidences) > self.T:
            self.incidences.pop(0)


    # --- ODE Models ---

    def sirs(self, t, X):
        """ Standard SIRS model. Uses self.beta_zero(float) directly. """
        I, R = X

        beta_value = self.beta(t)

        dI = I * (beta_value * (1. - R - I) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R

        return [dI, dR]


    def sirs_zero_layer(self, t, X):
        I, R, M = X

        beta_value = self.beta(t, M)

        dI = I * (beta_value * (1. - I - R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = dI

        return [dI, dR, dM]


    def sirs_one_layer(self, t, X):
        I, R, M = X

        beta_value = self.beta(t, M)

        dI = I * (beta_value * (1. - I - R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = self.a1 * (I - M)

        return [dI, dR, dM]


    def sirs_two_layer(self, t, X):
        I, R, M1, M2 = X

        beta_value = self.beta(t, M1, M2)

        dI = I * (beta_value * (1. - I - R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (I - M1)
        dM2 = self.a2 * (M1 - M2)

        return [dI, dR, dM1, dM2]


    def sirs_three_layer(self, t, X):
        I, R, M1, M2, M3 = X

        beta_value = self.beta(t, M1, M2, M3)

        dI = I * (beta_value * (1. - I - R) - (self.mu + self.gamma))
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (I - M1)
        dM2 = self.a2 * (M1 - M2)
        dM3 = self.a3 * (M2 - M3)

        return [dI, dR, dM1, dM2, dM3]


    def sirs_zero_layer_incidence(self, t, X):
        I, R, M = X

        beta_value = self.beta(t, M)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = dI

        return [dI, dR, dM]


    def sirs_one_layer_incidence(self, t, X):
        I, R, M = X

        beta_value = self.beta(t, M)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = self.a1 * (incidence - M)

        return [dI, dR, dM]


    def sirs_two_layer_incidence(self, t, X):
        I, R, M1, M2 = X

        beta_value = self.beta(t, M1, M2)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)

        return [dI, dR, dM1, dM2]


    def sirs_three_layer_incidence(self, t, X):
        I, R, M1, M2, M3 = X

        beta_value = self.beta(t, M1, M2, M3)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)
        dM3 = self.a3 * (M2 - M3)

        return [dI, dR, dM1, dM2, dM3]


    def sirs_two_layer_one_memory(self, t, X):
        I, R, M1, M2 = X

        beta_value = self.beta(t, M2)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)

        return [dI, dR, dM1, dM2]


    def sirs_two_layer_incidence_one_memory(self, t, X):
        I, R, M1, M2 = X

        beta_value = self.beta(t, M2)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)

        return [dI, dR, dM1, dM2]


    def sirs_three_layer_one_memory(self, t, X):
        I, R, M1, M2, M3 = X

        beta_value = self.beta(t, M3)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (I - M1)
        dM2 = self.a2 * (M1 - M2)
        dM3 = self.a3 * (M2 - M3)

        return [dI, dR, dM1, dM2, dM3]


    def sirs_three_layer_incidence_one_memory(self, t, X):
        I, R, M1, M2, M3 = X

        beta_value = self.beta(t, M3)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)
        dM3 = self.a3 * (M2 - M3)

        return [dI, dR, dM1, dM2, dM3]


    def sirs_three_layer_two_memory(self, t, X):
        I, R, M1, M2, M3 = X

        beta_value = self.beta(t, M2, M3)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)
        dM3 = self.a3 * (M2 - M3)

        return [dI, dR, dM1, dM2, dM3]


    def sirs_three_layer_incidence_two_memory(self, t, X):
        I, R, M1, M2, M3 = X

        beta_value = self.beta(t, M2, M3)
        incidence = beta_value * I * (1. - R - I)

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM1 = self.a1 * (incidence - M1)
        dM2 = self.a2 * (M1 - M2)
        dM3 = self.a3 * (M2 - M3)

        return [dI, dR, dM1, dM2, dM3]


    def sirs_delay(self, t, X):
        I, R, M = X

        self.Is.append(I)

        beta_value = self.beta(t, M)
        incidence = beta_value * I * (1. - R - I)

        delay_I = self.Is[0]
        self._manage_delay()

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = (I - delay_I) / self.T

        return [dI, dR, dM]


    def sirs_delay_incidence(self, t, X):
        I, R, M = X

        beta_value = self.beta(t, M)
        incidence = beta_value * I * (1. - R - I)

        self.incidences.append(incidence)
        delay_incidence = self.incidences[0]
        self._manage_delay()

        dI = incidence - I * (self.mu + self.gamma)
        dR = self.gamma * I - (self.mu + self.theta) * R
        dM = (incidence - delay_incidence) / self.T

        return [dI, dR, dM]



if __name__ == '__main__':
    model = SIRSModels(
        model_type='sirs_delay',
        r0 = 2,
        mu= 1 / 80 / 365,
        theta= 1 / 365,
        gamma= 1 / 14,
        a1 = 1,
        a2 = 1,
        a3 = 1,
        k1 = 1,
        k2 = 1,
        k3 = 0.99,
        alpha1 = 1,
        alpha2 = 1,
        alpha3 = 1,
        delta = 4,
        omega = 2 * np.pi / 365,
        T = 7
    )

    result = model.model(t=5, X=[0.3, 0.1, 0.])
    print(result)