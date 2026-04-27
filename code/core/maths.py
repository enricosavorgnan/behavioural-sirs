"""
Some mathematical tools for the SIRS stability
"""
from code.core.models import SIRS

class RH:

    def __init__(self, target : str, equilibrium : list[int|float], model : SIRS):
        self.target = target
        self.equilibrium = equilibrium
        self.model = model

    def compute(self):
        """
        Compute the RH condition
        """
        pass

    def _get_beta(self):
        pass

    def _define_functions(self):
        pass



class RH_ThirdOrder(RH):
    def __init__(self, model: SIRS, target: str = 'a1', equilibrium=None):
        super().__init__(target, equilibrium, model)

        assert self.target in ['a1'], f"Currently the condition for target {self.target} is not available"
        assert len(self.equilibrium) == 3, f"The equilibrium should have length 3. Received {len(self.equilibrium)}"

        self.i = self.equilibrium[0]
        self.r = self.equilibrium[1]
        self.m = self.equilibrium[2]
        self.beta = self._get_beta()
        self.dot_beta = self._get_dot_beta()

        self._define_functions()


    def _get_beta(self):
        beta_func = self.model._physics.beta
        return beta_func(t=0., X=self.m)


    def _get_dot_beta(self):
        return - self.model.alpha1 / (1 + self.model.alpha1 * self.m) * self.model._physics.beta(t=0., X1=self.m)


    def _define_functions(self):
        pass


    def compute(self, x : float | int = 0.):
        """
        Computes the Routh Hurwitz condition for 3rd-order polynomials
        """
        pass




class RH_ForthOrder(RH):

    def __init__(self, model: SIRS, target: str = 'a1', equilibrium=None):
        super().__init__(target, equilibrium, model)

        assert self.target in ['a1'], f"Currently the condition for target {self.target} is not available"
        assert len(self.equilibrium) == 4, f"The equilibrium should have length 5. Received {len(self.equilibrium)}"

        self.non_target = self.model.a2 if self.target == 'a1' else self.model.a1
        self.i = self.equilibrium[0]
        self.r = self.equilibrium[1]
        self.s = 1 - self.i - self.r
        self.m1 = self.equilibrium[2]
        self.m2 = self.equilibrium[3]
        self.beta = self._get_beta()
        self.dot_beta = self._get_dot_beta()

        self._define_functions()


    def _get_beta(self):
        beta_func = self.model._physics.beta

        if self.model.model_type in ['sirs_two_layer_one_memory', 'sirs_two_layer_incidence_one_memory']:
            return beta_func(t=0., X=self.m2)
        elif self.model.model_type in ['sirs_two_layer']:
            return beta_func(t=0., X1=self.m1, X2=self.m2)
        return None


    def _get_dot_beta(self):
        return - self.model.alpha2 / (1 + self.model.alpha2 * self.m2) * self.model._physics.beta(t=0., X1=self.m2, X2=self.m1)


    def _define_functions(self):
        self._X = lambda x: x * self.non_target
        self._Y = lambda x: x + self.non_target
        self._A = self.beta * self.i + self.model.mu + self.model.theta
        self._B = self.beta * self.i * (self.model.mu + self.model.theta + self.model.gamma)
        self._C = self.beta * self.s + self.model.mu + self.model.theta
        self._D = self.beta * self.s * (self.model.mu + self.model.theta)
        self._E = - self.model.k1 * self.i * self.s * self.dot_beta

        self._q0 = lambda x: self._X * (self._B + self._D * self._E)
        self._q1 = lambda x: self._B * self._Y + self._A * self._X + self._X * self._C * self._E
        self._q2 = lambda x: self._X * self._E + self._X + self._B + self._A * self._Y
        self._q3 = lambda x: self._A + self._Y

        self.condition1 = lambda x: self._q3 * self._q2 - self._q1
        self.condition2 = lambda x: self._q3 * self._q2 * self._q1 - self._q1 **2 - self._q0 * self._q3 **2


    def compute(self, x : float | int = 0.):
        """
        Computes the two Routh Hurwitz conditions for 4th-order polynomials
        """
        return self.condition1(x), self.condition2(x)




class RH_FifthOrder(RH):

    def __init__(self, model: SIRS, target: str = 'a1', equilibrium=None):
        super().__init__(target, equilibrium, model)

        assert self.target in ['a1'], f"Currently the condition for target {self.target} is not available"
        assert len(self.equilibrium) == 5, f"The equilibrium should have length 5. Received {len(self.equilibrium)}"

        self.i = self.equilibrium[0]
        self.r = self.equilibrium[1]
        self.m1 = self.equilibrium[2]
        self.m2 = self.equilibrium[3]
        self.m3 = self.equilibrium[4]
        self.beta = self._get_beta()
        self.dot_beta = self._get_dot_beta()

        self._define_functions()


    def _define_functions(self):
        if self.model.model_type in ['sirs_three_layer', 'sirs_three_layer_incidence']:
            self._T = 1 - self.model.k1 * self.i * (1 - self.i - self.r) * self.dot_beta
        else:
            self._T = 1.

        if self.target == 'a1':
            self._B = self.model.mu + self.model.theta
            self._C = - self.beta * self.i
            self._D = self.model.a2 + self.model.a3
            self._E = self.model.a2 * self.model.a3
            self._A = (self.model.mu + self.model.gamma) - self._C

            self._q0 = lambda x: x * self._T * self._A * self._B * self._E
            self._q1 = lambda x: x * self._T  * (self._A * self._B * self._D + self._A * self._E + self._B * self._E) + self._A*self._B*self._E
            self._q2 = lambda x: x * self._T  * (self._A * self._B + self._A*self._D + self._B * self._D + self._E) + self._E*(self._B + self._A) + self._A*self._B*self._D
            self._q3 = lambda x: x * self._T  * (self._A + self._B + self._D) + self._A*(self._B + self._D) + self._B * self._D + self._E
            self._q4 = lambda x: x * self._T  + self._A + self._D + self._B

            self._condition1 = lambda x: self._q4(x)*self._q3(x) - self._q2(x)
            self._condition2 = lambda x: (self._q4(x) * self._q3(x) - self._q2(x)) * (self._q2(x) * self._q1(x) - self._q3(x) * self._q0(x)) - (self._q4(x) * self._q1(x) - self._q0(x))**2
            return

        else:
            return


    def _get_beta(self):
        """
        Retrieves the correct value of beta function
        """
        beta_func = self.model._physics.beta

        if self.model.model_type in ['sirs_three_layer_one_memory',
                                     'sirs_three_layer_incidence_one_memory']:
            return beta_func(t=0., X=self.m3)

        elif self.model.model_type in ['sirs_three_layer_two_memory',
                                       'sirs_three_layer_incidence_two_memory']:
            return beta_func(t = 0, X1=self.m2, X2=self.m3)

        elif self.model.model_type in ['sirs_three_layer']:
            return beta_func(t=0, X1=self.m1, X2=self.m2, X3=self.m3)

        return None


    def _get_dot_beta(self):
        return - self.model.alpha1 / (1 + self.model.alpha1 * self.m1) * self.model._physics.beta(t=0., X1=self.m1, X2=self.m2, X3=self.m3)


    def compute(self, x : float | int = 0.):
        """
        Computes the two Routh Hurwitz conditions for 5th-order polynomials
        """
        return self._condition1(x), self._condition2(x)

