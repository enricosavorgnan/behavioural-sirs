"""
Some mathematical tools for the SIRS stability
"""
from code.core.models import SIRS

class RH:

    def __init__(self, target : str | list[str], equilibrium : list[int|float], model : SIRS):
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
    def __init__(self, model: SIRS, target: str | list[str] = 'a1', equilibrium=None):
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

    def __init__(self, model: SIRS, target: str | list[str] = 'a1', equilibrium=None):
        super().__init__(target, equilibrium, model)

        # assert self.target in ['a1', 'a2'], f"Currently the condition for target {self.target} is not available"
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
        return - self.model.alpha2 / (1 + self.model.alpha2 * self.m2) * self.model._physics.beta(t=0., X=self.m2)


    def _define_functions(self):
        self._A = self.beta * self.i + self.model.mu + self.model.theta
        self._B = self.beta * self.i * (self.model.mu + self.model.theta + self.model.gamma)
        self._C = self.beta * self.s + self.model.mu + self.model.theta
        self._D = self.beta * self.s * (self.model.mu + self.model.theta)
        self._E = - self.model.k1 * self.i * self.s * self.dot_beta

        if self.target == 'a1' or self.target == 'a2':
            self._X = lambda x: x * self.non_target
            self._Y = lambda x: x + self.non_target

            self._q0 = lambda x: self._X(x) * (self._B + self._D * self._E)
            self._q1 = lambda x: self._B * self._Y(x) + self._A * self._X(x) + self._X(x) * self._C * self._E
            self._q2 = lambda x: self._X(x) * self._E + self._X(x) + self._B + self._A * self._Y(x)
            self._q3 = lambda x: self._A + self._Y(x)

            self.condition1 = lambda x: self._q3(x) * self._q2(x) - self._q1(x)
            self.condition2 = lambda x: self._q3(x) * self._q2(x) * self._q1(x) - self._q1(x) **2 - self._q0(x) * self._q3(x) **2

        else:
            self._X = lambda x,y: x * y
            self._Y = lambda x,y: x + y

            self._q0 = lambda x,y: self._X(x, y) * (self._B + self._D * self._E)
            self._q1 = lambda x,y: self._B * self._Y(x, y) + self._A * self._X(x, y) + self._X(x, y) * self._C * self._E
            self._q2 = lambda x,y: self._X(x, y) * self._E + self._X(x, y) + self._B + self._A * self._Y(x, y)
            self._q3 = lambda x,y: self._A + self._Y(x, y)

            self.condition1 = lambda x,y: self._q3(x,y) * self._q2(x,y) - self._q1(x,y)
            self.condition2 = lambda x,y: self._q3(x,y) * self._q2(x,y) * self._q1(x,y) - self._q1(x,y) **2 - self._q0(x,y) * self._q3(x,y) **2

            self._z1 = lambda x,y: self._A * self._Y(x,y) * (self._X(x,y) - self._B)**2
            self._z2 = lambda x,y: -self._D * self._E * self._X(x, y) * (self._A + self._Y(x,y))**2 + self._C * self._E * self._X(x, y) * ((self._X(x,y) - self._B) * (self._Y(x,y) - self._A) - self._C * self._E) + (self._A + self._Y(x,y)) * (self._A * self._X(x,y) + self._B * self._Y(x,y)) * (self._E * self._X(x,y) + self._A * self._Y(x,y))
            self._z3 = lambda x,y: self._C * self._E * self._X(x, y) * ((self._E * self._X(x,y) + self._A * self._Y(x,y)) * (self._A + self._Y(x,y)))


    def compute(self, x : float | int = 0., y : float | int | None = None, **kwargs):
        """
        Computes the two Routh Hurwitz conditions for 4th-order polynomials
        """

        verb = kwargs.get('verbose', False)
        if verb:
            print(f"A: {round(self._A, 4)}\t\t\t\tB: {round(self._B, 4)}\t\t\t\tC: {round(self._C, 4)}\t\t\t\tD: {round(self._D, 4)}\t\t\t\tE: {round(self._E, 4)}")
            print(f"Beta: {round(self.beta, 4)}\t\t\tDot Beta: {round(self.dot_beta, 4)}")

        if self.target == 'a1' or self.target == 'a2':
            if verb:
                print(f"X: {round(self._X(x), 4)}\t\t\t\tY: {round(self._Y(x), 4)}")
            return self.condition1(x), self.condition2(x)
        else:
            assert y is not None, "For target a1 and a2, only one variable is needed. For other targets, two variables are needed."
            if verb:
                print(f"X: {round(self._X(x,y), 8)}\t\t\t\t\tY: {round(self._Y(x,y), 8)}")
                print(f"Z1: {round(self._z1(x,y), 15)}\t\t\t\t\tZ2: {round(self._z2(x,y), 15)}\t\t\t\tZ3: {round(self._z3(x,y), 15)}")
                print(f"q0: {round(self._q0(x,y), 12)}\t\t\t\t\tq1: {round(self._q1(x,y), 12)}\t\t\tq2: {round(self._q2(x,y), 12)}\t\t\t3: {round(self._q3(x,y), 12)}")
                print(f"Cond1: {round(self.condition1(x,y), 8)}\t\t\tCond2: {self.condition2(x,y)}\n\n")
            return self.condition1(x, y), self.condition2(x, y)



class RH_FifthOrder(RH):

    def __init__(self, model: SIRS, target: str | list[str] = 'a1', equilibrium=None):
        super().__init__(target, equilibrium, model)

        # assert self.target in ['a1', 'a2', 'a3'], f"Currently the condition for target {self.target} is not available"
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
        self._A = self.beta * self.i + self.model.mu + self.model.theta
        self._B = self.beta * self.i * (self.model.mu + self.model.theta + self.model.gamma)
        self._C = self.beta * (1 - self.i - self.r) + self.model.mu + self.model.theta
        self._D = self.beta * (1 - self.i - self.r) * (self.model.mu + self.model.theta)
        self._E = - self.model.k1 * self.i * (1 - self.i - self.r) * self.dot_beta

        if self.target == 'a1':
            # TODO
            pass
        elif self.target == 'a2':
            # TODO
            pass
        elif self.target == 'a3':
            # TODO
            pass

        else:
            self._X = lambda x,y,z: x * y * z
            self._Y = lambda x,y,z: x + y + z
            self._Z = lambda x,y,z: x * y + x * z + y * z

            self._q0 = lambda x,y,z: self._X(x,y,z) * (self._B + self._D * self._E)
            self._q1 = lambda x,y,z: self._B * self._Z(x,y,z) + self._A * self._X(x,y,z) + self._X(x,y,z) * self._C * self._E
            self._q2 = lambda x,y,z: self._X(x,y,z) * self._E + self._X(x,y,z) + self._B * self._Y(x,y,z) + self._A * self._Z(x,y,z)
            self._q3 = lambda x,y,z: self._A * self._Y(x,y,z) + self._Z(x,y,z) + self._B
            self._q4 = lambda x,y,z: self._A + self._Y(x,y,z)

            self.condition1 = lambda x,y,z: self._q4(x,y,z) * self._q3(x,y,z) - self._q2(x,y,z)
            self.condition2 = lambda x,y,z: (self._q4(x,y,z) * self._q3(x,y,z) - self._q2(x,y,z)) * (self._q2(x,y,z) * self._q1(x,y,z) - self._q0(x,y,z) * self._q3(x,y,z)) - (self._q4(x,y,z) * self._q1(x,y,z) - self._q0(x,y,z))**2



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
        return - self.model.alpha1 / (1 + self.model.alpha1 * self.m3) * self.model._physics.beta(t=0., X=self.m3)


    def compute(self, x : float | int = 0., y : float | int | None = None, z : float | int | None = None, **kwargs):
        """
        Computes the two Routh Hurwitz conditions for 5th-order polynomials
        """

        verb = kwargs.get('verbose', False)
        if verb:
            print(f"A: {round(self._A, 4)}\t\t\t\tB: {round(self._B, 4)}\t\t\t\tC: {round(self._C, 4)}\t\t\t\tD: {round(self._D, 4)}\t\t\t\tE: {round(self._E, 4)}")
            print(f"Beta: {round(self.beta, 4)}\t\t\tDot Beta: {round(self.dot_beta, 4)}")

        if self.target == 'a1' or self.target == 'a2':
            if verb:
                print(f"X: {round(self._X(x), 4)}\t\t\t\tY: {round(self._Y(x), 4)}")
            return self.condition1(x), self.condition2(x)
        else:
            assert y is not None, "For target a1 and a2, only one variable is needed. For other targets, two variables are needed."
            if verb:
                print(f"X: {round(self._X(x,y,z), 8)}\t\t\t\t\tY: {round(self._Y(x,y,z), 8)}\t\t\t\tZ: {round(self._Z(x,y,z), 8)}")
                print(f"q0: {round(self._q0(x,y,z), 12)}\t\t\t\t\tq1: {round(self._q1(x,y,z), 12)}\t\t\tq2: {round(self._q2(x,y,z), 12)}\t\t\t3: {round(self._q3(x,y,z), 12)}\t\t\tq4: {round(self._q4(x,y,z), 12)}")
                print(f"Cond1: {round(self.condition1(x,y,z), 8)}\t\t\tCond2: {self.condition2(x,y,z)}\n\n")
            return self.condition1(x, y,z), self.condition2(x, y,z)