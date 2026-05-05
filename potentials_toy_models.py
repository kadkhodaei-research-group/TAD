# Class for implementing various analytical model Potential Energy Surfaces(PESs)

from numpy import pi, sin, cos
import numpy as np


class pes_models(object):

    # This part contains info on potential parameters
    def __init__(self, potential):

        if potential == 'Muller-Brown':  # The trickiest potential. Climber helps a lot here.
            self.domain = ((-2.0, 1.5), (-1.0, 2.0))
            self.energies = (-200.0, 200.0, 5.0)  # min / max / step for energy contours in plot
            self.AA = (-200., -100., -170., 15.)
            self.b = (0., 0., 11., 0.6)
            self.a = (-1., -1., -6.5, 0.7)
            self.c = (-10., -10., -6.5, 0.7)
            self.x0 = (1., 0., -0.5, -1.)
            self.y0 = (0., 0.5, 1.5, 1.)
        elif potential == 'Leps' or potential == 'Leps-HO':
            self.domain = ((0.0, 5.0), (0.0, 5.0))
            self.energies = (-10.0, 15.0, 0.25)
            self.a = 0.05
            self.b = 0.30
            self.c = 0.05
            self.d_ab = 4.746
            self.d_bc = 4.746
            self.d_ac = 3.445
            self.r0_ab = 0.742
            self.r0_bc = 0.742
            self.r0_ac = 0.742
            self.alpha_ab = 1.942
            self.alpha_bc = 1.942
            self.alpha_ac = 1.942
            if potential == 'Leps-HO':
                self.domain = ((0.0, 5.0), (0.0, 5.0))
                self.energies = (-10.0, 15.0, 0.25)
                self.r_ac = 3.742
                self.k_c = 0.2025  # force constant for harmonic coupling
                self.c = 1.154
        elif potential == 'Karplus':
            self.domain = ((-3.0, 3.0), (-2.0, 2.0))
            self.energies = (-6.0, 2.0, 0.2)
        elif potential == 'Wolfe-Quapp':
            self.domain = ((-2.0, 2.0), (-2.0, 2.0))
            self.energies = (-12.0, 10.0, 0.2)
        elif potential == 'three_hole_pot':
            self.domain = ((-2.0, 2.0), (-2.0, 2.0))
            self.energies = (-12.0, 10.0, 0.2)
        elif potential == 'V1':
            self.domain = ((-2.0, 2.0), (-2.0, 2.0))
            self.energies = (-12.0, 10.0, 0.2)
        elif potential == 'V2':
            self.domain = ((-2.0, 2.0), (-2.0, 2.0))
            self.energies = (-12.0, 10.0, 0.2)
        elif potential == 'V3':
            self.domain = ((-2.0, 2.0), (-2.0, 2.0))
            self.energies = (0, 6, 0.2)
        elif potential == 'V4':
            self.domain = ((-2.0, 2.0), (-2.0, 2.0))
            self.energies = (0, 10.0, 0.2)
        elif potential == 'V5':
            self.domain = ((-1.0, 1.0), (-1.0, 1.0))
            self.energies = (-1.2, 1.2, 0.2)
        elif potential == 'V7':
            self.domain = ((4, 11), (4, 11))
            self.energies = (-4.5, 4.5, 0.2)
        elif potential == 'V8':  # New sinusoidal potential
            self.domain = ((-1.5, 1.5), (-1.5, 1.5))
            self.energies = (-1.0, 1.0, 0.1)
        return

    def mb_pot(self, x):
        potvalue = 0.0
        for i in range(4):
            potterm = self.AA[i] * (np.exp(self.a[i] * (x[0] - self.x0[i]) ** 2 + \
                                           self.b[i] * (x[0] - self.x0[i]) * (x[1] - self.y0[i]) + \
                                           self.c[i] * (x[1] - self.y0[i]) ** 2))
            potvalue += potterm
        return potvalue

    def mb_pot_grad(self, x):
        grad_x = 0.0
        grad_y = 0.0
        for i in range(4):
            grad_x += 2.0 * self.AA[i] * self.a[i] * (x[0] - self.x0[i]) * (
                np.exp(self.a[i] * (x[0] - self.x0[i]) ** 2 + \
                       self.b[i] * (x[0] - self.x0[i]) * (x[1] - self.y0[i]) + \
                       self.c[i] * (x[1] - self.y0[i]) ** 2))
            grad_x += self.AA[i] * self.b[i] * (x[1] - self.y0[i]) * (np.exp(self.a[i] * (x[0] - self.x0[i]) ** 2 + \
                                                                             self.b[i] * (x[0] - self.x0[i]) * (
                                                                                         x[1] - self.y0[i]) + \
                                                                             self.c[i] * (x[1] - self.y0[i]) ** 2))
            grad_y += self.AA[i] * self.b[i] * (x[0] - self.x0[i]) * (np.exp(self.a[i] * (x[0] - self.x0[i]) ** 2 + \
                                                                             self.b[i] * (x[0] - self.x0[i]) * (
                                                                                         x[1] - self.y0[i]) + \
                                                                             self.c[i] * (x[1] - self.y0[i]) ** 2))
            grad_y += 2.0 * self.AA[i] * self.c[i] * (x[1] - self.y0[i]) * (
                np.exp(self.a[i] * (x[0] - self.x0[i]) ** 2 + \
                       self.b[i] * (x[0] - self.x0[i]) * (x[1] - self.y0[i]) + \
                       self.c[i] * (x[1] - self.y0[i]) ** 2))
        return np.array([grad_x, grad_y])

    def coul(self, r, pair='ab'):
        if pair == 'ab':
            q = ((self.d_ab / 2.0) * ((3.0 / 2.0) * np.exp(-2.0 * self.alpha_ab * (r - self.r0_ab)) - np.exp(
                -self.alpha_ab * (r - self.r0_ab)))) \
                / (1.0 + self.a)
        elif pair == 'bc':
            q = ((self.d_bc / 2.0) * ((3.0 / 2.0) * np.exp(-2.0 * self.alpha_bc * (r - self.r0_bc)) - np.exp(
                -self.alpha_bc * (r - self.r0_bc)))) \
                / (1.0 + self.b)
        elif pair == 'ac':
            q = ((self.d_ac / 2.0) * ((3.0 / 2.0) * np.exp(-2.0 * self.alpha_ac * (r - self.r0_ac)) - np.exp(
                -self.alpha_ac * (r - self.r0_ac)))) \
                / (1.0 + self.c)
        return q

    def exchange(self, r, pair='ab'):
        if pair == 'ab':
            j = (self.d_ab / 4.0) * (np.exp(-2.0 * self.alpha_ab * (r - self.r0_ab)) - 6.0 * np.exp(
                -self.alpha_ab * (r - self.r0_ab)))
        elif pair == 'bc':
            j = (self.d_bc / 4.0) * (np.exp(-2.0 * self.alpha_bc * (r - self.r0_bc)) - 6.0 * np.exp(
                -self.alpha_bc * (r - self.r0_bc)))
        elif pair == 'ac':
            j = (self.d_ac / 4.0) * (np.exp(-2.0 * self.alpha_ac * (r - self.r0_ac)) - 6.0 * np.exp(
                -self.alpha_ac * (r - self.r0_ac)))
        return j

    def leps_pot(self, x):
        coulvalue = 0.0
        exchvalue = 0.0
        pairs = ['ab', 'bc', 'ac']
        params = [self.a, self.b, self.c]
        x = np.append(x, x[1] - x[0])
        for i, pair in enumerate(pairs):
            q = self.coul(x[i], pair)
            coulvalue += q
            j1 = np.square(self.exchange(x[i], pair)) / np.square(1.0 + params[i])
            exchvalue += j1
            try:
                j2 = self.exchange(x[i], pair) * self.exchange(x[i + 1], pairs[i + 1]) / (
                            (1.0 + params[i]) * (1.0 + params[i + 1]))
            except IndexError:
                j2 = self.exchange(x[i], pair) * self.exchange(x[0], pairs[0]) / ((1.0 + params[i]) * (1.0 + params[0]))
            exchvalue -= j2
        potvalue = coulvalue - (exchvalue) ** (1.0 / 2.0)
        return potvalue

    def leps_ho_pot(self, x):
        leps_val = self.leps_pot((x[0], self.r_ac - x[0]))
        ho_val = 2 * self.k_c * (x[0] - ((self.r_ac / 2.) - (x[1] / self.c)) ** 2)
        return leps_val + ho_val

    def wq_pot(self, x):
        return x[0] ** 4 + x[1] ** 4 - (2.0 * np.square(x[0])) - (4.0 * np.square(x[1])) + (x[0] * x[1]) + (
                    0.3 * x[0]) + (0.1 * x[1])

    def karplus_pot(self, x):
        return 0.06 * np.square(np.square(x[0]) + np.square(x[1])) + (x[0] * x[1]) - \
               (9.0 * (np.exp(-np.square(x[0] - 3.0) - np.square(x[1])) + np.exp(
                   -np.square(x[0] + 3.0) - np.square(x[1]))))

    def karplus_pot_grad(self, x):
        return np.array([0.06 * 2 * (x[0] ** 3 + x[0] * x[1] ** 2) + x[1] - 9 * (
                -2 * (x[0] - 3) * np.exp(-np.square(x[0] - 3) - np.square(x[1])) + 2 * (
                x[0] + 3) * np.exp(-np.square(x[0] + 3) - np.square(x[1]))),
                         0.06 * 2 * (x[1] ** 3 + x[1] * x[0] ** 2) + x[0] - 9 * (
                                 -2 * (x[1]) * np.exp(-np.square(x[0] - 3) - np.square(x[1])) - 2 * (
                             x[1]) * np.exp(-np.square(x[0] + 3) - np.square(x[1])))])

    def three_hole_pot(self, x):
        return 3. * np.exp(-x[0] ** 2 - ((x[1] - (1. / 3.)) ** 2)) - 3. * np.exp(-x[0] ** 2 - ((x[1] - (5. / 3.)) ** 2)) \
               - 5. * np.exp(-(x[0] - 1.) ** 2 - x[1] ** 2) - 5. * np.exp(-(x[0] + 1.) ** 2 - x[1] ** 2) \
               + 0.2 * x[0] ** 4 + 0.2 * (x[1] - (1. / 3.)) ** 4

    def three_hole_pot_grad(self, x):
        return np.array([-6. * x[0] * np.exp(-x[0] ** 2 - ((x[1] - (1. / 3.)) ** 2)) + 6. * x[0] * np.exp(
            -x[0] ** 2 - ((x[1] - (5. / 3.)) ** 2)) \
                         + 10. * (x[0] - 1.) * np.exp(-(x[0] - 1.) ** 2 - x[1] ** 2) - 10. * (x[0] + 1.) * np.exp(
            -(x[0] + 1.) ** 2 - x[1] ** 2) \
                         + 0.8 * x[0] ** 3,
                         -6. * (x[1] - (1. / 3.)) * np.exp(-x[0] ** 2 - ((x[1] - (1. / 3.)) ** 2)) + 6. * (
                                     x[1] - (5. / 3.)) * np.exp(-x[0] ** 2 - ((x[1] - (5. / 3.)) ** 2)) \
                         + 10. * x[1] * np.exp(-(x[0] - 1.) ** 2 - x[1] ** 2) + 10. * x[1] * np.exp(
                             -(x[0] + 1.) ** 2 - x[1] ** 2) \
                         + 0.8 * (x[1] - (1. / 3.)) ** 3])

    def V1(self, x):
        return .01 * np.power(x[0], 2) - .1 * np.power(x[1], 2)

    def V2(self, x):
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

    def V3(self, x):
        return 4 * (x[0] ** 2 + x[1] - 1) ** 2 + 2 * (x[0] + x[1] ** 2 - 1) ** 2

    def V4(self, x):  # saddle point is at -0.5 and 3/11 # there is only one saddle point
        return (x[1] + x[0]) ** 2 + 10 * (x[1] - x[0] ** 2) ** 2

    def V5(self, x):
        return -sin(pi * x[0]) * sin(pi * x[1])

    def dV5(self, x):
        return np.array([-pi * cos(pi * x[0]) * sin(pi * x[1]), -pi * sin(pi * x[0]) * cos(pi * x[1])])

    def V6(self, x):
        return x[0] ** 2 + x[1] ** 2

    def dV6(self, x):
        return np.array([x[0] ** 2 + 2 * x[1], 2 * x[0] + x[1] ** 2])

    def V7(self, x):
        return cos(x[0]) * 2 - sin(x[1]) * 2
    
    def V8(self, x):
        """V = -sin(pi*x)*sin(pi*y) - has saddle points at (0,0), (±1,0), (0,±1), (±1,±1)"""
        return -sin(pi * x[0]) * sin(pi * x[1])
    
    def dV8(self, x):
        """Gradient of V8"""
        return np.array([-pi * cos(pi * x[0]) * sin(pi * x[1]), 
                         -pi * sin(pi * x[0]) * cos(pi * x[1])])