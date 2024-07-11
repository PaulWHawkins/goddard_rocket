"""
Outline for code for Goddard Rocket and Backbone of 
"""

import casadi as ca
import numpy as np
from typing import Optional
from constants import g0, umax, umin, m0, mTf, D0, h0, href, c
from pydantic import Field, PositiveInt
from ocp_framework import OcpPhase, PhaseSettings


class GoddardSettings(PhaseSettings):
    """Settings for Goddard Rocket Problem"""

    phase: PositiveInt = Field(default=1, le=3)
    path_constraint: bool = False


class GoddardRocketPhase(OcpPhase):
    """Phase of Goddard Rocket OCP problem.

    Supports 1-3 phases.
    """

    def __init__(
        self, opti: ca.Opti, settings: GoddardSettings, N: Optional[int] = None
    ) -> None:

        super().__init__(opti, settings=settings, N=N)

        # Setting state vars
        x = self.opti.variable(3, self.N + 1)
        self.x = x

        # Setting input vars
        u = self.opti.variable(1, self.N + 1)
        self.u = u

        # Setting time
        h = (self.tf - self.t0) / (self.N - 1)

        # Set defect
        self.set_defect(x, u, h)

        # Set state constraints
        self.opti.subject_to(ca.Opti_bounded(h0, x[0, :], ca.inf))
        self.opti.subject_to(ca.Opti_bounded(mTf, x[2, :], m0))
        self.opti.subject_to(self.tf > self.t0)

        # Set input contstraints
        self.opti.subject_to(ca.Opti_bounded(umin, u[0, :], umax))

        # Set intial values
        self.opti.set_initial(
            u,
            np.concatenate(
                (
                    np.ones(round(np.floor(self.N / 2))) * umax,
                    np.zeros(round(np.ceil(self.N / 2) + 1)),
                )
            ),
        )

        self.opti.set_initial(x[0, :], h0)
        self.opti.set_initial(x[1, :], 1)
        self.opti.set_initial(x[2, :], (mTf + m0) / 2)

        # Apply custom settings for each phase
        if settings.phase == 1:
            self.apply_initial_constraints()
        elif settings.phase == 3:
            self.zero_u()
        if settings.path_constraint:
            self.add_path_constraint()
            self.limit_u  # clean up dividing line between stages

    def dxdt(self, state_vector: ca.MX, u: ca.MX) -> ca.MX:
        """Get forces and moment on the vehicle"""
        # Extract forces
        h: ca.MX = state_vector[0, :]
        v: ca.MX = state_vector[1, :]
        m: ca.MX = state_vector[2, :]
        T: ca.MX = u[0, :]

        g = g0
        D = self.get_aero_forces(state_vector)

        vdot = 1 / m * (T - D) - g
        hdot = v
        mdot = -T / c

        return ca.vertcat(hdot, vdot, mdot)

    def get_aero_forces(self, state_vector: ca.MX) -> ca.MX:
        """Return aerodynamic forces on vehicle

        Just drag in 1D case
        """
        h: ca.MX = state_vector[0, :]
        v: ca.MX = state_vector[1, :]
        D = D0 * v**2 * ca.exp(-h / href)

        return D

    def apply_initial_constraints(self):
        """Apply limits on initial phase"""
        # Set state constraints
        self.opti.subject_to(self.t0 == 0)  # ca.Opti_bounded(0, self.tf, ca.inf))
        # self.opti.subject_to(ca.Opti_bounded(0.0, self.tf, ca.inf))

        # initial constraints
        self.opti.subject_to(ca.Opti_bounded(h0, self.x[0, 0], h0))
        self.opti.subject_to(ca.Opti_bounded(0, self.x[1, 0], 0))
        self.opti.subject_to(ca.Opti_bounded(m0, self.x[2, 0], m0))

        # Set input contstraints
        if not self.settings.single_phase:
            self.opti.subject_to(self.u[0, :] == umax)

    def add_path_constraint(self):
        """Add path constraint used on singular arc"""
        state_vector = self.x
        h = state_vector[0, :]
        v = state_vector[1, :]
        m = state_vector[2, :]
        T = self.u

        term1 = T - D0 * v**2 * ca.exp(-h / href) - m * g0
        term2 = -m * g0 / (1 + 4 * (c / v) + 2 * (c**2 / (v**2)))
        term3 = c**2 / href / g0 * (1 + v / c) - 1 - 2 * c / v
        sarcconst = term1 + term2 * term3
        self.opti.subject_to(sarcconst == 0)

    def limit_u(self):
        """Limit input to just below maximum

        Used to separate max thrust phase from singular control
        """
        self.opti.subject_to(ca.Opti_bounded(umin, self.u, umax * 0.99))

    def zero_u(self):
        """Constrain u to zero"""
        self.opti.subject_to(self.u[0, :] == umin)
