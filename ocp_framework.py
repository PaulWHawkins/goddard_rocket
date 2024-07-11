"""
Framework for solving a nonlinear optimal control problem with Hermite Simpson collocation and 
casadi
"""

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
from pydantic import BaseModel, PositiveInt


class PhaseSettings(BaseModel):
    """Holds settings for an OCP phase"""

    phase: PositiveInt = 1
    single_phase: bool = False


class OcpPhase:
    """Class with methods used for all OCP problems"""

    default_N = 50

    def __init__(
        self, opti: ca.Opti, settings: PhaseSettings, N: Optional[int] = None
    ) -> None:

        self.settings = settings
        self.sol: ca.OptiSol
        self.opti: ca.Opti = opti

        self.N: int = N if N is not None else self.default_N

        self.x: ca.MX
        self.u: ca.MX
        self.t0: ca.MX = opti.variable(1, 1)
        self.tf: ca.MX = opti.variable(1, 1)

    def set_defect(self, x: ca.MX, u: ca.MX, h: Union[float, ca.SX]) -> None:
        """Set defect for a phase/ocp"""

        u_lower = u[:, 0:-1]
        u_upper = u[:, 1:]
        u_mid = (u_lower + u_upper) / 2

        x_lower = x[:, 0:-1]
        x_upper = x[:, 1:]

        # Perform collocation
        Xc = 1 / 2 * (x_lower + x_upper) + h / 8 * (
            self.dxdt(x_lower, u_lower) - self.dxdt(x_upper, u_upper)
        )
        defect_state = (x_lower - x_upper) + h / 6 * (
            self.dxdt(x_lower, u_lower)
            + 4 * self.dxdt(Xc, u_mid)
            + self.dxdt(x_upper, u_upper)
        )

        self.opti.subject_to(ca.vertcat(defect_state) == 0)

    @staticmethod
    def norm(x: ca.MX) -> ca.MX:
        return ca.sqrt(x[0, :] ** 2 + x[1, :] ** 2)

    def dxdt(self, x, u) -> ca.MX:
        raise NotImplementedError("Must be overridden!")

    def plot_results(
        self,
        sol,
        u_names: Optional[list[str]] = None,
        u_idx: Optional[list[int]] = None,
        x_names: Optional[list[str]] = None,
        x_idx: Optional[list[int]] = None,
        plot_str: Optional[str] = None,
        title: Optional[str] = None,
        makeFig=True,
    ):
        """Generate plot of final states"""

        x = self.opti.value(self.x)
        print(f"Final values of x: {x[:, -1]}")
        u = self.opti.value(self.u)
        print(f"Final values of u: {u[-1]}")
        print(f"Average value of u for stage:  {np.mean(u)}")

        t0 = self.opti.value(self.t0)
        tf = self.opti.value(self.tf)
        t1 = np.linspace(t0, tf, len(x[0, :]))

        if makeFig:
            plt.figure()

        # Plot controls
        len_u = 1
        if len_u > 1:
            idx_list = range(len(u)) if u_idx is None else u_idx
            for pos, uidx in enumerate(idx_list):
                if u_names:
                    label = u_names[pos]
                else:
                    label = f"u_{pos}"
                plt.plot(t1, sol.value(u[uidx, :]), label=label)
        else:
            idx_list = range(len(u)) if u_idx is None else u_idx
            if u_idx is None or len(u_idx) > 0:
                if u_names:
                    label = u_names[0]
                else:
                    label = f"u_{0}"
                plt.plot(t1, sol.value(u[:]), label=label)

        # Plot states
        idx_list = range(len(x)) if x_idx is None else x_idx

        for pos, xidx in enumerate(idx_list):
            if x_names:
                label = x_names[pos]
            else:
                label = f"x_{pos}"

            plt.plot(t1, sol.value(x)[xidx, :], label=label)

        if title:
            plt.title(title)

        plt.xlabel("time (s)")
        plt.ylabel("value")
        if plot_str:
            plt.savefig(plot_str)
            plt.close()


class OCP:
    """OCP phase with classes"""

    def __init__(self, n_phases: int) -> None:
        """Initialize with list of phases"""
        self.opti = ca.Opti()

        self.n_phases = n_phases
        self.phases: list[Optional[OcpPhase]] = [None] * n_phases

    def run(self) -> ca.OptiSol:
        self.validate_ocp_phase()

        # Link phases
        if self.n_phases > 1:
            for i in range(self.n_phases - 1):
                self.link_states(i, i + 1, link_control=False)

        # Apply objective function
        self.x = self.combine_x()
        self.opti.minimize(-ca.mmax(self.x[0, :]))

        s_opts = {"ipopt.tol": 1e-6}
        self.opti.solver("ipopt", s_opts)

        self.sol = self.opti.solve()
        return self.sol

    def get_opti(self):
        return self.opti

    def combine_x(self) -> ca.MX:
        phases = self.validate_ocp_phase()
        x_list = [obj.x for obj in phases]
        return ca.horzcat(*x_list)

    def validate_ocp_phase(
        self, idx: Optional[Union[int, list[int]]] = None
    ) -> list[OcpPhase]:
        """Ensure that all phases (or all phases in idx) are OcpPhases"""
        # Handle input
        if idx is None:
            validx = list(range(self.n_phases))
        elif isinstance(idx, int):
            validx = [idx]
        else:
            validx = idx

        # Check each value and return phases if valid
        return_list: list[OcpPhase] = []
        for i in validx:
            phase = self.phases[i]
            if isinstance(phase, OcpPhase):
                return_list.append(phase)
            else:
                raise ValueError(f"Phase {i} is not an OcpPhase")
        return return_list

    def link_states(self, phase_idx1: int, phase_idx2: int, link_control=True):
        """Link all states between two phases"""

        phase_1, phase_2 = self.validate_ocp_phase([phase_idx1, phase_idx2])

        self.opti.subject_to(phase_1.x[:, -1] == phase_2.x[:, 0])
        if link_control:
            self.opti.subject_to(phase_1.u[:, -1] == phase_2.u[:, 0])

        self.opti.subject_to(phase_1.tf == phase_2.t0)

    def plot_results(self, u_names, x_names, title):
        base_save_name = title  # f"goodard_rocket_phases_{self.n_phases}"
        plt.figure()
        for idx in range(self.n_phases):
            u_mod = [n + f"_{idx+1}" for n in u_names]
            x_mod = [n + f"_{idx+1}" for n in x_names]
            self.phases[idx].plot_results(
                self.sol,
                u_names=u_mod,
                x_names=x_mod,
                makeFig=False,
            )

        if self.n_phases > 1:
            plt.legend()

        plt.title(
            f"{self.n_phases} phase with hmax = {ca.mmax(self.opti.value(self.x[0, :]))}"
        )
        plt.savefig(base_save_name)

        for idx, u_name in enumerate(u_names):
            plt.figure()
            for phase_idx in range(self.n_phases):
                u_phase = f"phase_{phase_idx+1}"
                self.phases[phase_idx].plot_results(
                    self.sol,
                    u_names=[u_phase],
                    x_names=[],
                    x_idx=[],
                    u_idx=[idx],
                    makeFig=False,
                )
            if self.n_phases > 1:
                plt.legend()
            plt.title(u_name)
            plt.savefig(base_save_name + "_" + u_name)
            plt.close()

        for idx, x_name in enumerate(x_names):
            plt.figure()
            for phase_idx in range(self.n_phases):
                u_phase = f"phase_{phase_idx+1}"

                self.phases[phase_idx].plot_results(
                    self.sol,
                    x_names=[u_phase],
                    u_names=[],
                    u_idx=[],
                    x_idx=[idx],
                    makeFig=False,
                )
            if self.n_phases > 1:
                plt.legend()
            plt.title(x_name)
            plt.savefig(base_save_name + "_" + x_name)
            plt.close()
