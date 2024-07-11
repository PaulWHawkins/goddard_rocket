"""
Script to run Goddard Rocket
"""

from ocp_framework import OCP
from goddard_phases import GoddardRocketPhase, GoddardSettings


## One phase OCP
def run_one_phase():

    # Solve OCP
    ocp = OCP(1)

    # Set up OCP phases
    ocp.phases[0] = GoddardRocketPhase(
        ocp.opti, settings=GoddardSettings(phase=1, single_phase=True)
    )

    # Run and plot
    _ = ocp.run()
    ocp.plot_results(
        u_names=["thrust"],
        x_names=["height", "velocity", "mass"],
        title="goddard_one_stage",
    )


## Three phase OCP
def run_three_phase():

    # Solve OCP
    ocp = OCP(3)

    # Set up OCP phases
    ocp.phases[0] = GoddardRocketPhase(ocp.opti, settings=GoddardSettings(phase=1))
    ocp.phases[1] = GoddardRocketPhase(
        ocp.opti, settings=GoddardSettings(phase=2, path_constraint=False)
    )
    ocp.phases[2] = GoddardRocketPhase(ocp.opti, settings=GoddardSettings(phase=3))

    # Run and plot
    _ = ocp.run()
    ocp.plot_results(
        u_names=["thrust"],
        x_names=["height", "velocity", "mass"],
        title="goddard_three_stage",
    )


def run_three_phase_path_constraint():

    # Solve OCP
    ocp = OCP(3)

    # Set up OCP phases
    # Set up OCP phases
    ocp.phases[0] = GoddardRocketPhase(ocp.opti, settings=GoddardSettings(phase=1))
    ocp.phases[1] = GoddardRocketPhase(
        ocp.opti, settings=GoddardSettings(phase=2, path_constraint=True)
    )
    ocp.phases[2] = GoddardRocketPhase(ocp.opti, settings=GoddardSettings(phase=3))

    # Run and plot
    _ = ocp.run()
    ocp.plot_results(
        u_names=["thrust"],
        x_names=["height", "velocity", "mass"],
        title="goddard_three_stage_constraint",
    )


if __name__ == "__main__":
    run_one_phase()
    run_three_phase()
    run_three_phase_path_constraint()
