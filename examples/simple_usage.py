"""Build and plot a simple ternary phase diagram."""

import matplotlib.pyplot as plt
import numpy as np

from pyphasediagram import PhaseDiagram


def main() -> None:
    """Build a symmetric phase diagram and display its summary plot."""
    chi = 2.7
    chis = np.array(
        [
            [0.0, chi, chi],
            [chi, 0.0, chi],
            [chi, chi, 0.0],
        ]
    )

    diagram = PhaseDiagram(chis)
    diagram.build()
    diagram.plot_summary()
    plt.show()


if __name__ == "__main__":
    main()
