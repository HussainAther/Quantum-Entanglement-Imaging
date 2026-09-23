from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_case(index: int) -> None:
    path = Path(
        f"outputs/prism_singularity_trajectory/"
        f"remove_{index:02d}.csv"
    )

    df = pd.read_csv(path)

    plt.figure()

    plt.semilogy(
        df["iteration"],
        df["sigma_min"],
        label="smallest singular value",
    )

    plt.semilogy(
        df["iteration"],
        df["sigma_second_min"],
        label="second-smallest singular value",
    )

    plt.semilogy(
        df["iteration"],
        df["residual"],
        label="equilibrium residual",
    )

    plt.xlabel("Iteration")
    plt.ylabel("Magnitude")
    plt.title(
        f"Prism cable removal {index}: "
        "approach to singularity"
    )

    plt.legend()
    plt.tight_layout()

    output = Path(
        f"outputs/prism_singularity_trajectory/"
        f"remove_{index:02d}.png"
    )

    plt.savefig(
        output,
        dpi=200,
    )

    plt.close()

    print(f"Saved {output}")


def main() -> None:
    # One representative from each symmetry class.
    plot_case(3)
    plot_case(9)


if __name__ == "__main__":
    main()
