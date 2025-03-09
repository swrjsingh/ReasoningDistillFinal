import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional

# Path to the JSON files with the faithfulness data
RESULTS_DIR = "./results/discovered-circuits/tokenwise/template_intersection/qwen2.5-1.5b-instruct"
# Models that were evaluated
VARIATIONS = ["C", "A", "computation"]


def load_json_to_dict(file_path: str) -> Dict:
    """Load a JSON file into a dictionary."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return {}


def plot_dual_y_axis(
    data: Dict[str, Dict[str, Union[Tuple, List]]],
    file_path: str,
    x_axis_label: str = "X-axis",
    x_axis_ticks: Optional[List[str]] = None,
    y_lim: List[int] = [0, 120],
    title: str = "Dual Y-axis Plot",
) -> None:
    """
    Creates and saves a dual y-axis plot based on the input dictionary.

    Parameters:
        data (Dict[str, Dict[str, Union[Tuple, List]]]):
            A dictionary where the first-level keys are x-axis values (as strings),
            and the values are dictionaries with two keys: "faithfulness" and "n_edges".
            Each of these keys maps to a tuple (mean, std_dev).
        file_path (str):
            The path where the plot will be saved. Ensures the directory exists.
        x_axis_label (Optional[str]):
            Label for the x-axis. Defaults to "X-axis".
        x_axis_ticks (Optional[List[str]]): Tick descriptions for x-axis. Defaults to None.
        y_lim (List[int]): Range of y-axis for faithfulness plotting. Defaults to [0, 120].
        title (Optional[str]):
            Title of the plot. Defaults to "Dual Y-axis Plot".
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        x_values = list(data.keys())
        x_numeric = np.arange(len(x_values))

        if x_axis_ticks is None:
            x_axis_ticks = [str(val) for val in x_numeric]
        else:
            assert len(x_axis_ticks) == len(x_values), (
                "'x_axis_ticks' is non-empty but of unequal length to x_values!"
            )

        faithfulness_mean = np.array([data[x]["faithfulness"][0] for x in x_values])
        faithfulness_std = np.array([data[x]["faithfulness"][1] for x in x_values])
        n_edges_mean = np.array([data[x]["n_edges"][0] for x in x_values])
        n_edges_std = np.array([data[x]["n_edges"][1] for x in x_values])

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the first line (faithfulness)
        ax1.errorbar(
            x_numeric,
            faithfulness_mean,
            yerr=faithfulness_std,
            fmt="-o",
            label="Faithfulness",
            capsize=5,
        )
        ax1.set_xlabel(x_axis_label)
        ax1.set_ylim(
            min(y_lim[0], min(1.2 * (faithfulness_mean - faithfulness_std))),
            max(y_lim[1], max(1.2 * (faithfulness_mean + faithfulness_std))),
        )
        ax1.set_ylabel("Faithfulness", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")
        ax1.set_xticks(x_numeric)
        ax1.set_xticklabels(x_axis_ticks, rotation=0, ha="center")

        # Create the second y-axis
        ax2 = ax1.twinx()
        ax2.errorbar(
            x_numeric,
            n_edges_mean,
            yerr=n_edges_std,
            fmt="-s",
            label="Number of Edges",
            color="green",
            capsize=5,
        )
        ax2.set_ylim(0, max(1.2 * (n_edges_mean + n_edges_std)))
        ax2.set_ylabel("Number of Edges", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # Add title and legend
        fig.suptitle(title)

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(file_path, format="png", dpi=300)

        # Show the plot in the notebook (added for Kaggle)
        plt.show()

        print(f"Plot saved to {file_path}")

    except Exception as e:
        print(f"An error occurred while creating the plot: {e}")


def main():
    """Generate the plots for all variations."""
    print("Starting plot generation...")

    for variation in VARIATIONS:
        print(f"Processing variation: {variation}")

        # Path to the JSON file
        json_file = f"{RESULTS_DIR}/intersection_faithfulness_{variation}_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_50.json"

        # Load the data
        data = load_json_to_dict(json_file)
        if not data:
            continue

        # Path for the output plot
        output_file = f"{RESULTS_DIR}/intersection_faithfulness_{variation}_template_intersection_gradfunc_logit_ansfunc_avg_diff_train_size_50.png"

        # Generate the plot
        plot_dual_y_axis(
            data=data,
            file_path=output_file,
            x_axis_label="overlap",
            x_axis_ticks=["1/6", "2/6", "3/6", "4/6", "5/6", "6/6"],
            title=f"Qwen2.5-1.5B-Instruct - {variation}",
        )

    print("Plot generation complete!")


if __name__ == "__main__":
    main()
