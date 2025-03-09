"""
Script to intervene on residual stream on the arithmetic error detection task.
"""

import argparse
import logging
import os
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import torch as t
from transformer_lens import HookedTransformer

from reasoning_mistake.circuits_functions.circuit_discovery import load_data, load_model
from reasoning_mistake.circuits_functions.intervene.residual_intervention import (
    intervene_on_residual_stream,
)
from reasoning_mistake.utils import (
    bar_plot_residual_interventions,
    set_seed,
    setup_logging,
)

TEMPLATES = ["0", "1", "2", "3", "4", "5", "6", "7"]


C_AND_A_LABELS = [
    ("[C-second]_occ_1", "[A-second]_occ_1"),
    ("[C]_occ_1", "[A]_occ_1"),
]


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line input arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    # General config
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/math",
        help="Directory to load the data from",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Directory of cached model weights",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random generator seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for computation"
    )

    # Data config
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for data loading"
    )

    # Model & circuit config
    parser.add_argument(
        "--model", type=str, default="gpt2", help="Name of the model to use"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=500,
        help="Number of samples used to compute edge scores",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=100,
        help="Number of samples used to select edges to include in circuit based",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="diff",
        choices=["correct", "diff", "kl"],
        help="Faithfulness metric to use",
    )
    parser.add_argument(
        "--grad_function",
        type=str,
        default="logit",
        choices=["logit", "prob", "logprob", "logit_exp"],
        help="Function to apply to logits for finding edge scores before computing gradient",
    )
    parser.add_argument(
        "--answer_function",
        type=str,
        default="avg_diff",
        choices=["avg_diff", "avg_val", "mse"],
        help="Loss function to apply to answer and wrong answer for finding edge scores",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scaling factor for the attention head intervention",
    )
    parser.add_argument(
        "--save_plot_dir",
        type=str,
        default="results/residual-interventions",
        help="Directory to save the attention intervention plots",
    )
    args = parser.parse_args()

    return args


def prepare_data(
    model: HookedTransformer,
    model_name: str,
    template: str,
    args: argparse.Namespace,
) -> Tuple[List[t.Tensor], List[t.Tensor], List[str]]:
    """
    Prepare data for causal intervention on residual stream activations.
    This function loads the filtered math data for the C variation
    and create a paired dataloader with the same data but
    variation in both C and A positions.

    Args:
        model (HookedTransformer): The model to use for the intervention.
        model_name (str): The name of the model.
        template (str): The template to load the math data for.
        args (argparse.Namespace): The input arguments.

    Returns:
        Tuple[List[t.Tensor], List[t.Tensor], List[str]]: The data loaders for the single variation
        and both variation data and the corresponding seq_labels.
    """

    _, promptloader_C, seq_labels = load_data(
        model=model,
        data_dir=args.data_dir,
        save_model_name=model_name,
        template=template,
        variation="C",
        num_train=args.train_size,
        num_test=args.test_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    if "llama-3.2" in model_name.lower():
        seq_labels = ["[bos]"] + seq_labels

    test_loader_C: List[t.Tensor] = [batch.clean for batch in promptloader_C]
    test_loader_both: List[t.Tensor] = []

    indexes_C_A = [
        (seq_labels.index(C), seq_labels.index(A))
        for C, A in C_AND_A_LABELS
        if C in seq_labels and A in seq_labels
    ]

    for batch in test_loader_C:
        new_batch = deepcopy(batch)
        new_batch[:, indexes_C_A[0][1]] = batch[:, indexes_C_A[0][0]]
        test_loader_both.append(new_batch)

    return test_loader_C, test_loader_both, seq_labels


def main() -> None:
    """
    Main script.
    """
    args = parse_arguments()
    args.device = (
        t.device("cuda")
        if t.cuda.is_available() and args.device == "cuda"
        else t.device("cpu")
    )

    setup_logging(args.verbose)
    set_seed(args.seed)

    # model & data
    model = load_model(args.model, args.device, args.cache_dir)
    model.tokenizer.add_bos_token = False
    save_model_name = args.model.split("/")[-1].lower()

    # Main loop
    original_accuracies_both = []
    intervened_accuracies_both = []
    original_accuracies_C = []
    intervened_accuracies_C = []

    for template in TEMPLATES:
        test_loader_C, test_loader_both, seq_labels = prepare_data(
            model=model,
            model_name=save_model_name,
            template=template,
            args=args,
        )
        logging.info(f"Loaded data for template {template}")

        # Intervene on residual stream for both data
        original_acc_both, intervened_acc_both = intervene_on_residual_stream(
            model=model,
            model_name=args.model,
            variation_prompts=test_loader_both,
            seq_labels=seq_labels,
            alpha=args.alpha,
        )

        logging.info(f"Original both accuracy: {original_acc_both}")
        logging.info(f"Intervened both accuracy: {intervened_acc_both}")

        original_accuracies_both.append(original_acc_both)
        intervened_accuracies_both.append(intervened_acc_both)

        # Intervene on residual stream for C data
        original_acc_C, intervened_acc_C = intervene_on_residual_stream(
            model=model,
            model_name=args.model,
            variation_prompts=test_loader_C,
            seq_labels=seq_labels,
            alpha=args.alpha,
        )

        logging.info(f"Original C accuracy: {original_acc_C}")
        logging.info(f"Intervened C accuracy: {intervened_acc_C}")

        original_accuracies_C.append(original_acc_C)
        intervened_accuracies_C.append(intervened_acc_C)

    logging.info(f"Original mean accuracies: {np.mean(original_accuracies_both)}")
    logging.info(f"Intervened mean accuracies: {np.mean(intervened_accuracies_both)}")
    logging.info(f"Original mean accuracies C: {np.mean(original_accuracies_C)}")
    logging.info(f"Intervened mean accuracies C: {np.mean(intervened_accuracies_C)}")

    os.makedirs(args.save_plot_dir, exist_ok=True)
    uid = f"alpha_{args.alpha}_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"
    file_path = f"{args.save_plot_dir}/{save_model_name}_{uid}.png"

    bar_plot_residual_interventions(
        original_accuracies_both=original_accuracies_both,
        intervened_accuracies_both=intervened_accuracies_both,
        original_accuracies_C=original_accuracies_C,
        intervened_accuracies_C=intervened_accuracies_C,
        file_path=file_path,
    )


if __name__ == "__main__":
    main()
