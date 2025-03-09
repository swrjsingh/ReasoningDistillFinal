"""
Script to interpret attention components on the arithmetic error detection task.
"""

import argparse
import logging

import torch as t

from reasoning_mistake.circuits_functions.circuit_analysis import load_circuit
from reasoning_mistake.circuits_functions.circuit_discovery import load_data, load_model
from reasoning_mistake.circuits_functions.interpret.attention_analysis import (
    visualize_attention_patterns,
    visualize_cross_template_attention_patterns,
)
from DistilledReasoningFinal.reasoning_mistake.circuits_functions.interpret.qk_analysis import (
    visualize_qk_patterns,
)
from reasoning_mistake.utils import get_save_dir_name, set_seed, setup_logging

TEMPLATES = ["0", "1", "4", "5", "6", "7"]


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
        "--intersection_overlap_C",
        type=str,
        default="1.0",
        help="Overlapping templates parameter in the intersection circuit for C",
    )
    parser.add_argument(
        "--intersection_overlap_A",
        type=str,
        default="1.0",
        help="Overlapping templates parameter in the intersection circuit for A",
    )
    parser.add_argument(
        "--cross_template_averaging",
        action="store_true",
        default=True,
        help="Enable averaging attention patterns across templates",
    )
    args = parser.parse_args()

    return args


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

    # load intersection circuits
    circuits = {}
    for variation_circuit in ["C", "A"]:
        intersection_overlap = (
            args.intersection_overlap_C
            if variation_circuit == "C"
            else args.intersection_overlap_A
        )

        circuit = load_circuit(
            save_model_name=save_model_name,
            variation=variation_circuit,
            template_name="intersection",
            grad_function=args.grad_function,
            answer_function=args.answer_function,
            train_size=args.train_size,
            intersection_overlap=intersection_overlap,
        )
        circuits[variation_circuit] = circuit

    logging.info("Circuits loaded.")

    # Pattern analysis
    if args.cross_template_averaging:
        # For cross-template averaging, collect all data first
        all_templates_dataloaders = {}
        for template in TEMPLATES:
            template_dataloaders = {}
            for variation in ["C", "A", "both", "none"]:
                if variation == "none":
                    variation_data = "C"
                else:
                    variation_data = variation

                filtered = False if variation == "both" else True
                train_loader, test_loader, seq_labels = load_data(
                    model=model,
                    data_dir=args.data_dir,
                    save_model_name=save_model_name,
                    template=template,
                    variation=variation_data,
                    num_train=args.train_size,
                    num_test=args.test_size,
                    batch_size=args.batch_size,
                    device=args.device,
                    filtered=filtered,
                )
                template_dataloaders[variation] = train_loader
            all_templates_dataloaders[template] = template_dataloaders

        # Now visualize with cross-template averaging
        result_path = get_save_dir_name(
            prefix="results/attention_analysis/patterns/tokenwise",
            template="cross_template",
        )
        result_path += f"/{save_model_name}"
        uid = f"pattern_cross_template_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"

        logging.info("Collecting activations across all templates.")
        visualize_cross_template_attention_patterns(
            model=model,
            model_name=save_model_name,
            all_templates_dataloaders=all_templates_dataloaders,
            circuits=circuits,
            save_dir=result_path,
            uid=uid,
        )
        logging.info("Cross-template attention patterns saved.")

    # Individual template analysis (still needed even with cross-template averaging)
    for template in TEMPLATES:
        dataloaders = {}
        for variation in ["C", "A", "both", "none"]:
            if variation == "none":
                variation_data = "C"
            else:
                variation_data = variation

            filtered = False if variation == "both" else True
            train_loader, test_loader, seq_labels = load_data(
                model=model,
                data_dir=args.data_dir,
                save_model_name=save_model_name,
                template=template,
                variation=variation_data,
                num_train=args.train_size,
                num_test=args.test_size,
                batch_size=args.batch_size,
                device=args.device,
                filtered=filtered,
            )
            dataloaders[variation] = train_loader

        result_path = get_save_dir_name(
            prefix="results/attention_analysis/patterns", template=template
        )
        result_path += f"/{save_model_name}"
        uid = f"pattern_template_{template}_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"

        logging.info("Collecting activations on dataloader.")
        visualize_attention_patterns(
            model=model,
            model_name=save_model_name,
            dataloaders=dataloaders,
            circuits=circuits,
            save_dir=result_path,
            uid=uid,
        )
        logging.info("Attention patterns saved.")

    # Query-key analysis
    result_path = "results/attention_analysis/qk_analysis"
    template = "intersection"
    result_path += f"/{save_model_name}"
    uid = f"pattern_template_{template}_gradfunc_{args.grad_function}_ansfunc_{args.answer_function}_train_size_{args.train_size}"

    visualize_qk_patterns(model=model, circuits=circuits, save_dir=result_path, uid=uid)
    logging.info("Query-key patterns saved.")


if __name__ == "__main__":
    main()
