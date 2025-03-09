"""
Functions and modules for circuit attention analysis.
"""

import os
import re
import logging
from typing import Dict, List, Literal, Tuple, cast

import matplotlib.pyplot as plt
import torch as t
from auto_circuit.data import PromptDataLoader
from auto_circuit.types import Edge
from tqdm import tqdm
from transformer_lens import HookedTransformer

from reasoning_mistake.circuits_functions.circuit_analysis import Circuit

RELEVANT_TOKENS = [
    "[op1-in-eq]_occ_1",
    "[plus]_occ_1",
    "[op2-in-eq]_occ_1",
    "[equals]_occ_1",
    "[space_after_eq]_occ_1",
    "[C-first]_occ_1",
    "[C-second]_occ_1",
    "[C]_occ_1",
    "[A-first]_occ_1",
    "[A-second]_occ_1",
    "[A]_occ_1",
]


def visualize_attention_patterns(
    model: HookedTransformer,
    model_name: str,
    dataloaders: Dict[str, PromptDataLoader],
    circuits: Dict[str, Circuit],
    save_dir: str,
    uid: str,
) -> None:
    """
    Visualize the attention patterns.

    Args:
        model (HookedTransformer): The model to analyze.
        model_name (str): The name of the model.
        dataloaders (Dict[str, PromptDataLoader]): The dataloaders to use for the analysis.
        circuits (Dict[str, Circuit]): The circuits to analyze.
        save_dir (str): The directory to save the attention patterns.
        uid (str): The unique identifier for the attention patterns.
    """

    # Get shared attention head information
    attn_info_list = []
    for circuit in circuits.values():
        attn_heads = get_attention_heads(circuit)

        attn_info = [
            parse_attention_edge(head, edge_type, lbl)
            for head, edge_type, lbl in attn_heads
        ]

        attn_info_list.append(attn_info)

    attn_info, heads_origin = get_union_heads(attn_info_list)

    # Collect results for these attention heads
    result_dict = {}

    for variation, dataloader in dataloaders.items():
        variation_lit = cast(Literal["C", "A", "none", "both"], variation)
        token_labels, filtered_patterns = obtain_pattern_for_variation(
            model, model_name, dataloader, attn_info, variation_lit
        )
        result_dict[variation] = (
            token_labels,
            filtered_patterns,
            attn_info,
            heads_origin,
        )

    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{uid}"

    plot_attention_patterns(save_path, result_dict)


def obtain_pattern_for_variation(
    model: HookedTransformer,
    model_name: str,
    dataloader: PromptDataLoader,
    attn_info: List[Tuple[str, str, int, int]],
    variation: Literal["C", "A", "none", "both"],
) -> Tuple[List[List[str]], t.Tensor]:
    """
    Get the attention patterns for the attention heads in the circuit.

    Args:
        model (PatchableModel): The model to analyze.
        model_name (str): The name of the model.
        data_loader (PromptDataLoader): The data loader to use for the analysis.
        attn_info (List[Tuple[str, str, int, int]]): The attention head information.
        variation (Literal["C", "A", "none", "both"]): The variation used.

    Returns:
        Tuple[List[List[str], t.Tensor]: The token labels and the patterns.
    """

    patterns: List[List[t.Tensor]] = [[] for _ in range(len(attn_info))]

    for batch in tqdm(dataloader, desc=f"Obtaining patterns for {variation}"):
        batch = batch.corrupt if variation == "none" else batch.clean
        _, cache = model.run_with_cache(batch, return_type="logits", prepend_bos=False)

        for i, head in enumerate(attn_info):
            patterns[i].append(cache["pattern", head[2]].cpu()[:, head[3]])
        del cache

    concat_patterns = [t.cat(p).mean(dim=0) for p in patterns]
    stacked_patterns = t.stack(concat_patterns, dim=0)

    if "llama-3.2" in model_name.lower():
        seq_labels = ["[bos]"] + dataloader.seq_labels
    else:
        seq_labels = dataloader.seq_labels

    token_labels, filtered_patterns = filter_relevant_token_positions(
        seq_labels, stacked_patterns
    )

    return token_labels, filtered_patterns


def get_attention_heads(
    circuit: Circuit,
    attn_type: Literal["qkv", "out"] = "qkv",
) -> List[Tuple[Edge, Literal["src", "dest"], str]]:
    """
    Extract attention head information from circuit edges.

    Args:
        circuit (Circuit): circuit object with relevant circuit components.
        attn_type (Literal["qkv", "out"], optional): The type of attention to extract. Defaults to "qkv".

    Returns:
        List[Tuple[Edge, Literal["src", "dest"], str]]: List of tuples containing edges,
            their type (src or dest) and the seq_label for attention heads in the circuit.
    """
    attention_hook_types = (
        ["hook_k_input", "hook_v_input", "hook_q_input"]
        if attn_type == "qkv"
        else ["attn.hook_result"]
    )
    attn_heads: List[Tuple[Edge, Literal["src", "dest"], str]] = []

    for seq_label, comp in circuit.tok_pos_edges.items():
        for edge in comp:
            if attn_type == "qkv":
                is_in_circuit = any(
                    hook in edge.dest.module_name for hook in attention_hook_types
                )
            else:
                is_in_circuit = any(
                    hook in edge.src.module_name for hook in attention_hook_types
                )

            edge_type: Literal["src", "dest"] = "src"
            if is_in_circuit and attn_type == "out":
                attn_heads.append((edge, edge_type, seq_label))
            elif is_in_circuit:
                edge_type = "dest"
                attn_heads.append((edge, edge_type, seq_label))

    return attn_heads


def get_union_heads(
    attn_info_list: List[List[Tuple[str, str, int, int]]],
) -> Tuple[List[Tuple[str, str, int, int]], List[str]]:
    """
    Return the union of attention heads across two sets. Also return a list of origin labels:
    'C' if a head is only in attn_info_list[0], 'A' if it is only in attn_info_list[1],
    or 'all' if it appears in both.

    Args:
        attn_info_list (List[List[Tuple[str, str, int, int]]]): A list of two lists of attention head info.

    Returns:
        (union_heads, origins):
        union_heads: Sorted union of all heads.
        origins: Labels for each head in union_heads.
    """
    C_heads = set(attn_info_list[0])
    A_heads = set(attn_info_list[1])

    union_heads = C_heads.union(A_heads)
    union_heads_sorted = sorted(union_heads, key=lambda x: (x[2], x[3]))

    origins = []
    for head in union_heads_sorted:
        in_C_heads = head in C_heads
        in_A_heads = head in A_heads
        if in_C_heads and in_A_heads:
            origins.append("all")
        elif in_C_heads:
            origins.append("C")
        else:
            origins.append("A")

    return union_heads_sorted, origins


def filter_relevant_token_positions(
    seq_labels: List[str],
    patterns: t.Tensor,
) -> Tuple[List[List[str]], t.Tensor]:
    """
    Filter the relevant token positions.

    Args:
        seq_labels (List[str]): The labels for the tokens.
        patterns (t.Tensor): The attention patterns.

    Returns:
        Tuple[List[List[str], t.Tensor]: The filtered labels and patterns.
    """
    relevant_indices = [
        i for i, label in enumerate(seq_labels) if label in RELEVANT_TOKENS
    ]
    token_labels = [
        [seq_labels[i] for i in relevant_indices] for _ in range(patterns.size(0))
    ]
    patterns = patterns[:, relevant_indices][:, :, relevant_indices]
    return token_labels, patterns


def parse_attention_edge(
    edge: Edge, edge_type: Literal["src", "dest"], seq_label: str
) -> Tuple[str, str, int, int]:
    """
    Parse the edge data to get attention head information.

    Args:
        edge (Edge): edge with a src attention head.
        edge_type (Literal["src", "dest"]): whether the head is source or destination.
        seq_label (str): the sequence label for the attention head.

    Returns:
        Tuple[str, str, int, int]: tuple with module name plus layer and head index.
    """
    if edge_type == "src":
        module_name = edge.src.module_name
        edge_name = edge.name.split("->")[0]
    else:
        module_name = edge.dest.module_name
        edge_name = edge.name.split("->")[1]

    search_res = re.search(r"A(\d+)\.(\d+)", edge_name)

    if search_res is not None:
        layer, head = search_res.groups()
    else:
        raise ValueError(f"No valid heads found for {edge.name}!")

    return seq_label, module_name, int(layer), int(head)


def plot_attention_patterns(
    save_path: str,
    result_dict: Dict[
        str,
        Tuple[
            List[List[str]], List[t.Tensor], List[Tuple[str, str, int, int]], List[str]
        ],
    ],
) -> None:
    """
    Plot the attention patterns.

    Args:
        save_path (str): The path to save the attention patterns.
        result_dict (Dict[str, Tuple[List[List[str]], List[t.Tensor], List[Tuple[str, str, int, int]], List[str]]]): The token_labels,
            filtered_patterns, attn_info, and heads_origin for each variation.
    """
    total_heads = len(result_dict["C"][2])

    title_dict = {
        "C": "Incorrect Calculation Only",
        "A": "Incorrect Answer Only",
        "none": "Both Correct",
        "both": "Both Incorrect (same value)",
    }

    for idx in range(total_heads):
        layer = result_dict["C"][2][idx][2]
        head = result_dict["C"][2][idx][3]

        # Increased figure width and adjusted spacing
        fig, axes = plt.subplots(1, 4, figsize=(29, 5))
        plt.subplots_adjust(wspace=0.5, hspace=0.3, top=0.85, bottom=0.2)

        for i, (pert, (lbls, pats, info, origin)) in enumerate(result_dict.items()):
            labels = [re.sub(r"_occ_\d", "", lbl) for lbl in lbls[idx]]

            # First do the standard replacements
            labels = [
                label.replace("C", "result") if "C" in label else label
                for label in labels
            ]
            labels = [
                label.replace("A", "answer") if "A" in label else label
                for label in labels
            ]

            # Now apply the more specific replacements for better clarity
            display_labels = []
            for label in labels:
                if label == "[result-first]":
                    display_labels.append("[calculation-first-digit]")
                elif label == "[result-second]":
                    display_labels.append("[calculation-second-digit]")
                elif label == "[answer-first]":
                    display_labels.append("[answer-first-digit]")
                elif label == "[answer-second]":
                    display_labels.append("[answer-second-digit]")
                else:
                    display_labels.append(label)

            ax = axes[i]
            ax.imshow(pats[idx], cmap="plasma", vmin=0, vmax=1)
            ax.set_title(title_dict[pert], fontsize=16)
            ax.set_xticks(range(len(display_labels)))
            ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=12)
            ax.set_yticks(range(len(display_labels)))
            ax.set_yticklabels(display_labels, fontsize=12)

        cbar = fig.colorbar(ax.images[0], ax=axes.ravel().tolist())
        cbar.ax.tick_params(labelsize=14)
        plt.savefig(
            f"{save_path}_l{layer}h{head}_{origin[idx]}.png", bbox_inches="tight"
        )
        plt.close()


def visualize_cross_template_attention_patterns(
    model: HookedTransformer,
    model_name: str,
    all_templates_dataloaders: Dict[str, Dict[str, PromptDataLoader]],
    circuits: Dict[str, Circuit],
    save_dir: str,
    uid: str,
) -> None:
    """
    Visualize the attention patterns averaged across all templates.

    Args:
        model (HookedTransformer): The model to analyze.
        model_name (str): The name of the model.
        all_templates_dataloaders (Dict[str, Dict[str, PromptDataLoader]]): The dataloaders for all templates.
        circuits (Dict[str, Circuit]): The circuits to analyze.
        save_dir (str): The directory to save the attention patterns.
        uid (str): The unique identifier for the attention patterns.
    """

    # Get shared attention head information
    attn_info_list = []
    for circuit in circuits.values():
        attn_heads = get_attention_heads(circuit)
        attn_info = [
            parse_attention_edge(head, edge_type, lbl)
            for head, edge_type, lbl in attn_heads
        ]
        attn_info_list.append(attn_info)

    attn_info, heads_origin = get_union_heads(attn_info_list)

    # Initialize result_dict to store the aggregated patterns across templates
    result_dict = {}

    # For each variation type - using the same order as in dataloader creation
    for variation in ["C", "A", "both", "none"]:
        variation_lit = cast(Literal["C", "A", "none", "both"], variation)

        # Store patterns and labels from all templates
        template_data = []

        # Process each template
        for template, dataloaders in all_templates_dataloaders.items():
            if variation in dataloaders:
                # Get patterns for this template and variation
                token_labels, filtered_patterns = obtain_pattern_for_variation(
                    model,
                    model_name,
                    dataloaders[variation],
                    attn_info,
                    variation_lit,
                )

                # Store both token labels and patterns
                template_data.append((token_labels, filtered_patterns, template))

        # Skip if no data
        if not template_data:
            logging.warning(f"No data found for variation {variation}")
            continue

        # Find a common set of standardized labels
        # This is a heuristic approach - we count occurrences of each standardized label
        # across templates and use the most common set
        all_label_sets = []
        for token_labels, _, _ in template_data:
            # Extract the standardized part of each label (remove template-specific suffixes)
            std_labels = []
            for head_labels in token_labels:
                std_head_labels = [
                    re.sub(r"_occ_\d+", "", label) for label in head_labels
                ]
                std_labels.append(frozenset(std_head_labels))
            all_label_sets.append(std_labels)

        # Find the most common set of labels for each head
        common_labels = []
        for head_idx in range(len(all_label_sets[0])):
            label_counts = {}
            for template_labels in all_label_sets:
                label_set = template_labels[head_idx]
                label_set_tuple = tuple(sorted(label_set))
                label_counts[label_set_tuple] = label_counts.get(label_set_tuple, 0) + 1

            # Get the most common label set
            most_common_labels = max(label_counts.items(), key=lambda x: x[1])[0]
            common_labels.append(list(most_common_labels))

        logging.info(f"Found common label sets for variation {variation}")

        # Now we can average attention patterns from templates that have matching labels
        # We'll create a separate average for each head
        avg_patterns = []

        for head_idx in range(len(common_labels)):
            head_patterns = []
            head_labels = []
            # Collect patterns from templates with matching labels for this head
            for token_labels, filtered_patterns, template_name in template_data:
                std_head_labels = [
                    re.sub(r"_occ_\d+", "", label) for label in token_labels[head_idx]
                ]

                # Check if this template's labels match the common set for this head
                if set(std_head_labels) == set(common_labels[head_idx]):
                    head_patterns.append(filtered_patterns[head_idx])
                    head_labels.append(token_labels[head_idx])
                    logging.info(
                        f"Including template {template_name} for head {head_idx} (labels match)"
                    )
                else:
                    logging.info(
                        f"Skipping template {template_name} for head {head_idx} (labels don't match)"
                    )
                    logging.info(f"  Template labels: {std_head_labels}")
                    logging.info(f"  Common labels: {common_labels[head_idx]}")

            # Skip if no matching patterns
            if not head_patterns:
                logging.warning(f"No matching patterns for head {head_idx}")
                # Use placeholder data
                avg_patterns.append(t.zeros((1, 1)))
                continue

            # Ensure all patterns have the same shape for this head
            if not all(p.shape == head_patterns[0].shape for p in head_patterns):
                logging.warning(
                    f"Inconsistent shapes for head {head_idx}, using first pattern only"
                )
                avg_patterns.append(head_patterns[0])
                continue

            # Average the patterns for this head
            head_avg = sum(head_patterns) / len(head_patterns)
            avg_patterns.append(head_avg)

        # Use token labels from the first template with matching labels
        # We need to find at least one template that matched for each head
        token_labels = []
        for head_idx in range(len(common_labels)):
            head_token_labels = None
            for template_labels, _, _ in template_data:
                std_head_labels = [
                    re.sub(r"_occ_\d+", "", label)
                    for label in template_labels[head_idx]
                ]
                if set(std_head_labels) == set(common_labels[head_idx]):
                    head_token_labels = template_labels[head_idx]
                    break

            if head_token_labels is None:
                head_token_labels = ["unknown"] * avg_patterns[head_idx].shape[0]

            token_labels.append(head_token_labels)

        # Convert to tensor
        avg_patterns_tensor = t.stack(avg_patterns)

        # Store in result_dict
        result_dict[variation] = (
            token_labels,
            avg_patterns_tensor,
            attn_info,
            heads_origin,
        )

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{uid}"

    # Plot the averaged attention patterns
    plot_attention_patterns(save_path, result_dict)
