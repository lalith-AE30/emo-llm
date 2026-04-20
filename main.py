import os
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from LLMs.my_gemma2 import Gemma2ForCausalLM
from LLMs.my_llama import LlamaForCausalLM
from LLMs.my_mistral import MistralForCausalLM
from LLMs.my_olmo import OlmoForCausalLM
from LLMs.my_olmo2 import Olmo2ForCausalLM
from LLMs.my_phi3 import Phi3ForCausalLM
from prompt_manager import build_prompt, build_prompt_first_word_prediction
from utils import (
    Log,
    activation_patching as activation_patching_fn,
    apply_random_intervention_and_extract_logits,
    apply_zero_intervention_and_extract_logits,
    emotion_to_token_ids,
    extract_hidden_states as extract_hidden_states_fn,
    find_token_length_distribution,
    get_emotion_logits,
    hf_login,
    log_system_info,
    make_projections,
    probe_classification,
    probe_classification_non_linear,
    probe_regression,
    promote_vec,
    seed_everywhere,
    TextDataset,
)


app = typer.Typer(help="Run emotion and appraisal probing experiments.")


MODEL_NAMES = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "microsoft/Phi-3.5-mini-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
    "allenai/OLMo-1B-hf",
    "allenai/OLMo-7B-0724-Instruct-hf",
    "mistralai/Ministral-8B-Instruct-2410",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
]

MODEL_SHORT_NAMES = [
    "Llama3.2_1B",
    "Llama3.1_8B",
    "gemma-2-2b-it",
    "gemma-2-9b-it",
    "Phi-3.5-mini-instruct",
    "Phi-3-medium-128k-instruct",
    "OLMo-1B-hf",
    "OLMo-7B-0724-Instruct-hf",
    "Ministral-8B-Instruct-2410",
    "Mistral-Nemo-Instruct-2407",
    "OLMo-2-1124-7B-Instruct",
    "OLMo-2-1124-13B-Instruct",
]

MODEL_CLASSES = [
    LlamaForCausalLM,
    LlamaForCausalLM,
    Gemma2ForCausalLM,
    Gemma2ForCausalLM,
    Phi3ForCausalLM,
    Phi3ForCausalLM,
    OlmoForCausalLM,
    OlmoForCausalLM,
    MistralForCausalLM,
    MistralForCausalLM,
    Olmo2ForCausalLM,
    Olmo2ForCausalLM,
]


def run_open_vocab_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    id_to_emotion: Dict[int, str],
    labels: torch.Tensor,
    model_short_name: str,
    save_prefix: str,
    logger,
) -> None:
    """Run free-form next-token emotion prediction and save text outputs.

    This experiment queries the model without constraining output tokens to the
    predefined emotion label set, decodes the resulting token predictions into
    strings, and stores both ground-truth labels and decoded predictions.

    Side effects:
        Writes a ``.pt`` file containing ``[ground_truth_labels, predictions]``
        to the model-specific directory under ``outputs/``.
    """
    logger.info(
        "------------------------------ Generating Open Vocab Predictions ------------------------------"
    )
    open_vocab_preds = get_emotion_logits(
        dataloader, tokenizer, model, ids_to_pick=None, apply_argmax=True
    )
    preds = tokenizer.batch_decode(open_vocab_preds, skip_special_tokens=True)
    gt = [" " + id_to_emotion[i.item()] for i in labels[:, 0]]

    torch.save(
        [gt, preds],
        f"outputs/{model_short_name}/{save_prefix}open_vocab_predictions.pt",
    )


def run_in_domain_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    emotions_to_tokenized_ids,
    labels: torch.Tensor,
    id_to_emotion: Dict[int, str],
    train_data: pd.DataFrame,
    model_short_name: str,
    save_prefix: str,
    logger,
) -> None:
    """Run constrained emotion classification and filter the training set.

    The model is evaluated only over the token IDs corresponding to the
    in-domain emotion labels. Predicted labels are compared against dataset
    labels, mismatches are logged, and a filtered dataset containing only
    correctly predicted samples is saved for downstream probing/interventions.

    Side effects:
        Saves raw argmax-vs-label tensors and ``filtered_train_data.csv`` under
        the model-specific ``outputs/`` directory.
    """
    logger.info(
        "------------------------------ Generating In-Domain Emotion Labels using Logits ------------------------------"
    )
    probs = get_emotion_logits(
        dataloader, tokenizer, model, ids_to_pick=emotions_to_tokenized_ids
    )

    torch.save(
        [probs.argmax(dim=1), labels[:, 0]],
        f"outputs/{model_short_name}/{save_prefix}unfiltered_emotion_logits.pt",
    )

    emotion_labels = [id_to_emotion[i.item()] for i in probs.argmax(dim=1)]
    filtered_train_data = train_data[train_data["emotion"] == emotion_labels]
    logger.info(
        f"Total mismatched labels: {len(train_data) - len(filtered_train_data)}/{len(train_data)} ({1 - len(filtered_train_data) / len(train_data):.2%})"
    )
    filtered_train_data.to_csv(
        f"outputs/{model_short_name}/filtered_train_data.csv", index=False
    )


def run_bbox_emotion_regression_experiment(
    labels: torch.Tensor,
    emotion_to_id: Dict[str, int],
    appraisals_to_id: Dict[str, int],
    model_short_name: str,
    logger,
) -> None:
    """Train a black-box classifier from appraisal annotations to emotions.

    Uses appraisal columns as input features and emotion IDs as targets,
    reports train/test accuracy, and stores learned linear parameters for later
    analysis and reproducibility.

    Side effects:
        Writes a checkpoint with label mappings, labels, classifier weights,
        and biases to ``outputs/<model_short_name>/emotion_appraisal_labels.pt``.
    """
    logger.info(
        "------------------------------ Black-box appraisal emotion regression ------------------------------"
    )

    reg_input = labels[:, 1:]
    reg_labels = labels[:, 0]
    r = probe_classification(reg_input, reg_labels, return_weights=True)
    w = r["weights"]
    b = r["bias"]

    print(
        f"Emotion Prediction Acc: train {r['accuracy_train']}, test {r['accuracy_test']}"
    )

    torch.save(
        {
            "emotion_to_id": emotion_to_id,
            "appraisals_to_id": appraisals_to_id,
            "labels": labels,
            "weights": w,
            "bias": b,
        },
        f"outputs/{model_short_name}/emotion_appraisal_labels.pt",
    )


def run_save_clean_logits_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    emotions_to_tokenized_ids,
    model_short_name: str,
    logger,
) -> None:
    """Compute and save baseline (no intervention) emotion logits.

    Produces the reference logits over the emotion-token subset that are later
    used as a control condition when quantifying intervention effects.

    Side effects:
        Saves logits tensor to ``outputs/<model_short_name>/original_logits.pt``.
    """
    logger.info(
        "------------------------------ Saving Original Clean Logits ------------------------------"
    )
    original_logits = get_emotion_logits(
        dataloader, tokenizer, model, ids_to_pick=emotions_to_tokenized_ids
    )
    torch.save(original_logits, f"outputs/{model_short_name}/original_logits.pt")


def extract_hidden_states_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    logger,
) -> torch.Tensor:
    """Extract hidden states over a fixed probing configuration.

    Captures hidden representations for all layers and selected extraction
    locations/tokens used across probing and intervention experiments.

    Returns:
        Concatenated hidden-state tensor indexed by
        ``[sample, layer, location, token, hidden_dim]``.
    """
    logger.info(
        "------------------------------ Extracting Hidden States ------------------------------"
    )
    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    return extract_hidden_states_fn(
        dataloader,
        tokenizer,
        model,
        logger,
        extraction_locs=extraction_locs,
        extraction_layers=extraction_layers,
        extraction_tokens=extraction_tokens,
    )


def _load_hidden_states(
    model_short_name: str,
    extraction_layers: List[int],
    extraction_locs: List[int],
    extraction_tokens: List[int],
    missing_flag_name: str,
) -> torch.Tensor:
    """Load hidden states from disk using the naming convention expected by probes."""
    try:
        return torch.load(
            f"outputs/{model_short_name}/hidden_states_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
            weights_only=False,
        )
    except FileNotFoundError as exc:
        raise Exception(
            f"Hidden states not found, run the code again with {missing_flag_name}"
        ) from exc


def run_emotion_probing_experiment(
    labels: torch.Tensor,
    model,
    model_short_name: str,
    extract_hidden_states_enabled: bool,
    all_hidden_states: Optional[torch.Tensor],
    logger,
) -> torch.Tensor:
    """Run linear probes that predict emotion labels from hidden states.

    For each layer/location/token slice, fits a linear classifier from hidden
    activations to emotion IDs and stores probe metrics/parameters.
    Hidden states are either reused from memory or loaded from disk.

    Returns:
        The hidden-state tensor used for probing.

    Side effects:
        Saves probing results to
        ``emotion_probing_layers_<...>_locs_<...>_tokens_<...>.pt``.
    """
    logger.info(
        "--------------------------------- Emotion Probing ---------------------------------"
    )

    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    if not extract_hidden_states_enabled:
        all_hidden_states = _load_hidden_states(
            model_short_name,
            extraction_layers,
            extraction_locs,
            extraction_tokens,
            "--extract-hidden-states",
        )

    size_on_memory = all_hidden_states.element_size() * all_hidden_states.numel()
    logger.info(f"Hidden states tensor size: {size_on_memory / (1024**2):.2f} MB")

    results = {}
    for i, layer in tqdm(enumerate(extraction_layers), total=len(extraction_layers)):
        results[layer] = {}
        for j, loc in enumerate(extraction_locs):
            results[layer][loc] = {}
            for k, token in enumerate(extraction_tokens):
                results[layer][loc][token] = probe_classification(
                    all_hidden_states[:, i, j, k], labels[:, 0], return_weights=True
                )

    torch.save(
        results,
        f"outputs/{model_short_name}/emotion_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )
    return all_hidden_states


def run_emotion_probing_non_linear_experiment(
    labels: torch.Tensor,
    model,
    model_short_name: str,
    extract_hidden_states_enabled: bool,
    all_hidden_states: Optional[torch.Tensor],
    logger,
) -> torch.Tensor:
    """Run non-linear emotion probes across layer/location/token slices.

    Mirrors the linear probing pipeline but uses a non-linear probe model to
    estimate emotion decodability from hidden activations.

    Returns:
        The hidden-state tensor used for probing.

    Side effects:
        Saves non-linear probe outputs to
        ``non_linaer_emotion_probing_layers_<...>_locs_<...>_tokens_<...>.pt``.
    """
    logger.info(
        "--------------------------------- Emotion Non-Linear Probing ---------------------------------"
    )

    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1, -2, -3, -4, -5]
    extraction_layers = list(range(model.config.num_hidden_layers))

    if not extract_hidden_states_enabled:
        all_hidden_states = _load_hidden_states(
            model_short_name,
            extraction_layers,
            extraction_locs,
            extraction_tokens,
            "--extract-hidden-states",
        )

    size_on_memory = all_hidden_states.element_size() * all_hidden_states.numel()
    logger.info(f"Hidden states tensor size: {size_on_memory / (1024**2):.2f} MB")

    results = {}
    for i, layer in tqdm(enumerate(extraction_layers), total=len(extraction_layers)):
        results[layer] = {}
        for j, loc in enumerate(extraction_locs):
            results[layer][loc] = {}
            for k, token in enumerate(extraction_tokens):
                results[layer][loc][token] = probe_classification_non_linear(
                    all_hidden_states[:, i, j, k], labels[:, 0], return_weights=True
                )

    torch.save(
        results,
        f"outputs/{model_short_name}/non_linaer_emotion_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )
    return all_hidden_states


def run_appraisal_probing_experiment(
    labels: torch.Tensor,
    appraisals: List[str],
    appraisals_to_id: Dict[str, int],
    model,
    model_short_name: str,
    extract_hidden_states_enabled: bool,
    all_hidden_states: Optional[torch.Tensor],
    logger,
) -> torch.Tensor:
    """Fit appraisal-specific regression probes over hidden representations.

    For each appraisal target and each layer/location/token slice, trains a
    regression probe to predict scalar appraisal values from hidden states.
    This quantifies where appraisal information is represented in the model.

    Returns:
        The hidden-state tensor used for probing.

    Side effects:
        Persists a nested results dictionary for all appraisals and positions.
    """
    logger.info(
        "------------------------------ Appraisal Probing ------------------------------"
    )

    results = {}
    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]
    extraction_layers = list(range(model.config.num_hidden_layers))

    if not extract_hidden_states_enabled:
        all_hidden_states = _load_hidden_states(
            model_short_name,
            extraction_layers,
            extraction_locs,
            extraction_tokens,
            "--extract-hidden-states",
        )

    for app_ in tqdm(appraisals, total=len(appraisals)):
        results[app_] = {}
        for i, layer in tqdm(
            enumerate(extraction_layers), total=len(extraction_layers)
        ):
            results[app_][layer] = {}
            for j, loc in enumerate(extraction_locs):
                results[app_][layer][loc] = {}
                for k, token in enumerate(extraction_tokens):
                    results[app_][layer][loc][token] = probe_regression(
                        all_hidden_states[:, i, j, k],
                        labels[:, appraisals_to_id[app_] + 1],
                        return_weights=True,
                    )

    torch.save(
        results,
        f"outputs/{model_short_name}/appraisal_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )
    return all_hidden_states


def run_extract_weights_experiment(
    model,
    model_short_name: str,
    emotions_list: List[str],
    appraisals: List[str],
) -> None:
    """Collect probe parameters into dense tensors for downstream interventions.

    Loads previously saved emotion/appraisal probing results, extracts each
    probe's learned weight and bias, and packs them into consistent tensor
    layouts aligned by label, layer, location, and token index.

    Side effects:
        Writes four tensor files: emotion weights/biases and appraisal
        weights/biases under the model-specific ``outputs/`` directory.
    """
    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]
    extraction_layers = list(range(model.config.num_hidden_layers))

    try:
        emotion_probing_results = torch.load(
            f"outputs/{model_short_name}/emotion_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
            weights_only=False,
        )
    except FileNotFoundError as exc:
        raise Exception(
            "Emotion probing results not found, run the code again with --emotion-probing enabled"
        ) from exc

    emotions_weights_ = torch.zeros(
        [
            len(emotions_list),
            len(extraction_layers),
            len(extraction_locs),
            len(extraction_tokens),
            model.config.hidden_size,
        ]
    )
    print(emotions_weights_.shape)
    emotions_biases_ = torch.zeros(
        [
            len(emotions_list),
            len(extraction_layers),
            len(extraction_locs),
            len(extraction_tokens),
        ]
    )

    for j, layer in enumerate(extraction_layers):
        for k, loc in enumerate(extraction_locs):
            for tok_idx, token in enumerate(extraction_tokens):
                emotions_weights_[:, j, k, tok_idx, :] = torch.from_numpy(
                    emotion_probing_results[layer][loc][token]["weights"]
                )
                emotions_biases_[:, j, k, tok_idx] = torch.from_numpy(
                    emotion_probing_results[layer][loc][token]["bias"]
                )

    torch.save(
        emotions_weights_,
        f"outputs/{model_short_name}/emotions_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )

    torch.save(
        emotions_biases_,
        f"outputs/{model_short_name}/emotions_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )

    try:
        appraisal_probing_results = torch.load(
            f"outputs/{model_short_name}/appraisal_probing_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
            weights_only=False,
        )
    except FileNotFoundError as exc:
        raise Exception(
            "Appraisal probing results not found, run the code again with --appraisal-probing enabled"
        ) from exc

    appraisals_weights = torch.zeros(
        [
            len(appraisals),
            len(extraction_layers),
            len(extraction_locs),
            len(extraction_tokens),
            model.config.hidden_size,
        ]
    )
    appraisals_biases = torch.zeros(
        [
            len(appraisals),
            len(extraction_layers),
            len(extraction_locs),
            len(extraction_tokens),
        ]
    )

    for i, app_ in enumerate(appraisals):
        for j, layer in enumerate(extraction_layers):
            for k, loc in enumerate(extraction_locs):
                for tok_idx, token in enumerate(extraction_tokens):
                    appraisals_weights[i, j, k, tok_idx, :] = torch.from_numpy(
                        appraisal_probing_results[app_][layer][loc][token]["weights"]
                    )
                    appraisals_biases[i, j, k, tok_idx] = torch.from_numpy(
                        appraisal_probing_results[app_][layer][loc][token]["bias"]
                    )

    torch.save(
        appraisals_weights,
        f"outputs/{model_short_name}/appraisals_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )
    torch.save(
        appraisals_biases,
        f"outputs/{model_short_name}/appraisals_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )


def run_zero_intervention_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    logger,
    emotions_to_tokenized_ids,
    model_short_name: str,
) -> None:
    """Ablate activations by zeroing selected positions over sliding layer spans.

    For each center layer, applies zero interventions over a local span and
    selected extraction locations/tokens, then records the resulting emotion
    logits to measure sensitivity to removed activations.

    Side effects:
        Saves intervention outputs per location/token configuration to
        ``zero_intervention_results_*.pt`` files.
    """
    logger.info(
        "------------------------------ Performing Zero Intervention ------------------------------"
    )

    span = 3
    intervention_layers = list(
        range(span // 2, model.config.num_hidden_layers - span // 2)
    )
    intervention_locs = [[3], [6], [7]]

    for intervention_tokens in [[-1], [-2], [-3], [-4], [-5]]:
        for loc in intervention_locs:
            intervention_results = {}
            for center_layer in tqdm(
                intervention_layers, total=len(intervention_layers)
            ):
                layers = list(
                    range(center_layer - span // 2, center_layer + span // 2 + 1)
                )
                intervention_results[center_layer] = (
                    apply_zero_intervention_and_extract_logits(
                        dataloader,
                        tokenizer,
                        model,
                        logger,
                        intervention_layers=layers,
                        intervention_locs=loc,
                        intervention_tokens=intervention_tokens,
                        ids_to_pick=emotions_to_tokenized_ids,
                    )
                )

            torch.save(
                intervention_results,
                f"outputs/{model_short_name}/zero_intervention_results_layers_{intervention_layers}_span_{span}_locs_{loc}_token_{intervention_tokens}.pt",
            )


def run_random_intervention_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    logger,
    emotions_to_tokenized_ids,
    model_short_name: str,
) -> None:
    """Inject random perturbations into selected activations and log effects.

    Uses a fixed RNG seed for reproducibility, applies random interventions
    over sliding layer spans, and stores resulting emotion logits for
    comparison with clean and zero-ablation baselines.

    Side effects:
        Writes seeded random intervention results to
        ``random_intervention_results_*.pt``.
    """
    logger.info(
        "------------------------------ Performing Random Intervention ------------------------------"
    )

    span = 3
    seed = 100
    seed_everywhere(seed)

    intervention_layers = list(
        range(span // 2, model.config.num_hidden_layers - span // 2)
    )
    intervention_locs = [[3], [6], [7]]

    for intervention_tokens in [[-1]]:
        for loc in intervention_locs:
            intervention_results = {}
            for center_layer in tqdm(
                intervention_layers, total=len(intervention_layers)
            ):
                layers = list(
                    range(center_layer - span // 2, center_layer + span // 2 + 1)
                )
                intervention_results[center_layer] = (
                    apply_random_intervention_and_extract_logits(
                        dataloader,
                        tokenizer,
                        model,
                        logger,
                        intervention_layers=layers,
                        intervention_locs=loc,
                        intervention_tokens=intervention_tokens,
                        ids_to_pick=emotions_to_tokenized_ids,
                    )
                )

            torch.save(
                intervention_results,
                f"outputs/{model_short_name}/random_intervention_results_layers_{intervention_layers}_span_{span}_locs_{loc}_token_{intervention_tokens}_seed_{seed}.pt",
            )


def run_activation_patching_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    logger,
    emotions_to_tokenized_ids,
    model_short_name: str,
    train_data: pd.DataFrame,
) -> None:
    """Run source-target activation patching across layers and positions.

    Repeatedly samples source and target sentences with different emotions,
    patches selected internal activations from source into target across a
    sliding layer span, and records resulting emotion logits.

    Side effects:
        Saves per-layer/per-trial patching outcomes to
        ``patching_results_layers_*.pt``.
    """
    logger.info(
        "------------------------------ Performing Activation Patching ------------------------------"
    )
    num_experiments = 200
    span = 5
    layers_ = list(range(span // 2, model.config.num_hidden_layers - span // 2))

    for locs in [[3], [6], [7]]:
        for intervention_tokens in [[-1]]:
            patching_results = {}
            for center_layer in tqdm(layers_, total=len(layers_)):
                layers = list(
                    range(center_layer - span // 2, center_layer + span // 2 + 1)
                )

                patching_results[center_layer] = {}
                for i in range(num_experiments):
                    source_sentence = train_data.sample(1)
                    target_sentence = train_data.sample(1)
                    while (
                        source_sentence["emotion"].values[0]
                        == target_sentence["emotion"].values[0]
                    ):
                        target_sentence = train_data.sample(1)

                    source_sentence = source_sentence["input_text"].values[0]
                    target_sentence = target_sentence["input_text"].values[0]

                    patching_results[center_layer][i] = activation_patching_fn(
                        source_sentence,
                        target_sentence,
                        tokenizer,
                        model,
                        logger,
                        intervention_layers=layers,
                        intervention_locs=locs,
                        ids_to_pick=emotions_to_tokenized_ids,
                        intervention_tokens=intervention_tokens,
                    )

            torch.save(
                patching_results,
                f"outputs/{model_short_name}/patching_results_layers_{layers_}_locs_{locs}_span_{span}_token_{intervention_tokens}.pt",
            )


def run_attention_weights_experiment(
    dataset: TextDataset,
    tokenizer: AutoTokenizer,
    model,
    logger,
    model_short_name: str,
) -> None:
    """Extract attention-adjacent tensors with batch size 1 for inspection.

    Runs hidden-state extraction in a configuration that returns tokenized
    inputs and non-concatenated outputs suitable for attention-level analysis
    and per-example diagnostics.

    Side effects:
        Saves extracted tensors to
        ``attention_weights_layers_<...>_locs_<...>_tokens_<...>.pt``.
    """
    logger.info(
        "------------------------------ Extracting Attention Weights ------------------------------"
    )

    dataloader_1bs = DataLoader(dataset, batch_size=1, shuffle=False)
    extraction_layers = list(range(model.config.num_hidden_layers))
    extraction_locs = [10]
    extraction_tokens = [-1]
    results = extract_hidden_states_fn(
        dataloader_1bs,
        tokenizer,
        model,
        logger,
        extraction_locs=extraction_locs,
        extraction_layers=extraction_layers,
        extraction_tokens=extraction_tokens,
        do_final_cat=False,
        return_tokenized_input=True,
    )
    torch.save(
        results,
        f"outputs/{model_short_name}/attention_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
    )


def load_weights_for_promotion_experiments(
    model,
    model_short_name: str,
    logger,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[int],
    List[int],
    List[int],
]:
    """Load probe parameter tensors required by intervention-style experiments.

    Reads previously exported emotion and appraisal probe weights/biases and
    returns them together with extraction index metadata used to slice tensors
    consistently in promotion/surgery routines.

    Returns:
        Tuple containing emotion weights, emotion biases, appraisal weights,
        appraisal biases, extraction locations, extraction tokens, and
        extraction layers.
    """
    logger.info(
        "------------------------------ Loading Weights ------------------------------"
    )
    extraction_locs = [3, 6, 7]
    extraction_tokens = [-1]
    extraction_layers = list(range(model.config.num_hidden_layers))
    emotions_weights_ = torch.load(
        f"outputs/{model_short_name}/emotions_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
        weights_only=False,
    )
    emotions_biases_ = torch.load(
        f"outputs/{model_short_name}/emotions_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
        weights_only=False,
    )

    appraisals_weights_ = torch.load(
        f"outputs/{model_short_name}/appraisals_weights_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
        weights_only=False,
    )
    appraisals_biases_ = torch.load(
        f"outputs/{model_short_name}/appraisals_biases_layers_{extraction_layers}_locs_{extraction_locs}_tokens_{extraction_tokens}.pt",
        weights_only=False,
    )
    return (
        emotions_weights_,
        emotions_biases_,
        appraisals_weights_,
        appraisals_biases_,
        extraction_locs,
        extraction_tokens,
        extraction_layers,
    )


def run_emotion_promotion_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    logger,
    emotions_to_tokenized_ids,
    model_short_name: str,
    emotions_weights_: torch.Tensor,
    extraction_locs: List[int],
    extraction_tokens: List[int],
    extraction_layers: List[int],
    emotions_list: List[str],
) -> None:
    """Promote emotion probe directions and evaluate induced logit shifts.

    Builds projection matrices from probe vectors, constructs a promotion
    vector for each target emotion, applies interventions over sliding layer
    spans, and stores resulting emotion logits.

    Side effects:
        Creates ``outputs/<model_short_name>/emotion_promotion/`` and writes
        one results file per promoted emotion/location configuration.
    """
    logger.info(
        "------------------------------ Emotion Promotion ------------------------------"
    )
    os.makedirs(f"outputs/{model_short_name}/emotion_promotion/", exist_ok=True)

    promotion_tokens = [-1]
    promotion_locs = [[7]]
    span = 3
    layers_ = list(range(span // 2, model.config.num_hidden_layers - span // 2))

    beta1 = 1.0
    beta2 = 0.0
    do_normalization = False

    emotions_weights = emotions_weights_[
        :,
        :,
        :,
        [-1]
        if promotion_tokens == "all"
        else [extraction_tokens.index(token) for token in promotion_tokens],
    ]

    for emotion_to_promote in ["anger", "joy", "pride", "fear", "guilt", "sadness"]:
        for locs in promotion_locs:
            print(
                f"================= Promoting emotion: {emotion_to_promote} at loc: {locs}, Beta1: {beta1}, Beta2: {beta2}, Normalization: {do_normalization}"
            )

            results = {}
            for center_layer in tqdm(layers_, total=len(layers_)):
                layers_with_span = list(
                    range(center_layer - span // 2, center_layer + span // 2 + 1)
                )

                ews = emotions_weights[
                    :, [extraction_layers.index(layer) for layer in layers_with_span]
                ][:, :, [extraction_locs.index(loc) for loc in locs]]

                projection = make_projections(ews.permute(1, 2, 3, 0, 4))
                ew = ews[emotions_list.index(emotion_to_promote)]

                if do_normalization:
                    ew = ew / torch.norm(ew, dim=-1, keepdim=True)

                ew = beta1 * ew

                results[center_layer], _ = promote_vec(
                    dataloader,
                    tokenizer,
                    model,
                    logger,
                    prom_vector=ew,
                    projection_matrix=projection,
                    Beta=beta2,
                    promotion_layers=layers_with_span,
                    promotion_locs=locs,
                    promotion_tokens=promotion_tokens,
                    ids_to_pick=emotions_to_tokenized_ids,
                )

            torch.save(
                results,
                f"outputs/{model_short_name}/emotion_promotion/emotion_promotion_{emotion_to_promote}_layers_{layers_}_span_{span}_locs_{locs}_tokens_{promotion_tokens}_Beta1_{beta1}_Beta2_{beta2}_normalization_{do_normalization}.pt",
            )


def run_appraisal_surgery_experiment(
    dataloader: DataLoader,
    tokenizer: AutoTokenizer,
    model,
    logger,
    emotions_to_tokenized_ids,
    model_short_name: str,
    appraisals_weights_: torch.Tensor,
    extraction_locs: List[int],
    extraction_tokens: List[int],
    extraction_layers: List[int],
    appraisals_to_id: Dict[str, int],
) -> None:
    """Execute appraisal-surgery interventions from predefined recipes.

    Each recipe specifies appraisals to change (with signed coefficients) and
    optionally appraisals to hold fixed. For each layer span/location, the
    routine constructs intervention vectors from appraisal probe directions,
    applies them, and records emotion-logit outcomes.

    Side effects:
        Creates ``outputs/<model_short_name>/appraisal_surgery/`` and writes
        one file per recipe/location configuration.
    """
    logger.info(
        "------------------------------ Appraisal Surgery ------------------------------"
    )
    os.makedirs(f"outputs/{model_short_name}/appraisal_surgery/", exist_ok=True)

    surgery_appraisals = [
        (["pleasantness"], [], [+1]),
        (["pleasantness"], [], [-1]),
        (["other_responsblt"], [], [+1]),
        (["other_responsblt"], [], [-1]),
        (
            ["pleasantness"],
            ["predict_event", "other_responsblt", "chance_control"],
            [+1],
        ),
        (
            ["other_responsblt"],
            ["predict_event", "pleasantness", "chance_control"],
            [+1],
        ),
        (
            ["predict_event"],
            ["pleasantness", "other_responsblt", "chance_control"],
            [+1],
        ),
        (
            ["chance_control"],
            ["pleasantness", "other_responsblt", "predict_event"],
            [+1],
        ),
        (
            ["pleasantness"],
            ["predict_event", "other_responsblt", "chance_control"],
            [-1],
        ),
        (
            ["other_responsblt"],
            ["predict_event", "pleasantness", "chance_control"],
            [-1],
        ),
        (
            ["predict_event"],
            ["pleasantness", "other_responsblt", "chance_control"],
            [-1],
        ),
        (
            ["chance_control"],
            ["pleasantness", "other_responsblt", "predict_event"],
            [-1],
        ),
        (
            ["pleasantness", "other_responsblt"],
            ["predict_event", "chance_control"],
            [+1, -1],
        ),
        (
            ["pleasantness", "other_responsblt"],
            ["predict_event", "chance_control"],
            [-1, -1],
        ),
        (
            ["pleasantness", "other_responsblt"],
            ["predict_event", "chance_control"],
            [+1, +1],
        ),
        (
            ["pleasantness", "other_responsblt"],
            ["predict_event", "chance_control"],
            [-1, +1],
        ),
        (
            ["pleasantness", "predict_event"],
            ["other_responsblt", "chance_control"],
            [+1, -1],
        ),
        (
            ["pleasantness", "predict_event"],
            ["other_responsblt", "chance_control"],
            [-1, -1],
        ),
        (
            ["pleasantness", "predict_event"],
            ["other_responsblt", "chance_control"],
            [+1, +1],
        ),
        (
            ["pleasantness", "predict_event"],
            ["other_responsblt", "chance_control"],
            [-1, +1],
        ),
        (["random_42"], [], [+1]),
        (["random_43"], [], [+1]),
        (["random_44"], [], [+1]),
    ]

    promotion_tokens = [-1]
    promotion_locs = [[7]]
    beta1 = 1.0
    beta2 = 0.0
    do_normalization = True

    span = 3
    layers_ = list(range(span // 2, model.config.num_hidden_layers - span // 2))

    appraisals_weights = appraisals_weights_[
        :,
        :,
        :,
        [-1]
        if promotion_tokens == "all"
        else [extraction_tokens.index(token) for token in promotion_tokens],
    ]

    for appraisals_to_change_, appraisals_to_fix_, coeffs_ in surgery_appraisals:
        logger.info(
            f"================= Changing appraisal: {appraisals_to_change_} ({coeffs_}) loc {promotion_locs} with Beta1: {beta1}, Beta2: {beta2}, Normalization: {do_normalization}, while keeping {appraisals_to_fix_} fixed"
        )

        assert len(appraisals_to_change_) > 0, (
            "At least one appraisal should be changed"
        )
        assert set(appraisals_to_change_).isdisjoint(set(appraisals_to_fix_)), (
            "Appraisals to change and fix should be disjoint"
        )

        for locs in promotion_locs:
            results = {}
            for center_layer in tqdm(layers_, total=len(layers_)):
                layers_with_span = list(
                    range(center_layer - span // 2, center_layer + span // 2 + 1)
                )
                aws = appraisals_weights[
                    :, [extraction_layers.index(layer) for layer in layers_with_span]
                ][:, :, [extraction_locs.index(loc) for loc in locs]]

                changing_appraisals_weights = []
                for _, appraisals_to_change in enumerate(appraisals_to_change_):
                    if "random" in appraisals_to_change:
                        seed = int(appraisals_to_change.split("_")[-1])
                        seed_everywhere(seed)
                        appraisals_to_change_weight = torch.randn_like(aws[0]) * aws[
                            0
                        ].norm(dim=-1, keepdim=True)
                        changing_appraisals_weights.append(appraisals_to_change_weight)
                    else:
                        appraisals_to_change_index = appraisals_to_id[
                            appraisals_to_change
                        ]
                        appraisals_to_change_weight = aws[appraisals_to_change_index]
                        changing_appraisals_weights.append(appraisals_to_change_weight)

                changing_appraisals_weights = torch.stack(
                    changing_appraisals_weights, dim=0
                )
                changing_appraisals = changing_appraisals_weights.permute(1, 2, 3, 0, 4)

                fixed_appraisals_weights = []
                for _, appraisals_to_fix in enumerate(appraisals_to_fix_):
                    if "random" in appraisals_to_fix:
                        seed = int(appraisals_to_fix.split("_")[-1])
                        seed_everywhere(seed)
                        appraisals_to_fix_weight = torch.randn_like(aws[0]) * aws[
                            0
                        ].norm(dim=-1, keepdim=True)
                        fixed_appraisals_weights.append(appraisals_to_fix_weight)
                    else:
                        appraisals_to_fix_index = appraisals_to_id[appraisals_to_fix]
                        appraisals_to_fix_weight = aws[appraisals_to_fix_index]
                        fixed_appraisals_weights.append(appraisals_to_fix_weight)

                if len(fixed_appraisals_weights) > 0:
                    fixed_appraisals_weights = torch.stack(
                        fixed_appraisals_weights, dim=0
                    )
                    fixed_appraisals = fixed_appraisals_weights.permute(1, 2, 3, 0, 4)
                    p_fixed = make_projections(fixed_appraisals)
                else:
                    p_fixed = torch.zeros(
                        [
                            len(layers_with_span),
                            len(locs),
                            len(promotion_tokens),
                            model.config.hidden_size,
                            model.config.hidden_size,
                        ]
                    )

                identity_proj = (
                    torch.eye(p_fixed.size(-1))
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .broadcast_to(p_fixed.size())
                )

                coeffs = torch.tensor(coeffs_).to(torch.float32)
                if len(coeffs.shape) == 0:
                    coeffs = coeffs.unsqueeze(0)
                prom_vector = torch.einsum(
                    "ijkal, a -> ijkl", changing_appraisals, coeffs
                )

                if do_normalization:
                    prom_vector = prom_vector / torch.norm(
                        prom_vector, dim=-1, keepdim=True
                    )

                prom_vector = beta1 * prom_vector

                results[center_layer], _ = promote_vec(
                    dataloader,
                    tokenizer,
                    model,
                    logger,
                    prom_vector=prom_vector,
                    projection_matrix=identity_proj,
                    Beta=beta2,
                    promotion_layers=layers_with_span,
                    promotion_locs=locs,
                    promotion_tokens=promotion_tokens,
                    ids_to_pick=emotions_to_tokenized_ids,
                )

            appraisals_to_change = "_".join(appraisals_to_change_)
            appraisals_to_fix = "_".join(appraisals_to_fix_)
            coeffs = "_".join([str(c) for c in coeffs_])

            torch.save(
                results,
                f"outputs/{model_short_name}/appraisal_surgery/appraisal_surgery_{appraisals_to_change}_fixed_{appraisals_to_fix}_coeffs_{coeffs}_layers_{layers_}_span_{span}_locs_{locs}_tokens_{promotion_tokens}_Beta1_{beta1}_Beta2_{beta2}_normalization_{do_normalization}.pt",
            )


def run_experiments(
    model_index: int,
    bs: int,
    prompt_type: str,
    task_type: str,
    open_vocab: bool,
    in_domain: bool,
    bbox_emotion_regression: bool,
    save_clean_logits: bool,
    extract_hidden_states_enabled: bool,
    emotion_probing: bool,
    emotion_probing_non_linear: bool,
    appraisal_probing: bool,
    extract_weights: bool,
    zero_intervention: bool,
    random_intervention: bool,
    activation_patching_enabled: bool,
    attention_weights: bool,
    emotion_promotion: bool,
    appraisal_surgery: bool,
    clean_probings: bool,
) -> None:
    """Run the end-to-end experiment pipeline for one model/task setup.

    This orchestration function prepares data/prompts, loads tokenizer/model,
    computes label/token mappings, and conditionally executes each experiment
    stage controlled by CLI flags (prediction, filtering, probing, weight
    extraction, and intervention families).

    Side effects:
        Produces logs and writes multiple artifacts under ``outputs/`` based on
        the enabled stages. Raises descriptive exceptions when prerequisite
        artifacts for a requested stage are missing.
    """
    log = Log(log_name="intervention")
    logger = log.logger
    log_system_info(logger)
    hf_login(logger)

    batch_size = bs
    hooked = True
    device_map = "cuda"
    save_prefix = ""

    os.makedirs("outputs/", exist_ok=True)

    def resolve_task_configuration(
        task_type: str, prompt_type: str, train_data: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame,
        List[str],
        Dict[str, int],
        Dict[int, str],
        Callable[[str], str],
    ]:
        """Prepare task-specific labels and prompt-building function."""
        if task_type == "FirstWord":
            first_words_list = [
                "I",
                "My",
                "my",
                "When",
                "when",
                "i",
                "A",
                "a",
                "The",
                "the",
                "someone",
                "Someone",
                "we",
            ]
            first_word_to_id = {word: i for i, word in enumerate(first_words_list)}
            id_to_first_word = {v: k for k, v in first_word_to_id.items()}
            emotions_list, emotion_to_id, id_to_emotion = (
                first_words_list,
                first_word_to_id,
                id_to_first_word,
            )

            train_data["emotion_id"] = train_data["hidden_emo_text"].apply(
                lambda x: (
                    first_word_to_id[x.split()[0]]
                    if x.split()[0] in first_words_list
                    else -1
                )
            )
            train_data = train_data[train_data["emotion_id"] != -1]

            func = build_prompt_first_word_prediction()
            train_data["emotion"] = train_data["hidden_emo_text"].apply(
                lambda x: x.split()[0]
            )
            return train_data, emotions_list, emotion_to_id, id_to_emotion, func

        emotions_list = [
            "anger",
            "boredom",
            "disgust",
            "fear",
            "guilt",
            "joy",
            "neutral",
            "pride",
            "relief",
            "sadness",
            "shame",
            "surprise",
            "trust",
        ]
        emotion_to_id = {emotion: i for i, emotion in enumerate(emotions_list)}
        id_to_emotion = {v: k for k, v in emotion_to_id.items()}
        train_data["emotion_id"] = train_data["emotion"].map(emotion_to_id).astype(int)

        if "_" in prompt_type:
            shots = prompt_type.split("_")[:-1]
            prompt_index = int(prompt_type.split("_")[-1])
        else:
            shots = []
            prompt_index = int(prompt_type)

        func = build_prompt(shots=shots, prompt_index=prompt_index)
        return train_data, emotions_list, emotion_to_id, id_to_emotion, func

    train_data = pd.read_csv("data/enVent_gen_Data.csv", encoding="ISO-8859-1")
    train_data["emotion"] = train_data["emotion"].replace("no-emotion", "neutral")
    train_data, emotions_list, emotion_to_id, id_to_emotion, func = (
        resolve_task_configuration(task_type, prompt_type, train_data)
    )
    train_data["input_text"] = train_data["hidden_emo_text"].apply(func)

    appraisals = [
        "predict_event",
        "pleasantness",
        "other_responsblt",
        "chance_control",
        "suddenness",
        "familiarity",
        "unpleasantness",
        "goal_relevance",
        "self_responsblt",
        "predict_conseq",
        "goal_support",
        "urgency",
        "self_control",
        "other_control",
        "accept_conseq",
        "standards",
        "social_norms",
        "attention",
        "not_consider",
        "effort",
    ]
    assert len(appraisals) == len(set(appraisals)), (
        "Duplicate appraisals found in the list"
    )
    appraisals_to_id = {appraisal: i for i, appraisal in enumerate(appraisals)}

    labels = torch.from_numpy(train_data[["emotion_id"] + appraisals].to_numpy())

    dataset = TextDataset(train_data["input_text"].tolist(), labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model_name, model_short_name, model_class = list(
        zip(MODEL_NAMES, MODEL_SHORT_NAMES, MODEL_CLASSES)
    )[model_index]

    if not hooked:
        logger.info("Using an unhooked model ...")
        model_class = AutoModelForCausalLM
        save_prefix = "UNHOOKED_"

    logger.info(f"Model Name: {model_name}")

    if prompt_type == "joy_sadness_0":
        path_prefix = ""
    else:
        path_prefix = prompt_type + "_"

    if task_type == "FirstWord":
        path_suffix = "_FirstWord"
    else:
        path_suffix = ""

    model_short_name = path_prefix + model_short_name + path_suffix

    os.makedirs(f"outputs/{model_short_name}", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        logger.info("Adding padding token to the tokenizer")
        tokenizer.pad_token = tokenizer.eos_token

    model = model_class.from_pretrained(model_name, device_map=device_map)

    token_length_distribution = find_token_length_distribution(
        train_data["input_text"], tokenizer
    )
    logger.info(f"Token Length Distribution: {token_length_distribution}")
    assert token_length_distribution["max_length"] <= tokenizer.model_max_length, (
        "Token length exceeds model's max length."
    )
    logger.info(f"Original Model_Max_Length: {tokenizer.model_max_length}")

    num_params = sum(p.numel() for p in model.parameters())
    size_on_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(
        f"Loaded model '{model_name}' with {num_params}  parameters ({size_on_memory / (1024**2):.2f} MB)"
    )
    logger.info(f"Model configuration: {model.config}")

    emotions_to_tokenized_ids = emotion_to_token_ids(emotions_list, tokenizer)
    all_hidden_states: Optional[torch.Tensor] = None

    if open_vocab:
        run_open_vocab_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            id_to_emotion=id_to_emotion,
            labels=labels,
            model_short_name=model_short_name,
            save_prefix=save_prefix,
            logger=logger,
        )

    if in_domain:
        run_in_domain_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            emotions_to_tokenized_ids=emotions_to_tokenized_ids,
            labels=labels,
            id_to_emotion=id_to_emotion,
            train_data=train_data,
            model_short_name=model_short_name,
            save_prefix=save_prefix,
            logger=logger,
        )

    assert hooked, "Unhooked models are not supported for the following experiments"

    try:
        train_data = pd.read_csv(f"outputs/{model_short_name}/filtered_train_data.csv")
        labels = torch.from_numpy(train_data[["emotion_id"] + appraisals].to_numpy())
    except FileNotFoundError as exc:
        raise Exception(
            "Filtered train data not found, run the code again with --in-domain enabled"
        ) from exc

    dataset = TextDataset(train_data["input_text"].tolist(), labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    if bbox_emotion_regression:
        run_bbox_emotion_regression_experiment(
            labels=labels,
            emotion_to_id=emotion_to_id,
            appraisals_to_id=appraisals_to_id,
            model_short_name=model_short_name,
            logger=logger,
        )

    if save_clean_logits:
        run_save_clean_logits_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            emotions_to_tokenized_ids=emotions_to_tokenized_ids,
            model_short_name=model_short_name,
            logger=logger,
        )

    if extract_hidden_states_enabled:
        all_hidden_states = extract_hidden_states_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            logger=logger,
        )

    if emotion_probing:
        all_hidden_states = run_emotion_probing_experiment(
            labels=labels,
            model=model,
            model_short_name=model_short_name,
            extract_hidden_states_enabled=extract_hidden_states_enabled,
            all_hidden_states=all_hidden_states,
            logger=logger,
        )

    if emotion_probing_non_linear:
        all_hidden_states = run_emotion_probing_non_linear_experiment(
            labels=labels,
            model=model,
            model_short_name=model_short_name,
            extract_hidden_states_enabled=extract_hidden_states_enabled,
            all_hidden_states=all_hidden_states,
            logger=logger,
        )

    if appraisal_probing:
        all_hidden_states = run_appraisal_probing_experiment(
            labels=labels,
            appraisals=appraisals,
            appraisals_to_id=appraisals_to_id,
            model=model,
            model_short_name=model_short_name,
            extract_hidden_states_enabled=extract_hidden_states_enabled,
            all_hidden_states=all_hidden_states,
            logger=logger,
        )

    if extract_weights:
        logger.info(
            "------------------------------ Extracting Weights From Probes ------------------------------"
        )
        run_extract_weights_experiment(
            model=model,
            model_short_name=model_short_name,
            emotions_list=emotions_list,
            appraisals=appraisals,
        )

    if zero_intervention:
        run_zero_intervention_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            logger=logger,
            emotions_to_tokenized_ids=emotions_to_tokenized_ids,
            model_short_name=model_short_name,
        )

    if random_intervention:
        run_random_intervention_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            logger=logger,
            emotions_to_tokenized_ids=emotions_to_tokenized_ids,
            model_short_name=model_short_name,
        )

    if activation_patching_enabled:
        run_activation_patching_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            logger=logger,
            emotions_to_tokenized_ids=emotions_to_tokenized_ids,
            model_short_name=model_short_name,
            train_data=train_data,
        )

    if attention_weights:
        run_attention_weights_experiment(
            dataset=dataset,
            tokenizer=tokenizer,
            model=model,
            logger=logger,
            model_short_name=model_short_name,
        )

    emotions_weights_ = None
    appraisals_weights_ = None
    extraction_locs = None
    extraction_tokens = None
    extraction_layers = None
    if clean_probings or emotion_promotion or appraisal_surgery:
        (
            emotions_weights_,
            _,
            appraisals_weights_,
            _,
            extraction_locs,
            extraction_tokens,
            extraction_layers,
        ) = load_weights_for_promotion_experiments(
            model=model,
            model_short_name=model_short_name,
            logger=logger,
        )

    if emotion_promotion:
        run_emotion_promotion_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            logger=logger,
            emotions_to_tokenized_ids=emotions_to_tokenized_ids,
            model_short_name=model_short_name,
            emotions_weights_=emotions_weights_,
            extraction_locs=extraction_locs,
            extraction_tokens=extraction_tokens,
            extraction_layers=extraction_layers,
            emotions_list=emotions_list,
        )

    if appraisal_surgery:
        run_appraisal_surgery_experiment(
            dataloader=dataloader,
            tokenizer=tokenizer,
            model=model,
            logger=logger,
            emotions_to_tokenized_ids=emotions_to_tokenized_ids,
            model_short_name=model_short_name,
            appraisals_weights_=appraisals_weights_,
            extraction_locs=extraction_locs,
            extraction_tokens=extraction_tokens,
            extraction_layers=extraction_layers,
            appraisals_to_id=appraisals_to_id,
        )


@app.callback(invoke_without_command=True)
def run(
    model_index: int = typer.Option(
        0,
        help="Index of the model to use, 0: Llama3.2_1B, 1: Llama3.1_8B, 2: Gemma-2-2b-it, 3: Gemma-2-9b-it, 4: Phi-3.5-mini-instruct, 5: Phi-3-medium-128k-instruct, 6: OLMo-1B-hf, 7: OLMo-7B-0724-Instruct-hf, 8: Ministral-8B-Instruct-2410, 9: Mistral-Nemo-Instruct-2407, 10: OLMo-2-1124-7B-Instruct, 11: OLMo-2-1124-13B-Instruct",
    ),
    bs: int = typer.Option(4, help="Batch Size"),
    prompt_type: str = typer.Option("joy_sadness_0"),
    task_type: str = typer.Option("Emotion", help="Emotion or FirstWord"),
    open_vocab: bool = typer.Option(
        True,
        "--open-vocab/--no-open-vocab",
        help="Generate open vocab predictions",
    ),
    in_domain: bool = typer.Option(
        True,
        "--in-domain/--no-in-domain",
        help="Generate in-domain emotion labels and filter to correct predictions",
    ),
    bbox_emotion_regression: bool = typer.Option(
        True,
        "--bbox-emotion-regression/--no-bbox-emotion-regression",
        help="Perform black-box emotion regression using appraisals",
    ),
    save_clean_logits: bool = typer.Option(
        True,
        "--save-clean-logits/--no-save-clean-logits",
        help="Save the original clean logits",
    ),
    extract_hidden_states: bool = typer.Option(
        True,
        "--extract-hidden-states/--no-extract-hidden-states",
        help="Extract hidden states",
    ),
    emotion_probing: bool = typer.Option(
        True,
        "--emotion-probing/--no-emotion-probing",
        help="Perform emotion probing",
    ),
    emotion_probing_non_linear: bool = typer.Option(
        True,
        "--emotion-probing-non-linear/--no-emotion-probing-non-linear",
        help="Perform non-linear emotion probing",
    ),
    appraisal_probing: bool = typer.Option(
        True,
        "--appraisal-probing/--no-appraisal-probing",
        help="Perform appraisal probing",
    ),
    extract_weights: bool = typer.Option(
        True,
        "--extract-weights/--no-extract-weights",
        help="Extract weights from probes",
    ),
    zero_intervention: bool = typer.Option(
        True,
        "--zero-intervention/--no-zero-intervention",
        help="Perform zero intervention",
    ),
    random_intervention: bool = typer.Option(
        True,
        "--random-intervention/--no-random-intervention",
        help="Perform random intervention",
    ),
    activation_patching: bool = typer.Option(
        True,
        "--activation-patching/--no-activation-patching",
        help="Perform activation patching",
    ),
    attention_weights: bool = typer.Option(
        True,
        "--attention-weights/--no-attention-weights",
        help="Extract attention weights",
    ),
    emotion_promotion: bool = typer.Option(
        True,
        "--emotion-promotion/--no-emotion-promotion",
        help="Perform emotion promotion",
    ),
    appraisal_surgery: bool = typer.Option(
        False,
        "--appraisal-surgery/--no-appraisal-surgery",
        help="Perform appraisal surgery",
    ),
    clean_probings: bool = typer.Option(
        False,
        "--clean-probings/--no-clean-probings",
    ),
) -> None:
    """CLI entrypoint that mirrors the previous argparse flags using Typer options."""
    if task_type not in {"Emotion", "FirstWord"}:
        raise typer.BadParameter("task_type must be one of: Emotion, FirstWord")

    run_experiments(
        model_index=model_index,
        bs=bs,
        prompt_type=prompt_type,
        task_type=task_type,
        open_vocab=open_vocab,
        in_domain=in_domain,
        bbox_emotion_regression=bbox_emotion_regression,
        save_clean_logits=save_clean_logits,
        extract_hidden_states_enabled=extract_hidden_states,
        emotion_probing=emotion_probing,
        emotion_probing_non_linear=emotion_probing_non_linear,
        appraisal_probing=appraisal_probing,
        extract_weights=extract_weights,
        zero_intervention=zero_intervention,
        random_intervention=random_intervention,
        activation_patching_enabled=activation_patching,
        attention_weights=attention_weights,
        emotion_promotion=emotion_promotion,
        appraisal_surgery=appraisal_surgery,
        clean_probings=clean_probings,
    )


if __name__ == "__main__":
    app()
