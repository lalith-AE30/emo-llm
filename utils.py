import logging
import os
import subprocess
from datetime import datetime

from functools import partial
import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim

from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from cuml.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    GridSearchCV,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from transformer_lens.hook_points import HookPoint
import random


# Define the dataset class for handling text data
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        # self.labels = labels
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):  # ty:ignore[invalid-method-override]
        label = self.labels[idx]
        text = self.texts[idx]
        return text, label


# Log class to handle logging activities
class Log:
    def __init__(self, log_name="probe"):
        filename = (
            f"{log_name}_date-{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.txt"
        )
        os.makedirs("logs", exist_ok=True)
        self.log_path = os.path.join("logs/", filename)
        self.logger = self._setup_logging()

    def _setup_logging(self):
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%d-%b-%y %H:%M:%S",
            handlers=[logging.FileHandler(self.log_path), logging.StreamHandler()],
        )
        return logging.getLogger()


def log_system_info(logger):
    """
    Logs system memory and GPU details.
    """

    def run_command(command):
        """
        Runs a shell command and returns its output.

        Args:
        - command (list): Command and arguments to execute.

        Returns:
        - str: Output of the command.
        """
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return result.stderr

    gpu_info = run_command(["nvidia-smi"])

    if os.name == "nt":  # windows system
        pass
    else:
        memory_info = run_command(["free", "-h"])
        logger.info("Memory Info:\n" + memory_info)

    logger.info("GPU Info:\n" + gpu_info)


def hf_login(logger):
    load_dotenv()
    try:
        # Retrieve the token from an environment variable
        token = os.getenv("HUGGINGFACE_TOKEN")
        if token is None:
            logger.error("Hugging Face token not set in environment variables.")
            return

        # Attempt to log in with the Hugging Face token
        login(token=token)
        logger.info("Logged in successfully to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Login failed: {str(e)}")


def find_token_length_distribution(data, tokenizer):
    token_lengths = []
    for text in data:
        tokens = tokenizer.tokenize(text)
        token_lengths.append(len(tokens))

    token_lengths = np.array(token_lengths)
    quartiles = np.percentile(token_lengths, [25, 50, 75])
    min_length = np.min(token_lengths)
    max_length = np.max(token_lengths)

    return {
        "min_length": min_length,
        "25th_percentile": quartiles[0],
        "median": quartiles[1],
        "75th_percentile": quartiles[2],
        "max_length": max_length,
    }


def emotion_to_token_ids(emotion_labels, tokenizer):
    some_random_text = "Hello, I am a random text."
    new_batch = [f"{some_random_text} {label}" for label in emotion_labels]

    inputs = tokenizer(
        new_batch,
        padding="longest",
        truncation=False,
        return_tensors="pt",
    )
    label_ids = inputs["input_ids"][:, -1]
    return label_ids


def get_emotion_logits(
    dataloader, tokenizer, model, ids_to_pick=None, apply_argmax=False
):

    probs = []

    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        logits = outputs.logits

        logits = logits[:, -1, :]
        if ids_to_pick is not None:
            if torch.is_tensor(ids_to_pick):
                ids_to_pick = ids_to_pick.to(logits.device)
            logits = logits[:, ids_to_pick]

        if apply_argmax:
            logits = torch.argmax(logits, dim=-1)

        probs.append(logits)

    probs = torch.cat(probs, dim=0)
    return probs.cpu()


def probe(all_hidden_states, labels, appraisals, logger):
    if isinstance(all_hidden_states, torch.Tensor):
        all_hidden_states = all_hidden_states.cpu().numpy()

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    # Normalize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    Y_emotion = labels[:, 0]
    Y_appraisals = labels[:, 1:]

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}

    # Probing for emotion (classification)
    try:
        # logger.info(f"Feature matrix shape: {X.shape}")
        # logger.info(f"Target vector shape: {Y_emotion.shape}")

        cv_accuracies = cross_val_score(
            LogisticRegression(max_iter=2000),
            X,
            Y_emotion,
            cv=kfold,
            scoring="accuracy",
        )
        classifier = LogisticRegression(max_iter=2000)
        classifier.fit(
            X, Y_emotion
        )  # Train on the entire dataset for full model training after CV
        training_accuracy = classifier.score(X, Y_emotion)

        logger.info(
            f"5-Fold CV Accuracy for emotion category: {cv_accuracies.mean():.4f} ± {cv_accuracies.std():.4f}"
        )
        logger.info(f"Training Accuracy for emotion category: {training_accuracy:.4f}")

        results["emotion"] = {
            "cv_accuracy": cv_accuracies.mean(),
            "cv_std": cv_accuracies.std(),
            "training_accuracy": training_accuracy,
        }
    except Exception as e:
        logger.error(f"Error while probing emotion category: {e}")

    # Probing for each appraisal (regression)
    for i, appraisal_name in enumerate(appraisals):
        try:
            Y = Y_appraisals[:, i]
            logger.info(f"Probing appraisal: {appraisal_name}")
            # logger.info(f"Feature matrix shape: {X.shape}")
            # logger.info(f"Target vector shape: {Y.shape}")
            # logger.info(f"Feature 1st 5: {X[:5]}")
            # logger.info(f"Target 1st 5: {Y[:5]}")

            # Define parameter grid for ElasticNet
            param_grid = {
                "alpha": [0.1],  # , 1.0, 10.0
                "l1_ratio": [0.1],  # , 0.5, 0.9
            }

            enet = ElasticNet(max_iter=5000)
            grid_search = GridSearchCV(
                enet, param_grid, cv=kfold, scoring="r2", n_jobs=-1
            )
            grid_search.fit(X, Y)
            # enet.fit(X, Y)
            # best_model  = enet
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best hyperparameters for '{appraisal_name}': {best_params}")

            cv_mse = cross_val_score(
                best_model, X, Y, cv=kfold, scoring="neg_mean_squared_error"
            )
            cv_r2 = cross_val_score(best_model, X, Y, cv=kfold, scoring="r2")

            training_predictions = best_model.predict(X)
            training_mse = mean_squared_error(Y, training_predictions)
            training_r2 = r2_score(Y, training_predictions)

            logger.info(
                f"5-Fold CV MSE for '{appraisal_name}': {-cv_mse.mean():.4f} ± {cv_mse.std():.4f}"
            )
            logger.info(f"Training MSE for '{appraisal_name}': {training_mse:.4f}")
            logger.info(
                f"5-Fold CV R-squared for '{appraisal_name}': {cv_r2.mean():.4f} ± {cv_r2.std():.4f}"
            )
            logger.info(f"Training R-squared for '{appraisal_name}': {training_r2:.4f}")
            logger.info("- -" * 25)
        except Exception as e:
            logger.error(f"Error while probing appraisal '{appraisal_name}': {e}")

        results[appraisal_name] = {
            "training_mse": training_mse,
            "cv_mse": -cv_mse.mean(),
            "cv_mse_std": cv_mse.std(),
            "training_r2": training_r2,
            "cv_r2": cv_r2.mean(),
            "cv_r2_std": cv_r2.std(),
        }

    return results


def probe_regression(all_hidden_states, labels, return_weights=False):
    if len(labels.shape) == 1:
        labels = labels[:, None]

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)
    X = X.cpu().numpy()
    Y = labels
    Y = Y.cpu().numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # scaler = StandardScaler(with_std=False)
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # scaler = StandardScaler()
    # Y_train = scaler.fit_transform(Y_train)
    # Y_test = scaler.transform(Y_test)

    net = Ridge(alpha=5.0)  # ElasticNet(alpha=0.1, l1_ratio=0.1)
    net.fit(X_train, Y_train)
    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    mse_train = mean_squared_error(Y_train, y_pred_train)
    mse_test = mean_squared_error(Y_test, y_pred_test)
    r2_train = r2_score(Y_train, y_pred_train)
    r2_test = r2_score(Y_test, y_pred_test)
    res = {
        "mse_train": mse_train,
        "mse_test": mse_test,
        "r2_train": r2_train,
        "r2_test": r2_test,
    }
    if return_weights:
        res["weights"] = net.coef_
        res["bias"] = net.intercept_
    return res


def probe_classification(
    all_hidden_states,
    labels,
    return_weights=False,
    Normalize_X=False,
    reg_strength=1.0,
    fit_intercept=True,
):
    if len(labels.shape) == 2:
        labels = labels[:, 0]

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)
    X = X.cpu().numpy()
    Y = labels
    Y = Y.cpu().numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    if Normalize_X:
        scaler = StandardScaler(with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    net = LogisticRegression(C=1 / reg_strength, fit_intercept=fit_intercept)
    net.fit(X_train, Y_train)

    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    if isinstance(Y_train, np.ndarray):
        Y_train = torch.tensor(Y_train)
        Y_test = torch.tensor(Y_test)

    if isinstance(y_pred_train, np.ndarray):
        y_pred_train = torch.tensor(y_pred_train)
        y_pred_test = torch.tensor(y_pred_test)

    accuracy_train = (Y_train == y_pred_train).float().mean()

    accuracy_test = (Y_test == y_pred_test).float().mean()
    res = {"accuracy_train": accuracy_train, "accuracy_test": accuracy_test}
    if return_weights:
        res["weights"] = net.coef_
        res["bias"] = net.intercept_

    return res


class MLPClassifier:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=200,
        random_state=None,
        device="auto",
        use_amp="auto",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.device = device
        self.use_amp = use_amp

        self.model_ = None
        self.classes_ = None
        self.device_ = None  # resolved device stored after fit()

    def _resolve_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():  # Apple Silicon
            return torch.device("mps")
        return torch.device("cpu")

    def _build_model(self, input_size, output_size):
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "logistic": nn.Sigmoid(),
            "identity": nn.Identity(),
        }
        act = activations[self.activation]

        layer_sizes = [input_size, *self.hidden_layer_sizes, output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act)

        return nn.Sequential(*layers)

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self.device_ = self._resolve_device()  # <-- resolve once at fit time

        # Normalize incoming array/tensor types once to avoid per-batch conversion overhead.
        if torch.is_tensor(X):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if torch.is_tensor(y):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)

        self.classes_, y_mapped = np.unique(y_np, return_inverse=True)

        input_size = X_np.shape[1]
        output_size = len(self.classes_)

        self.model_ = self._build_model(input_size, output_size).to(
            self.device_
        )  # <-- move model

        solvers = {"adam": optim.Adam, "sgd": optim.SGD}
        optimizer = solvers[self.solver](
            self.model_.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha,
        )

        n_samples = X_np.shape[0]
        if self.batch_size == "auto":
            if self.device_.type == "cuda":
                batch_size = min(4096, n_samples)
            elif self.device_.type == "mps":
                batch_size = min(2048, n_samples)
            else:
                batch_size = min(1024, n_samples)
        else:
            batch_size = int(self.batch_size)

        X_t = torch.as_tensor(X_np, dtype=torch.float32, device=self.device_)
        y_t = torch.as_tensor(y_mapped, dtype=torch.long, device=self.device_)

        amp_enabled = (self.use_amp is True) or (
            self.use_amp == "auto" and self.device_.type == "cuda"
        )
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        criterion = nn.CrossEntropyLoss()
        self.model_.train()

        for _ in range(self.max_iter):
            perm = torch.randperm(n_samples, device=self.device_)
            for start in range(0, n_samples, batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device_.type,
                    enabled=amp_enabled,
                ):
                    loss = criterion(self.model_(X_t[idx]), y_t[idx])

                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        return self

    def predict(self, X):
        probas = self.predict_proba(X)
        indices = np.argmax(probas, axis=1)
        return self.classes_[indices]  # ty:ignore[not-subscriptable]

    def predict_proba(self, X):
        assert self.model_ is not None, "Call fit() first."
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device_)
            logits = self.model_(X_t)
            return (
                torch.softmax(logits, dim=1).cpu().numpy()
            )  # <-- .cpu() before .numpy()

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


def probe_classification_non_linear(
    all_hidden_states,
    labels,
    return_weights=False,
    Normalize_X=False,
    reg_strength=1.0,
    fit_intercept=True,
):
    if len(labels.shape) == 2:
        labels = labels[:, 0]

    X = all_hidden_states.reshape(all_hidden_states.shape[0], -1)

    Y = labels

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    if Normalize_X:
        scaler = StandardScaler(with_std=False)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    net = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, activation="relu")
    net.fit(X_train, Y_train)

    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    if isinstance(Y_train, np.ndarray):
        Y_train = torch.tensor(Y_train)
        Y_test = torch.tensor(Y_test)

    if isinstance(y_pred_train, np.ndarray):
        y_pred_train = torch.tensor(y_pred_train)
        y_pred_test = torch.tensor(y_pred_test)

    accuracy_train = (Y_train == y_pred_train).float().mean()

    accuracy_test = (Y_test == y_pred_test).float().mean()
    res = {"accuracy_train": accuracy_train, "accuracy_test": accuracy_test}

    return res


extraction_locations = {
    1: "model.layers.[LID].hook_initial_hs",
    2: "model.layers.[LID].hook_after_attn_normalization",
    3: "model.layers.[LID].hook_after_attn",
    4: "model.layers.[LID].hook_after_attn_hs",
    5: "model.layers.[LID].hook_after_mlp_normalization",
    6: "model.layers.[LID].hook_after_mlp",
    7: "model.layers.[LID].hook_after_mlp_hs",
    8: "model.layers.[LID].self_attn.hook_attn_heads",
    9: "model.final_hook",
    10: "model.layers.[LID].self_attn.hook_attn_weights",
}


def name_to_loc_and_layer(name):
    layer = int(name.split("model.layers.")[1].split(".")[0])
    loc_suffixes = {v.split(".")[-1]: k for k, v in extraction_locations.items()}
    loc = loc_suffixes[name.split(".")[-1]]

    return loc, layer


def extract_from_cache(
    cache_dict_,
    extraction_layers=[0, 1],
    extraction_locs=[1, 7],
    extraction_tokens=[-1],
):
    return_value = []

    for layer in extraction_layers:
        return_value.append([])
        for el_ in extraction_locs:
            el = extraction_locations[el_].replace("[LID]", str(layer))
            if el_ != 10:  # attention weights should be treated differently
                return_value[-1].append(cache_dict_[el][:, extraction_tokens].cpu())  # ty:ignore[unresolved-attribute]
            else:
                return_value[-1].append(cache_dict_[el][:, :, extraction_tokens].cpu())  # ty:ignore[unresolved-attribute]

        return_value[-1] = torch.stack(return_value[-1], dim=1)  # ty:ignore[invalid-argument-type]
    return_value = torch.stack(return_value, dim=1)
    return return_value


def extract_hidden_states(
    dataloader,
    tokenizer,
    model,
    logger,
    extraction_layers=[0, 1],
    extraction_locs=[1, 7],
    extraction_tokens=[-1],
    do_final_cat=True,
    return_tokenized_input=False,
):
    assert [
        extraction_loc in extraction_locations.keys()
        for extraction_loc in extraction_locs
    ]
    assert (10 not in extraction_locs) or len(extraction_locs) == 1

    output_attentions = 10 in extraction_locs

    return_values = []

    tokenized_input = []

    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.run_with_cache(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
                output_attentions=output_attentions,
            )

        cache_dict_ = outputs[1]

        r = extract_from_cache(
            cache_dict_,
            extraction_layers=extraction_layers,
            extraction_locs=extraction_locs,
            extraction_tokens=extraction_tokens,
        )

        return_values.append(r)

        if return_tokenized_input:
            assert len(inputs["input_ids"]) == 1, (
                "Batch size must be 1 for tokenized input extraction"
            )
            tokenized_input.append(
                tokenizer.convert_ids_to_tokens(
                    [w for w in inputs["input_ids"][0].cpu()]
                )
            )

    if do_final_cat:
        return_values = torch.cat(return_values, dim=0)

    if return_tokenized_input:
        return return_values, tokenized_input
    return return_values


def apply_zero_intervention_and_extract_logits(
    dataloader,
    tokenizer,
    model,
    logger,
    intervention_layers=[0, 1],
    intervention_tokens="all",
    intervention_locs=[1, 7],
    ids_to_pick=None,
):
    assert [
        intervention_loc in extraction_locations.keys()
        for intervention_loc in intervention_locs
    ]
    intervention_tokens = (
        intervention_tokens if intervention_tokens != "all" else slice(None)
    )
    names_to_intervene = set(
        [
            extraction_locations[loc].replace("[LID]", str(layer))
            for layer in intervention_layers
            for loc in intervention_locs
        ]
    )

    def zero_intervention_hook(input_vector, hook: HookPoint):
        name = hook.name

        if name in names_to_intervene:
            input_vector[:, intervention_tokens] = (
                input_vector[:, intervention_tokens] * 0
            )

        return input_vector

    returned_logits = []
    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.run_with_hooks(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
                fwd_hooks=[(lambda x: True, zero_intervention_hook)],
            )

        logits = outputs.logits[:, -1, :]
        if ids_to_pick is not None:
            if torch.is_tensor(ids_to_pick):
                ids_to_pick = ids_to_pick.to(logits.device)
            logits = logits[:, ids_to_pick]

        returned_logits.append(logits)

    all_logits = torch.cat(returned_logits, dim=0)
    return all_logits.cpu()


def apply_random_intervention_and_extract_logits(
    dataloader,
    tokenizer,
    model,
    logger,
    intervention_layers=[0, 1],
    intervention_tokens="all",
    intervention_locs=[1, 7],
    ids_to_pick=None,
):
    assert [
        intervention_loc in extraction_locations.keys()
        for intervention_loc in intervention_locs
    ]
    intervention_tokens = (
        intervention_tokens if intervention_tokens != "all" else slice(None)
    )
    names_to_intervene = [
        extraction_locations[loc].replace("[LID]", str(layer))
        for layer in intervention_layers
        for loc in intervention_locs
    ]

    def random_intervention_hook(input_vector, hook: HookPoint):
        name = hook.name

        if name in names_to_intervene:
            v = input_vector[:, intervention_tokens]
            v_ = torch.randn_like(v)
            v_ = v_ / v_.norm(dim=-1, keepdim=True) * v.norm(dim=-1, keepdim=True)

            input_vector[:, intervention_tokens] = v * 0 + v_

        return input_vector

    returned_logits = []
    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.run_with_hooks(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
                fwd_hooks=[(lambda x: True, random_intervention_hook)],
            )

        logits = outputs.logits[:, -1, :]
        if ids_to_pick is not None:
            if torch.is_tensor(ids_to_pick):
                ids_to_pick = ids_to_pick.to(logits.device)
            logits = logits[:, ids_to_pick]

        returned_logits.append(logits)

    all_logits = torch.cat(returned_logits, dim=0)
    return all_logits.cpu()


def activation_patching(
    source_sentence,
    target_sentence,
    tokenizer,
    model,
    logger,
    intervention_layers=[0, 1],
    intervention_locs=[1, 7],
    intervention_tokens=[-1],
    ids_to_pick=None,
):
    assert [
        intervention_loc in extraction_locations.keys()
        for intervention_loc in intervention_locs
    ]

    source_sentence_ids = tokenizer(
        [source_sentence], return_tensors="pt", padding="longest", truncation=False
    ).to(model.device)
    target_sentence_ids = tokenizer(
        [target_sentence], return_tensors="pt", padding="longest", truncation=False
    ).to(model.device)

    with torch.no_grad():
        source_outputs = model.run_with_cache(
            **source_sentence_ids, return_dict=True, output_hidden_states=True
        )
        source_clean_cache = {k: v.cpu() for k, v in source_outputs[1].items()}
        source_clean_logits = source_outputs[0].logits[0, -1].cpu()
        del source_outputs

        target_outputs = model.run_with_cache(
            **target_sentence_ids, return_dict=True, output_hidden_states=True
        )
        target_clean_cache = {k: v.cpu() for k, v in target_outputs[1].items()}
        target_clean_logits = target_outputs[0].logits[0, -1].cpu()
        del target_outputs

    if ids_to_pick is not None:
        source_clean_logits = source_clean_logits[ids_to_pick]
        target_clean_logits = target_clean_logits[ids_to_pick]

    def patching_hook(input_vector, hook: HookPoint):
        name = hook.name
        names_to_intervene = [
            extraction_locations[loc].replace("[LID]", str(layer))
            for layer in intervention_layers
            for loc in intervention_locs
        ]
        if name in names_to_intervene:
            input_vector[:, intervention_tokens] = input_vector[
                :, intervention_tokens
            ] * 0 + source_clean_cache[name][:, intervention_tokens].to(
                input_vector.device
            )
        return input_vector

    with torch.no_grad():
        outputs = model.run_with_hooks(
            **target_sentence_ids,
            return_dict=True,
            output_hidden_states=True,
            fwd_hooks=[(lambda x: True, patching_hook)],
        )
        patched_logits = outputs.logits[0, -1].cpu()

    if ids_to_pick is not None:
        patched_logits = patched_logits[ids_to_pick]

    return {
        "source_clean_logits": source_clean_logits,
        "target_clean_logits": target_clean_logits,
        "patched_logits": patched_logits,
    }


def promote_vec(
    dataloader,
    tokenizer,
    model,
    logger,
    prom_vector,
    projection_matrix,
    Beta,
    promotion_layers=[1, 2],
    promotion_locs=[3, 6],
    promotion_tokens=[-1],
    extraction_layers=[0],
    extraction_locs=[7],
    extraction_tokens=[-1],
    ids_to_pick=None,
):

    assert [
        extraction_loc in extraction_locations.keys()
        for extraction_loc in extraction_locs
    ]
    assert 10 not in extraction_locs, (
        "Cannot extract attention weights from this function"
    )

    hs = []

    hidden_state_size = model.config.hidden_size

    assert promotion_tokens == "all" or isinstance(promotion_tokens, list)
    if promotion_tokens == "all":
        promotion_tokens = slice(None)
        assert prom_vector.shape == (
            len(promotion_layers),
            len(promotion_locs),
            1,
            hidden_state_size,
        )
        assert projection_matrix.shape == (
            len(promotion_layers),
            len(promotion_locs),
            1,
            hidden_state_size,
            hidden_state_size,
        )
    else:
        assert prom_vector.shape == (
            len(promotion_layers),
            len(promotion_locs),
            len(promotion_tokens),
            hidden_state_size,
        )
        assert projection_matrix.shape == (
            len(promotion_layers),
            len(promotion_locs),
            len(promotion_tokens),
            hidden_state_size,
            hidden_state_size,
        )

    first_terms = prom_vector.unsqueeze(0)  # add batch dimension

    names_to_promote = [
        extraction_locations[loc].replace("[LID]", str(layer))
        for layer in promotion_layers
        for loc in promotion_locs
    ]

    def promotion_hook(input_vector, hook: HookPoint, cache_dict: dict):

        name = hook.name

        if name in names_to_promote:
            loc, layer = name_to_loc_and_layer(name)

            layer_idx = promotion_layers.index(layer)
            loc_idx = promotion_locs.index(loc)

            uA = input_vector[:, promotion_tokens, :]  # batch, token, hidden

            P = projection_matrix[layer_idx, loc_idx, :, :, :].to(
                uA.device
            )  # token, hidden, hidden

            first_term = first_terms[:, layer_idx, loc_idx].to(uA.device)
            second_term = Beta * torch.einsum("tdh, bth->btd", P, uA)

            uA = uA + first_term - second_term

            input_vector[:, promotion_tokens, :] = uA

        cache_dict[name] = input_vector.clone()

        return input_vector

    returned_logits = []

    for i, (batch_texts, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = tokenizer(
            batch_texts,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            cache_dict_ = {}
            ph = partial(promotion_hook, cache_dict=cache_dict_)
            outputs = model.run_with_hooks(
                **inputs,
                return_dict=True,
                output_hidden_states=True,
                fwd_hooks=[(lambda x: True, ph)],
            )

        r = extract_from_cache(
            cache_dict_,
            extraction_layers=extraction_layers,
            extraction_locs=extraction_locs,
            extraction_tokens=extraction_tokens,
        )
        hs.append(r)

        logits = outputs.logits.cpu()
        logits = logits[:, -1, :]
        if ids_to_pick is not None:
            logits = logits[:, ids_to_pick]

        returned_logits.append(logits)

    hs = torch.cat(hs, dim=0)
    return torch.cat(returned_logits, dim=0), hs


def make_projections(w):
    # assuming the last 2 dimensions of w, shows the number of vectors and the size of the vector dimension respectively
    w_shape = w.shape[:-2]
    n = w.shape[-2]
    d = w.shape[-1]

    # flatten except the last 2 dimensions
    w = w.reshape(-1, n, d)

    return_result = torch.zeros([w.shape[0], d, d], device=w.device, dtype=w.dtype)

    for i in range(w.shape[0]):
        w_ = w[i]
        return_result[i] = w_.T @ (w_ @ w_.T).inverse() @ w_

    return return_result.reshape(w_shape + (d, d))


def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)


def apply_classification_probe(data, weights, bias):
    num_data = data.shape[0]
    features_dim = data.shape[-1]
    num_classes = weights.shape[0]

    data_shape = data.shape[1:-1]
    data = data.reshape(num_data, -1, features_dim)

    weights = weights.reshape(num_classes, -1, features_dim)
    bias = bias.reshape(num_classes, -1)

    logits = torch.einsum("ctd, btd->btc", weights, data) + bias.T
    logits = logits.reshape(num_data, *data_shape, num_classes)

    return logits


def apply_regression_probe(data, weights, bias):
    num_data = data.shape[0]
    features_dim = data.shape[-1]
    num_outputs = weights.shape[0]

    data_shape = data.shape[1:-1]
    data = data.reshape(num_data, -1, features_dim)

    weights = weights.reshape(num_outputs, -1, features_dim)
    bias = bias.reshape(num_outputs, -1)

    logits = torch.einsum("ctd, btd->btc", weights, data) + bias.T
    logits = logits.reshape(num_data, *data_shape, num_outputs)

    return logits


def apply_probes_on_hs(
    hs,
    emotions_weights,
    emotions_biases,
    appraisals_weights,
    appraisals_biases,
    extraction_layers=list(range(16)),
    extraction_locs=[3, 6, 7],
    extraction_tokens=[-1],
    layer_to_monitor=15,
    loc_to_monitor=7,
    token_to_monitor=-1,
):
    w = emotions_weights[
        :,
        extraction_layers.index(layer_to_monitor),
        extraction_locs.index(loc_to_monitor),
        extraction_tokens.index(token_to_monitor),
        :,
    ]
    b = emotions_biases[
        :,
        extraction_layers.index(layer_to_monitor),
        extraction_locs.index(loc_to_monitor),
        extraction_tokens.index(token_to_monitor),
    ]
    preds_emo = apply_classification_probe(hs, w, b)

    w = appraisals_weights[
        :,
        extraction_layers.index(layer_to_monitor),
        extraction_locs.index(loc_to_monitor),
        extraction_tokens.index(token_to_monitor),
        :,
    ]
    b = appraisals_biases[
        :,
        extraction_layers.index(layer_to_monitor),
        extraction_locs.index(loc_to_monitor),
        extraction_tokens.index(token_to_monitor),
    ]
    preds_app = apply_regression_probe(hs, w, b)

    return preds_emo, preds_app
