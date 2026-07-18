# train.py
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from models import get_model


def count_trainable_params(model: nn.Module) -> int:
    """Return number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        losses.append(loss.item())
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy())
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    acc = accuracy_score(all_labels, all_preds) if all_labels else 0.0
    return avg_loss, acc


def run_search(
    candidates,
    train_ds,
    val_ds,
    num_classes,
    device: str = "cpu",
    epochs_per_trial: int = 2,
    batch_size: int = 16,
    lr: float = 1e-3,
    early_stopping_patience: int = 3,
    epoch_callback=None,
    trial_start_callback=None,
):
    """
    Lightweight architecture search with early stopping and live callbacks.

    Parameters
    ----------
    candidates : list[str]
        Model names to try (e.g. ["tiny_cnn", "mobilenet_v2"]).
    train_ds, val_ds : Dataset
        PyTorch datasets for training and validation.
    num_classes : int
    device : str
    epochs_per_trial : int
        Maximum epochs to run for each candidate.
    batch_size : int
    lr : float
    early_stopping_patience : int
        How many epochs without improvement before stopping this trial.
        Set to epochs_per_trial+1 to effectively disable early stopping.
    epoch_callback : callable or None
        Called after each epoch with signature:
            epoch_callback(model_name, epoch, epochs_per_trial,
                           tr_loss, tr_acc, val_loss, val_acc,
                           stopped_early)
        Use this to push live updates to Streamlit.
    trial_start_callback : callable or None
        Called when a new candidate trial begins:
            trial_start_callback(model_name, param_count)

    Returns
    -------
    best_name : str
    best_score : float
    best_state_dict_on_cpu : dict
    search_logs : list[dict]
        Each dict has keys: name, val_acc, params, epochs_run, time_sec,
                            train_acc_history, val_acc_history,
                            train_loss_history, val_loss_history
    """
    best_name, best_score, best_state = None, -1.0, None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    criterion = nn.CrossEntropyLoss()
    search_logs = []

    for name in candidates:
        model = get_model(name, num_classes, pretrained=True).to(device)
        param_count = count_trainable_params(model)

        if trial_start_callback is not None:
            trial_start_callback(name, param_count)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping state
        best_val_in_trial = -1.0
        patience_counter = 0
        best_state_in_trial = {k: v.cpu() for k, v in model.state_dict().items()}

        # Per-epoch history (for charts)
        train_acc_hist, val_acc_hist = [], []
        train_loss_hist, val_loss_hist = [], []

        trial_start = time.time()
        stopped_early = False
        epochs_run = 0

        for e in range(epochs_per_trial):
            epochs_run = e + 1
            tr_loss, tr_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_epoch(
                model, val_loader, criterion, device
            )

            train_acc_hist.append(tr_acc)
            val_acc_hist.append(val_acc)
            train_loss_hist.append(tr_loss)
            val_loss_hist.append(val_loss)

            # Early stopping check
            if val_acc > best_val_in_trial + 1e-4:
                best_val_in_trial = val_acc
                patience_counter = 0
                best_state_in_trial = {
                    k: v.cpu() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1

            if epoch_callback is not None:
                epoch_callback(
                    name, e + 1, epochs_per_trial,
                    tr_loss, tr_acc, val_loss, val_acc,
                    stopped_early=False,
                )

            if patience_counter >= early_stopping_patience:
                stopped_early = True
                if epoch_callback is not None:
                    epoch_callback(
                        name, e + 1, epochs_per_trial,
                        tr_loss, tr_acc, val_loss, val_acc,
                        stopped_early=True,
                    )
                break

        trial_sec = time.time() - trial_start

        # Restore best weights seen during this trial
        model.load_state_dict(best_state_in_trial)
        final_val_acc = best_val_in_trial

        search_logs.append({
            "name": name,
            "val_acc": float(final_val_acc),
            "params": int(param_count),
            "epochs_run": int(epochs_run),
            "time_sec": round(trial_sec, 1),
            "train_acc_history": train_acc_hist,
            "val_acc_history": val_acc_hist,
            "train_loss_history": train_loss_hist,
            "val_loss_history": val_loss_hist,
        })

        if final_val_acc > best_score:
            best_score = final_val_acc
            best_name = name
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_name, best_score, best_state, search_logs


def run_config_search(
    candidates,
    train_ds,
    val_ds,
    num_classes,
    device: str = "cpu",
    epochs_per_trial: int = 2,
    batch_size: int = 16,
    lr: float = 1e-3,
    early_stopping_patience: int = 3,
    max_params=None,
    epoch_callback=None,
    trial_start_callback=None,
):
    """
    Same idea as run_search, but candidates are config dicts for the
    "Configurable Space" NAS mode (see search_space.py / models.get_model_from_config)
    instead of preset model name strings.

    Parameters
    ----------
    candidates : list[dict]
        Config dicts, e.g. {"depth":3,"base_channels":32,"block_type":"residual","dropout":0.2}
    max_params : int or None
        If given, any candidate whose parameter count exceeds this is skipped
        (not trained, not included in search_logs).
    (all other parameters match run_search)

    Returns
    -------
    best_config : dict
    best_score : float
    best_state_dict_on_cpu : dict
    search_logs : list[dict]
        Each dict has keys: name, config, val_acc, params, epochs_run, time_sec,
                            train_acc_history, val_acc_history,
                            train_loss_history, val_loss_history
    """
    from models import get_model_from_config
    from search_space import config_to_name

    best_config, best_score, best_state = None, -1.0, None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    criterion = nn.CrossEntropyLoss()
    search_logs = []

    for config in candidates:
        name = config_to_name(config)
        model = get_model_from_config(config, num_classes).to(device)
        param_count = count_trainable_params(model)

        if max_params is not None and param_count > max_params:
            # Skip candidates over the parameter budget entirely.
            continue

        if trial_start_callback is not None:
            trial_start_callback(name, param_count)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_in_trial = -1.0
        patience_counter = 0
        best_state_in_trial = {k: v.cpu() for k, v in model.state_dict().items()}

        train_acc_hist, val_acc_hist = [], []
        train_loss_hist, val_loss_hist = [], []

        trial_start = time.time()
        stopped_early = False
        epochs_run = 0

        for e in range(epochs_per_trial):
            epochs_run = e + 1
            tr_loss, tr_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_epoch(
                model, val_loader, criterion, device
            )

            train_acc_hist.append(tr_acc)
            val_acc_hist.append(val_acc)
            train_loss_hist.append(tr_loss)
            val_loss_hist.append(val_loss)

            if val_acc > best_val_in_trial + 1e-4:
                best_val_in_trial = val_acc
                patience_counter = 0
                best_state_in_trial = {
                    k: v.cpu() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1

            if epoch_callback is not None:
                epoch_callback(
                    name, e + 1, epochs_per_trial,
                    tr_loss, tr_acc, val_loss, val_acc,
                    stopped_early=False,
                )

            if patience_counter >= early_stopping_patience:
                stopped_early = True
                if epoch_callback is not None:
                    epoch_callback(
                        name, e + 1, epochs_per_trial,
                        tr_loss, tr_acc, val_loss, val_acc,
                        stopped_early=True,
                    )
                break

        trial_sec = time.time() - trial_start

        model.load_state_dict(best_state_in_trial)
        final_val_acc = best_val_in_trial

        search_logs.append({
            "name": name,
            "config": config,
            "val_acc": float(final_val_acc),
            "params": int(param_count),
            "epochs_run": int(epochs_run),
            "time_sec": round(trial_sec, 1),
            "train_acc_history": train_acc_hist,
            "val_acc_history": val_acc_hist,
            "train_loss_history": train_loss_hist,
            "val_loss_history": val_loss_hist,
        })

        if final_val_acc > best_score:
            best_score = final_val_acc
            best_config = config
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    return best_config, best_score, best_state, search_logs


def run_unified_search(
    preset_candidates,
    config_candidates,
    train_ds,
    val_ds,
    num_classes,
    device: str = "cpu",
    epochs_per_trial: int = 2,
    batch_size: int = 16,
    lr: float = 1e-3,
    early_stopping_patience: int = 3,
    max_params=None,
    epoch_callback=None,
    trial_start_callback=None,
):
    """
    Unified NAS search: trains every preset architecture first, then every
    configurable-space candidate, and keeps track of the single best result
    across BOTH pools combined.

    Parameters
    ----------
    preset_candidates : list[str]
        Preset model names (e.g. ["tiny_cnn", "mobilenet_v2"]). Can be empty.
    config_candidates : list[dict]
        Config dicts for the configurable search space (see search_space.py).
        Can be empty.
    max_params : int or None
        Parameter budget applied only to config_candidates. Preset
        architectures are never skipped on parameter count.
    epoch_callback : callable or None
        epoch_callback(name, epoch, epochs_per_trial, tr_loss, tr_acc,
                        val_loss, val_acc, stopped_early)
    trial_start_callback : callable or None
        trial_start_callback(name, param_count)

    Returns
    -------
    best_kind : "preset" | "config" | None
        Tells the caller how to rebuild the winning model.
    best_id : str or dict or None
        Preset name (if best_kind == "preset") or config dict (if "config").
    best_score : float
    best_state_dict_on_cpu : dict or None
    search_logs : list[dict]
        Each dict has keys: name, kind, config (dict or None), val_acc, params,
                            epochs_run, time_sec, train_acc_history,
                            val_acc_history, train_loss_history, val_loss_history
        Preset trials are listed first, in the order given, followed by
        configurable-space trials.
    """
    from models import get_model_from_config
    from search_space import config_to_name

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    state = {"best_kind": None, "best_id": None, "best_score": -1.0, "best_state": None}
    search_logs = []

    def _run_one_trial(name, kind, config, model):
        model = model.to(device)
        param_count = count_trainable_params(model)

        if trial_start_callback is not None:
            trial_start_callback(name, param_count)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_in_trial = -1.0
        patience_counter = 0
        best_state_in_trial = {k: v.cpu() for k, v in model.state_dict().items()}

        train_acc_hist, val_acc_hist = [], []
        train_loss_hist, val_loss_hist = [], []

        trial_start = time.time()
        epochs_run = 0

        for e in range(epochs_per_trial):
            epochs_run = e + 1
            tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

            train_acc_hist.append(tr_acc)
            val_acc_hist.append(val_acc)
            train_loss_hist.append(tr_loss)
            val_loss_hist.append(val_loss)

            if val_acc > best_val_in_trial + 1e-4:
                best_val_in_trial = val_acc
                patience_counter = 0
                best_state_in_trial = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            if epoch_callback is not None:
                epoch_callback(name, e + 1, epochs_per_trial, tr_loss, tr_acc,
                                val_loss, val_acc, stopped_early=False)

            if patience_counter >= early_stopping_patience:
                if epoch_callback is not None:
                    epoch_callback(name, e + 1, epochs_per_trial, tr_loss, tr_acc,
                                    val_loss, val_acc, stopped_early=True)
                break

        trial_sec = time.time() - trial_start
        final_val_acc = best_val_in_trial

        search_logs.append({
            "name": name,
            "kind": kind,
            "config": config,
            "val_acc": float(final_val_acc),
            "params": int(param_count),
            "epochs_run": int(epochs_run),
            "time_sec": round(trial_sec, 1),
            "train_acc_history": train_acc_hist,
            "val_acc_history": val_acc_hist,
            "train_loss_history": train_loss_hist,
            "val_loss_history": val_loss_hist,
        })

        if final_val_acc > state["best_score"]:
            state["best_score"] = final_val_acc
            state["best_kind"] = kind
            state["best_id"] = config if kind == "config" else name
            state["best_state"] = best_state_in_trial

    # 1) Preset architectures first, in the order given.
    for name in preset_candidates:
        model = get_model(name, num_classes, pretrained=True)
        _run_one_trial(name, "preset", None, model)

    # 2) Configurable-space candidates second, skipping any over budget.
    for config in config_candidates:
        model = get_model_from_config(config, num_classes)
        param_count_preview = count_trainable_params(model)
        if max_params is not None and param_count_preview > max_params:
            continue
        name = config_to_name(config)
        _run_one_trial(name, "config", config, model)

    return (
        state["best_kind"], state["best_id"], state["best_score"],
        state["best_state"], search_logs,
    )


def fine_tune_model(
    model,
    train_ds,
    val_ds,
    device: str = "cpu",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    early_stopping_patience: int = 5,
    epoch_callback=None,
):
    """
    Continue training (fine-tuning) a given model for more epochs,
    with optional early stopping and live epoch callbacks.

    Parameters
    ----------
    epoch_callback : callable or None
        Called after each epoch with signature:
            epoch_callback(epoch, total_epochs,
                           tr_loss, tr_acc, val_loss, val_acc,
                           is_best, stopped_early)

    Returns
    -------
    model : nn.Module  (best weights restored)
    best_val_acc : float
    history : dict with keys train_acc, val_acc, train_loss, val_loss
    """
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    best_val_acc = 0.0
    patience_counter = 0

    history = {
        "train_acc": [], "val_acc": [],
        "train_loss": [], "val_loss": [],
    }

    for e in range(epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device
        )

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        is_best = val_acc > best_val_acc + 1e-4
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch_callback is not None:
            epoch_callback(
                e + 1, epochs,
                tr_loss, tr_acc, val_loss, val_acc,
                is_best=is_best, stopped_early=False,
            )

        if patience_counter >= early_stopping_patience:
            if epoch_callback is not None:
                epoch_callback(
                    e + 1, epochs,
                    tr_loss, tr_acc, val_loss, val_acc,
                    is_best=False, stopped_early=True,
                )
            break

    model.load_state_dict(best_state)
    return model, best_val_acc, history