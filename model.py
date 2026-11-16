"""Compatibility imports for the original Deep-ML exercise functions."""

from diloco.network import (
    classification_accuracy,
    cross_entropy_loss,
    evaluate_loss,
    init_model_params,
    model_backward,
    model_forward,
    relu,
    softmax,
)
from diloco.parameters import (
    average_params,
    clone_params,
    scale_params,
    subtract_params,
)
from diloco.optim import (
    adam_param_step,
    bias_correct_moments,
    compute_outer_gradient,
    decoupled_weight_decay,
    init_adamw_state,
    init_outer_optimizer,
    nesterov_param_update,
    update_adam_moments,
    update_outer_momentum,
)
from diloco.data import (
    iid_shard_dataset,
    noniid_shard_dataset,
    sample_worker_batch,
)
from diloco.training import (
    communication_savings,
    inner_train_worker,
    local_train_step,
    run_diloco_round,
    train_diloco,
    train_synchronous_baseline,
)

__all__ = [
    "adam_param_step",
    "average_params",
    "bias_correct_moments",
    "classification_accuracy",
    "clone_params",
    "communication_savings",
    "compute_outer_gradient",
    "cross_entropy_loss",
    "decoupled_weight_decay",
    "evaluate_loss",
    "iid_shard_dataset",
    "init_adamw_state",
    "init_model_params",
    "init_outer_optimizer",
    "inner_train_worker",
    "local_train_step",
    "model_backward",
    "model_forward",
    "nesterov_param_update",
    "noniid_shard_dataset",
    "relu",
    "run_diloco_round",
    "sample_worker_batch",
    "scale_params",
    "softmax",
    "subtract_params",
    "train_diloco",
    "train_synchronous_baseline",
    "update_adam_moments",
    "update_outer_momentum",
]
