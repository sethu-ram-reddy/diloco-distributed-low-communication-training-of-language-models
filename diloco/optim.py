"""Local AdamW and the server-side Nesterov update."""

import numpy as np

from .parameters import average_params, subtract_params


def init_adamw_state(params):
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    return {
        'm': m,
        'v': v,
        't': 0,
    }


def update_adam_moments(state, grads, beta1, beta2):
    state['t'] += 1

    for k in grads:
        g = grads[k]
        state['m'][k] = beta1 * state['m'][k] + (1 - beta1) * g
        state['v'][k] = beta2 * state['v'][k] + (1 - beta2) * (g ** 2)

    return state


def bias_correct_moments(state, beta1, beta2):
    t = state['t']
    bias_correction1 = 1.0 - (beta1 ** t)
    bias_correction2 = 1.0 - (beta2 ** t)

    m_hat = {}
    v_hat = {}

    for k in state['m']:
        m_hat[k] = state['m'][k] / bias_correction1
        v_hat[k] = state['v'][k] / bias_correction2

    return m_hat, v_hat


def adam_param_step(params, m_hat, v_hat, lr, eps):
    new_params = {}
    for k in params:
        p = params[k]
        m = m_hat[k]
        v = v_hat[k]
        new_params[k] = p - lr * m / (np.sqrt(v) + eps)
    return new_params


def decoupled_weight_decay(params, lr, weight_decay):
    decay_factor = 1.0 - lr * weight_decay
    new_params = {}
    for k, v in params.items():
        new_params[k] = v * decay_factor
    return new_params


def init_outer_optimizer(params):
    momentum = {}
    for k, v in params.items():
        momentum[k] = np.zeros_like(v)
    return {'momentum': momentum}


def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    m = outer_state['momentum']
    for k in m:
        m[k] = momentum_coef * m[k] + outer_grad[k]
    return outer_state


def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    momentum_dict = outer_state['momentum'] if 'momentum' in outer_state else outer_state
    new_params = {}
    for k in params:
        p = params[k]
        m = momentum_dict[k]
        g = outer_grad[k]
        new_params[k] = p - outer_lr * (momentum_coef * m + g)
    return new_params


def compute_outer_gradient(global_params, worker_params_list):
    avg_worker_params = average_params(worker_params_list)
    return subtract_params(global_params, avg_worker_params)


def adamw_step(params, state, grads, lr, beta1, beta2, eps, weight_decay):
    """Apply decoupled decay to the old weights and then the adaptive update."""
    update_adam_moments(state, grads, beta1, beta2)
    m_hat, v_hat = bias_correct_moments(state, beta1, beta2)
    decayed = decoupled_weight_decay(params, lr, weight_decay)
    return adam_param_step(decayed, m_hat, v_hat, lr, eps)
