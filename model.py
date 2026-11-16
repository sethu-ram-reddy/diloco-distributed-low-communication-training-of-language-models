"""
DiLoCo: Distributed Low-Communication Training of Language Models

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - init_model_params
import numpy as np

def init_model_params(input_dim, hidden_dim, output_dim, seed=0):
    rng = np.random.default_rng(seed)
    
    scale1 = np.sqrt(2.0 / input_dim)
    scale2 = np.sqrt(2.0 / hidden_dim)
    
    W1 = rng.standard_normal((input_dim, hidden_dim)).astype(np.float64) * scale1
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    
    W2 = rng.standard_normal((hidden_dim, output_dim)).astype(np.float64) * scale2
    b2 = np.zeros(output_dim, dtype=np.float64)
    
    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
    }

# Step 2 - relu
import numpy as np

def relu(x):
    return np.maximum(0, x)

# Step 3 - model_forward
import numpy as np

def model_forward(params, x):
    """Run the 2-layer MLP forward pass and stash intermediates for backprop."""
    z1 = x @ params['W1'] + params['b1']
    h1 = relu(z1)
    logits = h1 @ params['W2'] + params['b2']
    
    cache = {
        'x': x,
        'z1': z1,
        'h1': h1,
        'logits': logits,
    }
    return logits, cache

# Step 4 - softmax
import numpy as np

def softmax(logits):
    # Subtract row-wise max for numerical stability
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs

# Step 5 - cross_entropy_loss
import numpy as np

def cross_entropy_loss(logits, labels):
    probs = softmax(logits)
    batch_size = logits.shape[0]
    true_class_probs = probs[np.arange(batch_size), labels]
    loss = -np.mean(np.log(np.maximum(true_class_probs, 1e-300)))
    return float(loss)

# Step 6 - model_backward
import numpy as np

def model_backward(params, cache, labels):
    x = cache['x']
    z1 = cache['z1']
    h1 = cache['h1']
    logits = cache['logits']
    
    batch_size = logits.shape[0]
    
    # Softmax cross-entropy gradient averaged over batch
    dlogits = softmax(logits)
    dlogits[np.arange(batch_size), labels] -= 1.0
    dlogits /= batch_size
    
    # Output linear layer gradients
    dW2 = h1.T @ dlogits
    db2 = np.sum(dlogits, axis=0)
    
    # Backprop through ReLU
    dh1 = dlogits @ params['W2'].T
    dz1 = dh1 * (z1 > 0)
    
    # Input linear layer gradients
    dW1 = x.T @ dz1
    db1 = np.sum(dz1, axis=0)
    
    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2,
    }

# Step 7 - init_adamw_state
import numpy as np

def init_adamw_state(params):
    m = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
    v = {k: np.zeros_like(v, dtype=np.float64) for k, v in params.items()}
    return {
        'm': m,
        'v': v,
        't': 0,
    }

# Step 8 - update_adam_moments
def update_adam_moments(state, grads, beta1, beta2):
    state['t'] += 1
    t = state['t']
    
    for k in grads:
        g = grads[k]
        state['m'][k] = beta1 * state['m'][k] + (1 - beta1) * g
        state['v'][k] = beta2 * state['v'][k] + (1 - beta2) * (g ** 2)
        
    return state

# Step 9 - bias_correct_moments
import numpy as np

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

# Step 10 - adam_param_step
import numpy as np

def adam_param_step(params, m_hat, v_hat, lr, eps):
    new_params = {}
    for k in params:
        p = params[k]
        m = m_hat[k]
        v = v_hat[k]
        new_params[k] = p - lr * m / (np.sqrt(v) + eps)
    return new_params

# Step 11 - decoupled_weight_decay
import numpy as np

def decoupled_weight_decay(params, lr, weight_decay):
    decay_factor = 1.0 - lr * weight_decay
    new_params = {}
    for k, v in params.items():
        new_params[k] = v * decay_factor
    return new_params

# Step 12 - clone_params
import numpy as np

def clone_params(params):
    cloned = {}
    for k, v in params.items():
        cloned[k] = v.copy()
    return cloned

# Step 13 - scale_params
import numpy as np

def scale_params(params, scalar):
    scaled = {}
    for k, v in params.items():
        scaled[k] = (v * scalar).copy()
    return scaled

# Step 14 - subtract_params
import numpy as np

def subtract_params(params_a, params_b):
    diff = {}
    for k in params_a:
        diff[k] = params_a[k] - params_b[k]
    return diff

# Step 15 - average_params
import numpy as np

def average_params(params_list):
    if not params_list:
        return {}
        
    num_models = len(params_list)
    avg = {}
    
    for k in params_list[0]:
        total = sum(p[k] for p in params_list)
        avg[k] = total / num_models
        
    return avg

# Step 16 - iid_shard_dataset
import numpy as np

def iid_shard_dataset(x, y, num_workers, seed=0):
    rng = np.random.default_rng(seed)
    n_samples = x.shape[0]
    
    # Randomly permute indices reproducibly
    perm = rng.permutation(n_samples)
    
    # np.array_split automatically distributes remainder across the earliest shards
    index_shards = np.array_split(perm, num_workers)
    
    shards = []
    for idx in index_shards:
        shards.append((x[idx], y[idx]))
        
    return shards

# Step 17 - noniid_shard_dataset
import numpy as np

def noniid_shard_dataset(x, y, num_workers, num_classes, seed=0):
    rng = np.random.default_rng(seed)
    shards = []
    
    for w in range(num_workers):
        # Collect indices where label c satisfies c % num_workers == w
        idx = np.where((y % num_workers) == w)[0]
        
        if len(idx) > 0:
            perm = rng.permutation(len(idx))
            idx = idx[perm]
            
        shards.append((x[idx], y[idx]))
        
    return shards

# Step 18 - sample_worker_batch
import numpy as np

def sample_worker_batch(x_shard, y_shard, batch_size, rng):
    n = x_shard.shape[0]
    replace = batch_size > n
    indices = rng.choice(n, size=batch_size, replace=replace)
    return x_shard[indices], y_shard[indices]

# Step 19 - local_train_step
def local_train_step(params, adam_state, x_batch, y_batch, lr, beta1, beta2, eps, weight_decay):
    # 1. Forward pass & cache
    logits, cache = model_forward(params, x_batch)
    
    # 2. Compute loss
    loss = cross_entropy_loss(logits, y_batch)
    
    # 3. Backward pass to get gradients
    grads = model_backward(params, cache, y_batch)
    
    # 4. Update Adam moments (mutates or updates state)
    new_adam_state = {
        't': adam_state['t'],
        'm': {k: v.copy() for k, v in adam_state['m'].items()},
        'v': {k: v.copy() for k, v in adam_state['v'].items()},
    }
    update_adam_moments(new_adam_state, grads, beta1, beta2)
    
    # 5. Bias correction
    m_hat, v_hat = bias_correct_moments(new_adam_state, beta1, beta2)
    
    # 6. Adam parameter step
    updated_params = adam_param_step(params, m_hat, v_hat, lr, eps)
    
    # 7. Decoupled weight decay
    new_params = decoupled_weight_decay(updated_params, lr, weight_decay)
    
    return new_params, new_adam_state, loss

# Step 20 - inner_train_worker
import numpy as np

def inner_train_worker(params, x_shard, y_shard, num_inner_steps, batch_size, lr, beta1, beta2, eps, weight_decay, seed):
    # Clone params so the original is not mutated
    worker_params = clone_params(params)
    adam_state = init_adamw_state(worker_params)
    
    rng = np.random.default_rng(seed)
    total_loss = 0.0
    
    for _ in range(num_inner_steps):
        x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
        worker_params, adam_state, loss = local_train_step(
            worker_params, adam_state, x_batch, y_batch, 
            lr, beta1, beta2, eps, weight_decay
        )
        total_loss += loss
        
    mean_loss = total_loss / num_inner_steps
    return worker_params, float(mean_loss)

# Step 21 - init_outer_optimizer
import numpy as np

def init_outer_optimizer(params):
    momentum = {}
    for k, v in params.items():
        momentum[k] = np.zeros_like(v)
    return {'momentum': momentum}

# Step 22 - update_outer_momentum
import numpy as np

def update_outer_momentum(outer_state, outer_grad, momentum_coef):
    m = outer_state['momentum']
    for k in m:
        m[k] = momentum_coef * m[k] + outer_grad[k]
    return outer_state

# Step 23 - nesterov_param_update
import numpy as np

def nesterov_param_update(params, outer_state, outer_grad, outer_lr, momentum_coef):
    momentum_dict = outer_state['momentum'] if 'momentum' in outer_state else outer_state
    new_params = {}
    for k in params:
        p = params[k]
        m = momentum_dict[k]
        g = outer_grad[k]
        new_params[k] = p - outer_lr * (momentum_coef * m + g)
    return new_params

# Step 24 - compute_outer_gradient
def compute_outer_gradient(global_params, worker_params_list):
    avg_worker_params = average_params(worker_params_list)
    return subtract_params(global_params, avg_worker_params)

# Step 25 - run_diloco_round
def run_diloco_round(global_params, outer_state, worker_shards, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    worker_params_list = []
    worker_losses = []
    
    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']
    
    for i, (x_shard, y_shard) in enumerate(worker_shards):
        w_params, w_loss = inner_train_worker(
            global_params, x_shard, y_shard,
            num_inner_steps, batch_size,
            lr, beta1, beta2, eps, weight_decay,
            seed + i
        )
        worker_params_list.append(w_params)
        worker_losses.append(w_loss)
        
    outer_grad = compute_outer_gradient(global_params, worker_params_list)
    outer_state = update_outer_momentum(outer_state, outer_grad, momentum_coef)
    new_global_params = nesterov_param_update(global_params, outer_state, outer_grad, outer_lr, momentum_coef)
    
    return new_global_params, outer_state, worker_losses

# Step 26 - train_diloco
def train_diloco(initial_params, worker_shards, num_outer_rounds, num_inner_steps, batch_size, inner_hparams, outer_lr, momentum_coef, seed):
    params = clone_params(initial_params)
    outer_state = init_outer_optimizer(params)
    round_losses = []
    
    num_shards = len(worker_shards)
    
    for round_idx in range(num_outer_rounds):
        round_seed = seed + round_idx * max(1, num_shards)
        
        # If num_inner_steps is 0, inner_train_worker will cause a division by zero.
        # We can bypass inner training for 0 steps and manually return default worker states/losses.
        if num_inner_steps == 0:
            worker_losses = [0.0 for _ in worker_shards]
            # When inner steps are 0, parameters and outer state remain unchanged
        else:
            params, outer_state, worker_losses = run_diloco_round(
                params, outer_state, worker_shards,
                num_inner_steps, batch_size,
                inner_hparams, outer_lr, momentum_coef,
                round_seed
            )
        
        if worker_losses:
            mean_round_loss = float(sum(worker_losses) / len(worker_losses))
            round_losses.append(mean_round_loss)
        else:
            round_losses.append(0.0)
            
    return params, {'round_losses': round_losses}

# Step 27 - train_synchronous_baseline
import numpy as np

def train_synchronous_baseline(init_params, worker_shards, num_steps, batch_size, inner_hparams, seed=0):
    params = clone_params(init_params)
    adam_state = init_adamw_state(params)
    rng = np.random.default_rng(seed)
    
    lr = inner_hparams['lr']
    beta1 = inner_hparams['beta1']
    beta2 = inner_hparams['beta2']
    eps = inner_hparams['eps']
    weight_decay = inner_hparams['weight_decay']
    
    step_losses = []
    num_workers = len(worker_shards)
    
    for _ in range(num_steps):
        worker_grads = []
        worker_losses = []
        
        for x_shard, y_shard in worker_shards:
            x_batch, y_batch = sample_worker_batch(x_shard, y_shard, batch_size, rng)
            logits, cache = model_forward(params, x_batch)
            loss = cross_entropy_loss(logits, y_batch)
            grads = model_backward(params, cache, y_batch)
            
            worker_grads.append(grads)
            worker_losses.append(loss)
            
        if worker_grads:
            avg_grads = average_params(worker_grads)
            mean_loss = float(sum(worker_losses) / num_workers)
        else:
            avg_grads = {k: np.zeros_like(v) for k, v in params.items()}
            mean_loss = 0.0
            
        step_losses.append(mean_loss)
        
        # AdamW step using the averaged gradients
        update_adam_moments(adam_state, avg_grads, beta1, beta2)
        m_hat, v_hat = bias_correct_moments(adam_state, beta1, beta2)
        params = adam_param_step(params, m_hat, v_hat, lr, eps)
        params = decoupled_weight_decay(params, lr, weight_decay)
        
    return params, {'step_losses': step_losses}

# Step 28 - evaluate_loss
def evaluate_loss(params, x, y):
    logits, _ = model_forward(params, x)
    loss = cross_entropy_loss(logits, y)
    return float(loss)

# Step 29 - classification_accuracy (not yet solved)
# TODO: implement

# Step 30 - communication_savings (not yet solved)
# TODO: implement

