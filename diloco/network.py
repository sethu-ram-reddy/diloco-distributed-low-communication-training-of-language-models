"""NumPy MLP and its analytical gradients."""

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


def relu(x):
    return np.maximum(0, x)


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


def softmax(logits):
    # Subtract row-wise max for numerical stability
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs


def cross_entropy_loss(logits, labels):
    """Mean cross-entropy without underflowing the target probability."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    selected = shifted[np.arange(len(labels)), labels]
    return float(np.mean(log_normalizer - selected))


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


def evaluate_loss(params, x, y):
    logits, _ = model_forward(params, x)
    loss = cross_entropy_loss(logits, y)
    return float(loss)


def classification_accuracy(params, x, y):
    logits, _ = model_forward(params, x)
    if not len(x):
        return 0.0
    labels = y.argmax(axis=1) if y.ndim == 2 and y.shape[1] > 1 else y.reshape(-1)
    return float(np.mean(logits.argmax(axis=1) == labels))
