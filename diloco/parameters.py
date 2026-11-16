"""Operations on independent NumPy parameter dictionaries."""




def clone_params(params):
    cloned = {}
    for k, v in params.items():
        cloned[k] = v.copy()
    return cloned


def scale_params(params, scalar):
    scaled = {}
    for k, v in params.items():
        scaled[k] = (v * scalar).copy()
    return scaled


def subtract_params(params_a, params_b):
    diff = {}
    for k in params_a:
        diff[k] = params_a[k] - params_b[k]
    return diff


def average_params(params_list):
    if not params_list:
        return {}

    num_models = len(params_list)
    avg = {}

    for k in params_list[0]:
        total = sum(p[k] for p in params_list)
        avg[k] = total / num_models

    return avg
