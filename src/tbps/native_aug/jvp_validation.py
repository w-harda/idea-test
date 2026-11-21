"""校验 RaSa 的 FP32 方向导数。相减和比值用 FP64，训练导数仍由原自动微分产生。"""
import math

from .common import need

DEFAULT_STEP = 0.01
RELATIVE_LIMIT = 0.02
STABILITY_LIMIT = 0.01
FALLBACK_STEPS = (0.04, 0.08, 0.16, 0.32, 0.64)


def symmetric_difference(evaluate, step):
    positive = evaluate(step).double()
    negative = evaluate(-step).double()
    return (positive - negative) / (2 * step)


def relative_difference(value, reference):
    denominator = float(reference.norm())
    numerator = float((value - reference).norm())
    need(math.isfinite(denominator) and math.isfinite(numerator), 'finite directional check')
    if denominator == 0:
        return 0.0 if numerator == 0 else math.inf
    return numerator / denominator


def verify_directional_derivative(evaluate, derivative):
    """先用原步长。原检查失败时，独立于自动微分选择收敛的差分尺度。"""
    reference = derivative.double()
    initial = symmetric_difference(evaluate, DEFAULT_STEP)
    initial_error = relative_difference(initial, reference)
    if initial_error <= RELATIVE_LIMIT:
        return {
            'FD_method': 'centered',
            'FD_step': DEFAULT_STEP,
            'FD_initial_relative_error': initial_error,
            'FD_relative_error': initial_error,
        }

    previous_difference = None
    previous_estimate = None
    for step in FALLBACK_STEPS:
        difference = symmetric_difference(evaluate, step)
        if previous_difference is not None:
            estimate = (4 * previous_difference - difference) / 3
            if previous_estimate is not None:
                drift = relative_difference(previous_estimate, estimate)
                if drift <= STABILITY_LIMIT:
                    error = relative_difference(estimate, reference)
                    need(error <= RELATIVE_LIMIT,
                         f'directional derivative mismatch: initial={initial_error:.6g}, '
                         f'converged={error:.6g}, drift={drift:.6g}, h={step / 2:g}')
                    return {
                        'FD_method': 'richardson',
                        'FD_step': step / 2,
                        'FD_coarse_step': step,
                        'FD_initial_relative_error': initial_error,
                        'FD_relative_error': error,
                        'FD_scale_drift': drift,
                    }
            previous_estimate = estimate
        previous_difference = difference
    need(False, f'directional finite differences did not converge: initial={initial_error:.6g}')


def audit_fd_receipt(value):
    """独立审计验证门槛及记录的步长来路，无神经前向。"""
    initial = value['FD_initial_relative_error']
    error = value['FD_relative_error']
    need(value['zero_exact'] is True
         and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in (initial, error))
         and error <= RELATIVE_LIMIT, 'strict native JVP receipt')
    if value['FD_method'] == 'centered':
        need(value['FD_step'] == DEFAULT_STEP and initial == error, 'default finite difference receipt')
    else:
        coarse_step = value['FD_coarse_step']
        drift = value['FD_scale_drift']
        need(value['FD_method'] == 'richardson'
             and initial > RELATIVE_LIMIT
             and coarse_step in FALLBACK_STEPS[2:]
             and value['FD_step'] * 2 == coarse_step
             and type(drift) in (int, float) and 0 <= drift <= STABILITY_LIMIT,
             'converged multi-scale finite difference receipt')
