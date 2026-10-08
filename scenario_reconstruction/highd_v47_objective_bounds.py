"""Fixed-bet, objective-level bounds for independent heterogeneous roots.

The bet and root coefficients MUST be fixed using separate training evidence.
These are fixed-budget bounds, not confidence sequences for optional stopping.
Leaves/actions within a branching tree are one root, never independent data.

For nonnegative independent X_i and m=sum E X_i:
 E[exp(-lambda*m)*prod(1+lambda*X_i)] <= 1.
For 0<=X_i<=b_i and lambda*max(b_i)<1:
 E[exp(+lambda*m)*prod(1-lambda*X_i)] <= 1.
Independence factors expectations and 1+u<=exp(u), 1-u<=exp(-u).
Markov inversion gives the one-sided sum bounds below. The formula is derived
here; the underlying nonnegative-bet/Markov principle is standard e-testing.
"""
import numpy as np


def _observations(values, rate, error):
    x = np.asarray(values, float)
    if (x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 0)
            or not np.isfinite(rate) or rate <= 0 or not 0 < error < 1):
        raise ValueError('Invalid independent objective-root observations or fixed bet')
    return x


def benefit_lower(values, rate, error):
    """Lower bound on sum E X_i, for independent nonnegative X_i.

    No bounded support is required. The rate must NOT be optimized on values.
    """
    x = _observations(values, rate, error)
    products = rate*x
    if not np.isfinite(products).all():
        raise FloatingPointError('Nonnegative bet overflow')
    score = float(np.log1p(products).sum())
    return dict(lower=max(0., (score-np.log(1/error))/rate),
                point=float(x.sum()), log_product=score, rate=float(rate), error=float(error))


def loss_upper(values, bounds, rate, error):
    """Upper bound on sum E X_i, 0<=X_i<=b_i, independent non-IID roots.

    All-negative observations retain a positive upper uncertainty allowance.
    Bounds are deterministic design quantities, not observed maxima.
    """
    x = _observations(values, rate, error); b = np.asarray(bounds, float)
    if (b.shape != x.shape or not np.isfinite(b).all() or np.any(b < 0)
            or np.any(x > b) or np.any(rate*b >= 1)):
        raise ValueError('Loss bet must stay positive over every designed root support')
    score = float(np.log1p(-rate*x).sum())
    return dict(upper=min(float(b.sum()), (np.log(1/error)-score)/rate),
                point=float(x.sum()), log_product=score, rate=float(rate), error=float(error),
                maximum_root_bound=float(b.max()) if len(b) else 0.)


def objective_difference(benefit, loss):
    """M2(q)-M2(alpha)=loss-benefit; the target and action Q stay fixed."""
    upper = loss['upper']-benefit['lower']
    return dict(M2_difference_point=loss['point']-benefit['point'],
                M2_difference_upper=upper,
                improvement_certified=bool(upper < 0), benefit=benefit, loss=loss,
                family_error=benefit['error']+loss['error'])


def choose_pilot_bet(pilot_batches, grid, error):
    """Choose one rate by the worst pilot-batch bound, BEFORE evaluation.

    Returned pilot bounds are fitting diagnostics, not coverage certificates.
    """
    rates = np.asarray(grid, float)
    if (rates.ndim != 1 or not len(rates) or not np.isfinite(rates).all()
            or np.any(rates <= 0) or np.any(np.diff(rates) <= 0) or not pilot_batches):
        raise ValueError('Invalid locked pilot bet grid')
    # Use the untruncated lower expression to make an all-zero pilot choose the
    # least pessimistic fixed rate; final bounds still enforce nonnegativity.
    scores = [min((benefit_lower(x, r, error)['log_product']-np.log(1/error))/r
                  for x in pilot_batches) for r in rates]
    index = int(np.argmax(scores))
    return dict(rate=float(rates[index]), grid_index=index,
                worst_pilot_lower_expression=float(scores[index]),
                interpretation='Training-only bet selection; pilot is not evaluation data.')
