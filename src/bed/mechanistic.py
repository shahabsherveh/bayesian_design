"""Mechanistic forward models and synthetic problem builders for the showcase experiments.

Two models: an individualized pharmacokinetic/pharmacodynamic (PK/PD) model (one compartment
with first-order absorption, an effect compartment and an E_max effect), and the DIVIDE
diffusion-MRI signal model with biomarker targets. All quantities are closed form, so the forward
models are pure JAX functions and the filters obtain their Jacobians by automatic differentiation.

Conventions follow the package: designs ``x`` of shape ``(B, 1, 1, d_x)``, outputs of shape
``(B, 1, 1, d_y)``, parameters ``z`` as a flat vector or a ``(d, 1)`` column.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from bed._models import Model
from bed.data import Data

# Population values used as the prior mean (theophylline-like): k_a, k_e [1/h], V [L],
# k_e0 [1/h], E_max, EC_50 [mg/L], E_0 (effect units).
PKPD_POPULATION = jnp.array([1.5, 0.09, 35.0, 0.5, 100.0, 8.0, 10.0])
PKPD_LOG_SD = jnp.array([0.5, 0.3, 0.3, 0.5, 0.3, 0.5, 0.3])
PKPD_NAMES = ("k_a", "k_e", "V", "k_e0", "E_max", "EC_50", "E_0")
STEADY_STATE_DOSES = 1e4


class ClosedFormModel(Model):
    """A forward model given by a closed-form ``_single(z, x)`` for one design.

    Subclasses set ``output_dim`` and ``weight_size`` and implement ``_single``; batching and
    the Jacobian (by automatic differentiation) are shared.
    """

    def _single(self, z, x):
        raise NotImplementedError

    def __call__(self, z, x, **kwargs):
        z = jnp.asarray(z).reshape(-1)
        x = jnp.asarray(x)
        designs = x.reshape(-1, x.shape[-1])
        out = jax.vmap(lambda xi: self._single(z, xi))(designs)
        return out.reshape(designs.shape[0], 1, 1, self.output_dim)

    def jacobian(self, z, x):
        """Jacobian of the outputs with respect to ``z``, shape ``(B, d_y, d)``."""
        z = jnp.asarray(z).reshape(-1)
        x = jnp.asarray(x)
        designs = x.reshape(-1, x.shape[-1])
        jac = jax.vmap(lambda xi: jax.jacobian(lambda zz: self._single(zz, xi))(z))(designs)
        return jac.reshape(designs.shape[0], self.output_dim, z.size)

    def train(self, *args, **kwargs):
        return self


class PKPDModel(ClosedFormModel):
    """One-compartment oral PK with an effect compartment and an E_max effect.

    Design ``x = (D, tau, n, t)``: dose [mg], dosing interval [h], number of doses given
    (``n = 1`` single dose; a large ``n`` is steady state), time since the last dose [h].
    Parameters ``z = log(k_a, k_e, V, k_e0, E_max, EC_50, E_0)``; bioavailability is fixed to
    one, so ``V`` is the apparent volume ``V/F``. Output ``(log C, E)`` with ``C`` the plasma
    concentration [mg/L] and ``E`` the effect.
    """

    output_dim = 2
    weight_size = 7

    def __init__(self, concentration_floor=1e-6):
        self.concentration_floor = concentration_floor

    def _single(self, z, x):
        k_a, k_e, V, k_e0, e_max, ec50, e_0 = jnp.exp(z)
        D, tau, n, t = x
        # superposition factor of n identical doses for an exponential with rate k
        def acc(k):
            return (1 - jnp.exp(-n * k * tau)) / (1 - jnp.exp(-k * tau))
        amp = D * k_a / (V * (k_a - k_e))
        conc = amp * (acc(k_e) * jnp.exp(-k_e * t) - acc(k_a) * jnp.exp(-k_a * t))
        # effect-site concentration: first-order lag with rate k_e0 applied to each exponential
        def lag(k):
            return k_e0 / (k_e0 - k) * (acc(k) * jnp.exp(-k * t) - acc(k_e0) * jnp.exp(-k_e0 * t))
        conc_e = amp * (lag(k_e) - lag(k_a))
        effect = e_0 + e_max * conc_e / (ec50 + conc_e)
        return jnp.stack([jnp.log(jnp.maximum(conc, self.concentration_floor)), effect])


@dataclass
class PKPDProblem:
    data: Data
    model: PKPDModel
    prior_mean: jnp.ndarray       # (7, 1)
    prior_cov: jnp.ndarray        # (7, 7)
    measurement_error: jnp.ndarray  # (2, 2)
    z_true: jnp.ndarray           # (7,)


def make_pkpd_problem(seed=0, prior_scale=1.0, noise_sd=(0.15, 5.0), num_times=150,
                      doses=(100.0, 200.0, 400.0), pool_doses=(200.0, 400.0, 600.0),
                      pool_intervals=(6.0, 8.0, 12.0, 24.0), pool_times=(None, 2.0),
                      truth="prior", num_global=120):
    """Build the synthetic PK/PD design problem.

    Candidates: single-dose occasions ``(D, 24, 1, t)`` on a grid of doses and sampling times in
    [0.25, 24] h. Targets (pool): steady state at ``(D, tau)`` with observations at trough
    (``t = tau``, encoded as ``None`` in ``pool_times``) and at 2 h post-dose. Global test set:
    random single-dose designs. Labels of the candidates carry Gaussian noise with the stated
    standard deviations on ``log C`` and ``E``; pool and global labels are noise-free truths.
    ``truth`` is ``"prior"`` (draw ``z_true`` from the prior) or an explicit 7-vector of
    log-parameters.
    """
    key = jax.random.PRNGKey(seed)
    k_truth, k_noise, k_glob = jax.random.split(key, 3)
    model = PKPDModel()
    prior_mean = jnp.log(PKPD_POPULATION)
    prior_sd = PKPD_LOG_SD * jnp.sqrt(prior_scale)
    prior_cov = jnp.diag(prior_sd ** 2)
    if isinstance(truth, str) and truth == "prior":
        z_true = prior_mean + prior_sd * jax.random.normal(k_truth, prior_mean.shape)
    else:
        z_true = jnp.asarray(truth).reshape(-1)
    times = jnp.linspace(0.25, 24.0, num_times)
    x_train = jnp.array([[D, 24.0, 1.0, t] for D in doses for t in times])[:, None, None, :]
    x_pool = jnp.array([[D, tau, STEADY_STATE_DOSES, (tau if tt is None else tt)]
                        for D in pool_doses for tau in pool_intervals for tt in pool_times])[:, None, None, :]
    kd, kt = jax.random.split(k_glob)
    x_glob = jnp.stack([
        jnp.asarray(doses)[jax.random.randint(kd, (num_global,), 0, len(doses))],
        jnp.full((num_global,), 24.0), jnp.ones((num_global,)),
        jax.random.uniform(kt, (num_global,), minval=0.25, maxval=24.0)], axis=1)[:, None, None, :]
    noise_sd = jnp.asarray(noise_sd)
    y_train = model(z_true, x_train) + noise_sd * jax.random.normal(k_noise, (x_train.shape[0], 1, 1, 2))
    y_pool = model(z_true, x_pool)
    y_glob = model(z_true, x_glob)
    data = Data(x_train, y_train, x_pool, y_pool, x_glob, y_glob)
    return PKPDProblem(data, model, prior_mean.reshape(-1, 1), prior_cov, jnp.diag(noise_sd ** 2), z_true)


# ---------------------------------------------------------------------------------------------
# DIVIDE: diffusional variance decomposition, biomarker-oriented protocol design (candidate D)
# ---------------------------------------------------------------------------------------------
# Units: b in ms/um^2, diffusivities in um^2/ms, variances in um^4/ms^2, TE and T2 in ms.
DIVIDE_NAMES = ("log S_0", "log T_2", "log MD", "log V_I", "log V_A")
DIVIDE_BIOMARKERS = ("MD", "MK_I", "MK_A")
# Population prior for a brain region mixing white and grey matter (log-normal).
DIVIDE_PRIOR_MEAN = jnp.log(jnp.array([1.0, 75.0, 0.8, 0.05, 0.2]))
DIVIDE_PRIOR_SD = jnp.array([0.1, 0.2, 0.25, 0.6, 0.5])   # V_A: MK_A 0.35-2.5 at +-2 sd (grey to white matter)


def divide_biomarkers(z):
    """(MD, MK_I, MK_A) from one voxel's log-parameters."""
    MD, VI, VA = jnp.exp(z[2]), jnp.exp(z[3]), jnp.exp(z[4])
    return jnp.stack([MD, 3 * VI / MD ** 2, 3 * VA / MD ** 2])


def divide_biomarker_scale(prior_mean=DIVIDE_PRIOR_MEAN, prior_sd=DIVIDE_PRIOR_SD, num_samples=20000, seed=0):
    """Prior standard deviation of each biomarker and of each log-biomarker, used to standardise
    the targets. The log-biomarkers are exactly linear in z: log MD = z_2, log MK_I = log 3 + z_3 −
    2 z_2, log MK_A = log 3 + z_4 − 2 z_2, so their prior sds are available in closed form."""
    rng = np.random.default_rng(seed)
    Z = np.asarray(prior_mean) + np.asarray(prior_sd) * rng.normal(size=(num_samples, 5))
    G = np.asarray(jax.vmap(divide_biomarkers)(jnp.asarray(Z)))
    sd = np.asarray(prior_sd)
    log_scale = jnp.asarray([sd[2], np.sqrt(sd[3] ** 2 + 4 * sd[2] ** 2), np.sqrt(sd[4] ** 2 + 4 * sd[2] ** 2)])
    return jnp.asarray(G.std(axis=0)), log_scale


class DivideModel(ClosedFormModel):
    """Powder-averaged DIVIDE signal with T2 weighting, for ``V`` independent voxels sharing every
    acquisition. The diffusivity distribution seen by an axisymmetric encoding of shape ``b_Delta``
    has mean MD and variance ``mu_2 = V_I + b_Delta^2 V_A``. With ``form="gamma"`` (default, the
    DIVIDE fitting model) it is gamma distributed and the signal is its Laplace transform,
    ``S_0 exp(-TE/T_2) (1 + b mu_2 / MD)^(-MD^2 / mu_2)``, monotone in b for every parameter value;
    with ``form="cumulant"`` it is the second-order cumulant expansion
    ``S_0 exp(-TE/T_2) exp(-b MD + b^2 mu_2 / 2)``, which the gamma form matches to second order.

    Design ``x = (b, b_Delta, TE, kind)``. ``kind = 0`` is a signal acquisition and returns the
    ``V`` voxel signals. ``kind = k``
    for k in 1..3 is a virtual readout of the k-th standardised biomarker (MD, MK_I, MK_A) of every
    voxel, and k in 4..6 of the k-th standardised *log*-biomarker, which is linear in z, so that biomarker targets can be scored by the same closed form as observations. The
    latent state is the concatenation of the ``V`` voxel vectors ``(log S_0, log T_2, log MD,
    log V_I, log V_A)``."""

    def __init__(self, num_voxels=1, biomarker_scale=None, form="gamma", log_scale=None):
        self.V = int(num_voxels); self.output_dim = self.V; self.weight_size = 5 * self.V
        self.scale = jnp.ones(3) if biomarker_scale is None else jnp.asarray(biomarker_scale)
        self.log_scale = jnp.ones(3) if log_scale is None else jnp.asarray(log_scale)
        if form not in ("gamma", "cumulant"):
            raise ValueError(form)
        self.form = form

    def _voxel(self, zv, x):
        b, bD, TE, kind = x[0], x[1], x[2], x[3]
        S0, T2, MD, VI, VA = jnp.exp(zv)
        mu2 = VI + bD ** 2 * VA
        if self.form == "gamma":
            attenuation = jnp.exp(-(MD ** 2 / mu2) * jnp.log1p(b * mu2 / MD))
        else:
            attenuation = jnp.exp(-b * MD + 0.5 * b ** 2 * mu2)
        signal = S0 * jnp.exp(-TE / T2) * attenuation
        g = divide_biomarkers(zv)
        readouts = jnp.concatenate([g / self.scale, jnp.log(g) / self.log_scale])   # kinds 1-3 linear, 4-6 log
        k = jnp.asarray(kind, dtype=jnp.int32)
        return jnp.where(k == 0, signal, readouts[jnp.clip(k - 1, 0, 5)])

    def _single(self, z, x):
        Z = z.reshape(self.V, 5)
        return jax.vmap(lambda zv: self._voxel(zv, x))(Z)


@dataclass
class DivideProblem:
    data: Data
    model: DivideModel
    prior_mean: jnp.ndarray
    prior_cov: jnp.ndarray
    measurement_error: jnp.ndarray
    z_true: jnp.ndarray            # (V, 5)
    biomarker_scale: jnp.ndarray   # (3,)
    candidates: jnp.ndarray        # (n, 4) with kind = 0
    targets: jnp.ndarray           # (3, 4) with kind = 1..3


def divide_te_min(b, b_delta):
    """Shortest echo time [ms] at which an axisymmetric encoding of trace ``b`` fits: spherical
    encoding needs longer gradient waveforms than linear, and both grow with b. A linear stand-in
    for the hardware coupling, not a waveform calculation."""
    return jnp.where(jnp.abs(b_delta) < 0.5, 60.0 + 20.0 * b, 50.0 + 15.0 * b)


def make_divide_problem(seed=0, num_voxels=1, snr=40.0, b_values=tuple(round(0.1 * i, 1) for i in range(21)),
                        shapes=(1.0, 0.0), te_offsets=(0.0, 20.0, 40.0), num_global=30, prior_scale=1.0, form="gamma"):
    """Candidates: every (b, shape, TE_min + offset) with signal readout; targets: the three
    standardised biomarkers as virtual designs; truth: ``num_voxels`` draws from the population
    prior; noise sd 1/snr on the (powder-averaged) signal, S_0 = 1."""
    key = jax.random.PRNGKey(seed); k_truth, k_noise, k_glob = jax.random.split(key, 3)
    scale, log_scale = divide_biomarker_scale(DIVIDE_PRIOR_MEAN, DIVIDE_PRIOR_SD)
    V = int(num_voxels); model = DivideModel(V, scale, form=form, log_scale=log_scale)
    prior_sd = DIVIDE_PRIOR_SD * jnp.sqrt(prior_scale)
    prior_mean = jnp.tile(DIVIDE_PRIOR_MEAN, V); prior_cov = jnp.diag(jnp.tile(prior_sd ** 2, V))
    z_true = DIVIDE_PRIOR_MEAN + prior_sd * jax.random.normal(k_truth, (V, 5))
    cand = jnp.asarray([[b, s, float(divide_te_min(b, s)) + o, 0.0] for b in b_values for s in shapes for o in te_offsets])
    targets = jnp.asarray([[0.0, 0.0, 0.0, float(k)] for k in (1, 2, 3)])
    log_targets = jnp.asarray([[0.0, 0.0, 0.0, float(k)] for k in (4, 5, 6)])
    sd = 1.0 / snr
    y_train = model(z_true.reshape(-1), cand) + sd * jax.random.normal(k_noise, (cand.shape[0], 1, 1, V))
    idx = jax.random.choice(k_glob, cand.shape[0], (min(num_global, cand.shape[0]),), replace=False)
    data = Data(cand[:, None, None, :], y_train, targets[:, None, None, :], model(z_true.reshape(-1), targets),
                cand[idx][:, None, None, :], model(z_true.reshape(-1), cand[idx]))
    prob = DivideProblem(data, model, prior_mean.reshape(-1, 1), prior_cov, sd ** 2 * jnp.eye(V), z_true, scale, cand, targets)
    prob.log_targets = log_targets; prob.log_scale = log_scale
    return prob
