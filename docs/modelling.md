# Structural statistical model

Checkpoint 5 adds a research-only structural layer over bounded synchronized 15-minute windows. It
does not fetch data, submit orders, size positions, generate signals, run a Kalman filter, or decide
which assets to trade.

## Chronology and reproducibility

Every estimator accepts a `ResearchWindow`, never an unrestricted future-containing frame. A fit
enforces:

```text
every input timestamp <= data_end <= fit_timestamp
```

The default primary window is 30 calendar days. Independent 14-day and 60-day fits are comparison
diagnostics; parameters are never averaged across windows. Missing synchronized rows remain visible
because window validity is based on calendar bounds, the actual observation count, and the maximum
internal gap. The default scheduler fits at most once per 24 hours using the latest completed model
bar at or before its fit time. It never retrospectively mutates an earlier immutable result.

Each result carries `ModelFitMetadata`, the exact bounded source-data fingerprint, explicit model
configuration, and a deterministic content-derived fit ID. Versioned JSON persistence stores arrays
as numeric lists and never depends on unsafe pickles.

The end-to-end clock remains:

```text
raw 5m completion -> synchronized 15m close t_k -> structural fit through t_k
-> possible future signal at t_k -> earliest execution in k+1
```

This package ends at the fit. It has no execution access.

## Integration diagnostics

For every BTC, ETH, SOL, XRP, and ADA log-price level `x_i,t` and first difference `Delta x_i,t`, the
layer reports finite/variance checks plus ADF and KPSS statistics, p-values, critical values, sample
counts, deterministic terms, library warnings, and explicit failures. V1 uses an intercept and no
linear trend for these diagnostics.

ADF and KPSS have different null hypotheses and can disagree. Failure to reject a unit root in a
level together with rejection in its first difference supports—but does not prove—an I(1)
description. No single test is converted into an automatic trading or rejection rule.

## Lag and deterministic specification

The levels VAR order is fitted for every `p` in 1 through 8. AIC, BIC, and HQIC are retained for
every successful candidate; BIC is the default selector to favour parsimony. The corresponding VECM
contains exactly `p - 1` lagged differences. This mapping is a named, directly tested function.

V1 has no deterministic linear trend in observed levels and uses a constant restricted to the
cointegration relation. The specification is explicit as Johansen `det_order=0` and statsmodels VECM
`deterministic="ci"`; it is not inherited from library defaults.

## Johansen rank and V1 scope

The Johansen result retains trace and maximum-eigenvalue statistics, their 90/95/99 percent critical
value tables, eigenvalues, the complete candidate eigenvector space, lag/deterministic settings,
sample count, and the configured significance level (5 percent by default). Trace rank is the V1
reported rank, while the maximum-eigenvalue rank is preserved as a separate diagnostic.

Rank is never forced:

- rank 0 is a valid result meaning no estimated cointegrating relation;
- rank 1 permits the full V1 VECM result; and
- rank above 1 retains diagnostics but is marked outside V1 tradability scope. No vector is chosen
  because it looks best in a backtest.

Johansen results are sensitive to the window, lag order, deterministic terms, and structural breaks.
In-sample cointegration is not evidence that a strategy will be profitable.

## Beta, alpha, Gamma, and ECM

For rank one, the fitted system is

```text
Delta x_t = alpha (beta' x_(t-1) + c) + sum_j Gamma_j Delta x_(t-j) + epsilon_t
```

Their roles remain separate:

- `beta` defines and measures the long-run equilibrium;
- `alpha` describes each variable's response to equilibrium error; and
- `Gamma_j` describes short-run propagation in lagged differences.

The native statsmodels beta, cointegration constant, and alpha remain untouched so that
`alpha beta'` is internally exact. A second reporting beta is L2-normalized and deterministically
sign-aligned by making its largest-absolute coefficient positive. The reporting alpha is inversely
rescaled, preserving `alpha beta'` and the conditional correction contribution. Tests cover positive,
negative, and arbitrary nonzero beta scaling.

The fitted object exposes native/reporting beta and alpha, alpha/beta standard errors, every Gamma
matrix and standard-error block, residuals, fitted values, and the residual covariance. Covariance
must be finite, symmetric, 5-by-5, and have positive diagonal variances.

The native equilibrium error is:

```text
ECM_t = beta' x_t + c
```

The result retains its full fitting-window series, current value, mean, standard deviation, median,
median absolute deviation, robust spread, and diagnostic z-score. These values are descriptive only.
For each asset it also reports native `alpha_i`, uncertainty, and the current correction contribution
`alpha_i * ECM_t`. Raw alpha magnitude is not an invariant leadership score, and alpha does not by
itself establish causal price leadership.

The decomposition utility reports error correction, Gamma-driven short-run contribution,
deterministic contribution, and their conditional-mean sum independently. Long-run alpha effects and
short-run Gamma effects must not be collapsed into one undocumented score.

Beta describes an equilibrium; it is not automatically a future trade-weight vector.

## Persistence and convergence

The empirical ECM diagnostic reports ADF, selected autocorrelations, variance, and an intercept-plus
AR(1) estimate. A finite half-life is computed only for `0 < |phi| < 1`:

```text
half_life = -log(2) / log(|phi|)
```

`|phi| >= 1` is explicitly non-mean-reverting under this diagnostic. Near-zero persistence is
reported as effectively immediate rather than clipped to a convenient value.

The VECM also reports the simplified persistence `1 + beta' alpha`. Because Gamma dynamics can make
that shortcut incomplete, the model separately evolves the full fitted deterministic recursion with
future shocks fixed to zero, traces expected ECM, and measures its first half-decay when one occurs.
An empirical versus model-implied difference of at least twofold is surfaced, never resolved by
choosing the more attractive estimate.

## Stability

Consecutive rank-one fits are compared using sign-aligned normalized-beta cosine similarity,
scale-compatible reporting-alpha similarity/change, rank, selected VAR order, and empirical and
deterministic half-life changes. Cross-window reports keep the 14-, 30-, and 60-day models distinct.
No hard stability gate is imposed before empirical distributions exist.

Structural breaks can invalidate a full-window equilibrium. Permanent synthetic tests include
independent random walks, a known rank-one VECM, a known dominant adjuster, one common stochastic
trend with multiple cointegrating relations, and a changed equilibrium whose beta comparison becomes
materially unstable.

## Methodological limits

Cointegration rank, coefficients, persistence, and significance are estimates, not economic facts.
Multiple cointegrating vectors need a later genuinely multivariate design and cannot be reduced
arbitrarily. Residual normality, heteroskedasticity, nonlinear behavior, multiple-testing corrections,
economic capacity, execution costs, and out-of-sample profitability are not established here.
Kalman filtering, state-space forecasting, sparse trade selection, signals, sizing, and risk rules
remain deliberately absent.
