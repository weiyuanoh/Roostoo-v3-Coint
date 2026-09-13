# ECM-driven VECM Kalman state-space model

Checkpoint 6 adds sequential state estimation and non-trading forecast diagnostics. The Kalman
filter improves state estimation and forecasting. It does not change the definition of equilibrium.

## Frozen research hypothesis and scope

The fixed hypothesis is that, during rank-one regimes in the five-coin system, equilibrium errors
are corrected asymmetrically: some assets perform most of the correction while others behave more
like weakly adjusting leaders or anchors. A later checkpoint may test a minimal corrector-versus-
anchor expression against the full Johansen basket.

This layer does not classify correctors, leaders, or anchors. It does not select assets, construct a
trade, optimize a portfolio, size risk, define entry/exit thresholds, generate an order, or calculate
strategy PnL. It only estimates and forecasts the state implied by one immutable structural fit.

## Rank gate and fixed structural economics

The ECM-driven filter is active only when the current structural fit has rank one. Rank zero and
rank above one return structured inactive states. When rank leaves one, the active filter is closed
and its final checkpoint is retained for diagnostics; stale dynamics are not allowed to forecast.
When rank returns to one, a new regime is initialized from the new immutable fit.

Within one active regime, native beta, alpha, every Gamma matrix, the cointegration constant,
deterministic terms, and residual covariance remain fixed. The filter estimates latent state only.
It never changes beta to make the current ECM smaller or to rescue a broken relationship. A new beta
can enter solely through a new Checkpoint 5 structural fit.

## VECM-to-state-space derivation

For five log prices and a levels VAR order `p`, write `m = p - 1`:

```text
Delta x_(t+1) = Pi x_t + Gamma_1 Delta x_t + ...
                + Gamma_m Delta x_(t-m+1) + a + epsilon_(t+1)

Pi = alpha beta'
a  = alpha c + d
```

Here `c` is the restricted cointegration constant and `d` contains deterministic terms outside the
relation. The augmented state is:

```text
s_t = [x_t, Delta x_t, Delta x_(t-1), ..., Delta x_(t-m+1)]'
```

Its dimension is `5p`, derived from the fitted order rather than hard-coded. For `p=1`, there are no
difference blocks and the state contains only `x_t`.

The first-order transition is:

```text
s_(t+1) = F s_t + b + G epsilon_(t+1)
```

The price row of `F` contains `[I + Pi, Gamma_1, ..., Gamma_m]`. When difference blocks exist, the
newest-difference row contains `[Pi, Gamma_1, ..., Gamma_m]`; lower blocks shift prior differences
back one lag. The affine vector places `a` in both new-price and newest-difference blocks. Permanent
tests prove that this transition equals direct VECM recursion for `p=1`, `p=2`, and `p=4`.

The representation retains a separate decomposition at every state:

```text
error correction = alpha (beta' x_t + c)
short run         = sum Gamma_j Delta x_(t-j+1)
deterministic     = d
expected change   = their sum
```

This keeps the future alpha/ECM corrector evidence separate from Gamma-driven short-run propagation.

## Process noise: G and Q

The structural innovation is the fitted VECM residual:

```text
epsilon_t ~ N(0, Sigma_epsilon)
```

The same innovation enters `Delta x_t` and, through `x_t = x_(t-1) + Delta x_t`, current prices.
Accordingly, `G` has identity blocks in the new-price and newest-difference rows and zeros elsewhere:

```text
Q_state = G Sigma_epsilon G'
```

This produces required price/difference cross-covariance and is generally singular in the untouched
lag blocks. Dimension, finiteness, symmetry, diagonal variances, and PSD behavior are validated.
Materially invalid covariance raises an error; it is never replaced by an arbitrary diagonal matrix.

## Measurement model and interpretation of R

Completed exchange log-price closes measure the latent VECM-consistent efficient state:

```text
y_t = H s_t + v_t
H   = [I_5, 0, ..., 0]
v_t ~ N(0, R)
```

The close is not declared wrong. `R` represents short-horizon/microstructure deviation around the
slower structural state. Its configuration is recorded and supports:

- a diagonal fraction of fitted residual variances (default `0.01`);
- an explicit five-element diagonal; or
- an explicit validated 5-by-5 covariance.

The default is a modelling convention, not a tuned trading parameter. A predefined sensitivity grid
may diagnose behavior, but no value is selected using Sharpe, PnL, or hypothetical trades.

## Initialization, predict/update chronology, and handoff

Initialization requires strictly chronological bounded observations ending exactly at the
initialization timestamp. Current latent prices start from the latest observed log prices unless a
compatible filtered price estimate is explicitly carried during a rank-one-to-rank-one refit.
Difference blocks are reconstructed from observed history through that timestamp. Initial covariance
uses explicit residual-covariance blocks and a recorded multiplier.

For completed bar `k`:

```text
fit.data_end <= t_k
posterior through k-1
-> predict s_(k|k-1)
-> receive y_k
-> update s_(k|k)
-> expose output at t_k
-> possible future signal at t_k
-> earliest execution in k+1
```

Prediction is exposed independently, making it testable that `y_k` has not yet been consumed.
Changing observations after `k` cannot change output or forecasts at `k`.

At a new rank-one fit, the manager consumes the current completed observation under the old regime,
closes it, builds the new matrices, reconstructs difference lags from bounded observations, and
carries only the compatible five-price posterior and its 5-by-5 covariance. Lag blocks and their
covariance are rebuilt. A lag-order change therefore changes state dimension safely. Returning from
an inactive regime uses deterministic reinitialization rather than a stale filtered price.

Each transition records old/new fit IDs, ranks, layout dimensions, carry behavior, and covariance
mapping.

## Auditable filter outputs

Every update retains timestamp, fit ID and data end, rank, filter/layout versions, configuration,
prior and posterior means/covariances, observation and predicted observation, innovation,
standardized innovation, innovation covariance, Kalman gain, `R`, `G`, structural residual
covariance, `Q`, and status. Singular innovations, incompatible dimensions, non-finite values,
materially non-PSD covariance, insufficient history, future fits, and exploding state fail loudly.

Observed and filtered ECM use the same native structural beta and constant:

```text
observed_ECM_t = beta' y_t + c
filtered_ECM_t = beta' x*_t + c
```

Both diagnostic z-scores use only the immutable fit-window ECM mean and standard deviation. Both
also expose `alpha_i * ECM_t` in canonical asset order without classifying any asset.

## One-step and multi-step forecasts

From posterior state `k`, one-step expected change retains its alpha/ECM, Gamma, and deterministic
components. Multi-step means and covariances propagate `F`, `b`, and `Q` without future measurement
updates. Supported diagnostic horizons include 4, 16, 32, 64, and 96 fifteen-minute steps—1, 4, 8,
16, and 24 hours—plus a rounded active empirical half-life within configured limits.

Every horizon exposes expected latent price, validated 5-by-5 price covariance, and the full expected
ECM path. Two return-like quantities deliberately remain distinct:

```text
latent-state evolution       = E_k[x*_(k+h)] - x*_k
observed-market-relative     = E_k[x*_(k+h)] - y_k
```

Neither is designated a trading forecast here. Under the same state and zero future shocks, the ECM
path equals Checkpoint 5 deterministic VECM simulation.

## Baselines and diagnostics

The random-walk Kalman baseline uses the same observations, structural residual covariance as its
simple process covariance, and the same `R`, but its conditional mean does not restore equilibrium.
The plain VECM baseline initializes from observed levels/differences and propagates the structural
system without filtering. Chronological forecast records compare:

```text
A. random-walk KF
B. plain VECM
C. ECM-driven VECM-KF
```

RMSE, MAE, directional accuracy, ECM forecast error, and cross-sectional relative-return error are
computed only after the target timestamp becomes historical. They are forecast diagnostics, never
strategy PnL.

Innovation summaries retain per-asset mean, MAE, RMSE, standardized mean, and autocorrelation. A
synthetic broken equilibrium deliberately produces worse persistent innovations while beta remains
unchanged. Such failure is evidence against the active model, not something the filter should erase.

## Persistence and restart

Atomic versioned JSON checkpoints retain timestamp, active fit ID/data end, rank/status, layout and
filter versions, mean, covariance, last observation, full filter/R configuration, and initialization
method. Regime checkpoints also retain inactive rank state and the final active filter checkpoint.
Restart requires the exact structural fit and compatible configuration. Continuous execution and
save/reload/continue paths are numerically identical in permanent tests. No pickle is used.

## Research limitations

Filtering cannot validate the frozen hypothesis, establish causality, or prove profitability.
Covariance and observation-noise assumptions remain model choices. Gaussian linear updates do not
capture jumps, changing volatility, exchange fragmentation, or nonlinear transitions. The current
diagnostic comparison is short and must not be treated as model selection. In particular, a simpler
baseline outperforming the ECM-driven model is a legitimate result, not a reason to tune against
future strategy performance.
