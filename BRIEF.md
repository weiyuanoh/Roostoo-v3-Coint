# Engineering and Research Brief

This document is the research/engineering handoff for the TARGET repository. It describes the
current working tree as inspected on **2026-09-17**, including the local, Git-ignored research data.
It is deliberately not a product README: it separates what the code implements, what recent data
shows, what the research assumes, and what remains only a proposal.

The terms below are used precisely throughout:

- **Implemented** — exists in source and is covered by the current test suite.
- **Empirical observation** — produced by a bounded, chronological run over the local dataset; it
  is evidence, not a theorem or a profitability claim.
- **Research hypothesis** — the claim the project is designed to test, not an established fact.
- **Modelling assumption** — an explicit simplification or specification choice.
- **Proposal** — a possible next design that is not implemented and must not be treated as frozen.

## Executive Summary

This is not a generic “cointegration trading bot.” It is a research system for a narrower question:

> When BTC, ETH, SOL, XRP, and ADA exhibit one multivariate long-run equilibrium, is a displacement
> from that equilibrium corrected asymmetrically, and can that correction eventually be expressed
> with fewer assets than the full Johansen cointegrating basket?

The research universe is the five canonical assets **BTC, ETH, SOL, XRP, ADA**. Research data
currently uses one common quote, **USDT**, and deterministic, synchronized 15-minute bars derived
from completed Binance 5-minute candles. The intended research chain is:

```text
five-coin synchronized system
    -> rank-one structural regime
    -> equilibrium displacement (ECM)
    -> asymmetric error correction
    -> dominant correcting asset(s)
    -> economically defensible counter-leg
    -> minimal long-short expression
    -> comparison with the full Johansen beta basket
```

The central distinction is:

- the Johansen/VECM vector $\beta$ **defines the equilibrium**;
- it is **not assumed to be the optimal trading portfolio**.

The repository contains mature infrastructure through structural role diagnostics: market-data
collection and validation, signed long/short accounting, deterministic next-bar backtesting,
bounded research panels, Johansen/VECM estimation, an ECM-driven fixed-parameter Kalman layer, and
scale-invariant correction-role analysis. It does **not** contain an entry/exit strategy, position
sizing, a risk policy, a strategy PnL study, a production execution coordinator, or live trading
commands.

The latest local evidence is mixed and must be preserved as such:

- The latest 14/30/60-day fits have ranks **0/1/1**. Rank one is therefore regime- and
  window-dependent, not a permanent property of these assets.
- In six recent rank-one daily 30-day fits, correction was strongly asymmetric: XRP was the
  dominant restoring asset in 6/6 fits, with mean correction share about 67.65%; SOL contributed
  about 31.64% and qualified as a secondary corrector in 3/6 fits.
- The original conservative counter-leg hypothesis was not supported in that sample. Every asset’s
  $\alpha$ was significant; no weakly adjusting asset also satisfied the Gamma leadership rule.
  The result was `NO_CREDIBLE_ANCHOR` in 6/6 rank-one fits and no pair or minimal-basket candidate.
- In the limited recent CP6 forecast comparison, the ECM-driven Kalman model behaved almost
  identically to the plain VECM and did not beat a random-walk Kalman baseline on aggregate absolute
  log-price RMSE. This does not prove the relative-convergence hypothesis false, but it is adverse
  evidence against a stronger absolute-price-forecasting claim.

The project is consequently paused **before the first trading-strategy backtest**. The unresolved
research-design question is the counter-leg. One current proposal is to use opposite structural
poles of $\kappa_i=\beta_i\alpha_i$—the strongest credible restorer against the asset at the other
end of the $\kappa$ spectrum—rather than require a statistically weak adjuster. That rule is **not
implemented or frozen**. It requires an explicit checkpoint/specification before anyone codes it.

The engineering design exists to keep this experiment honest. Historical information is bounded,
structural fits are immutable, rank gates are explicit, exchange actions are separated from signed
inventory accounting, and simulated fills occur no earlier than the next bar. These constraints are
research controls, not incidental code style.

## Core Research Hypothesis

Let the synchronized log-price vector be

$$
x_t = [\log BTC_t,\ \log ETH_t,\ \log SOL_t,\ \log XRP_t,\ \log ADA_t]' .
$$

In a rank-one regime, the fitted equilibrium displacement is

$$
ECM_t = \beta' x_t + c,
$$

and the VECM is

$$
\Delta x_t = \alpha ECM_{t-1}
  + \sum_{j=1}^{p-1}\Gamma_j\Delta x_{t-j}
  + d + \epsilon_t .
$$

Interpret these objects separately:

- $\beta$ defines the long-run equilibrium relation.
- $ECM_t$ measures displacement from that fitted relation.
- $\alpha$ describes each asset’s long-run response to disequilibrium.
- $\Gamma_j$ describes conditional short-run lagged dynamics.
- $\Sigma_\epsilon$ is the VECM residual covariance.

For asset $i$, define

$$
\kappa_i = \beta_i\alpha_i .
$$

The asset’s error-correction contribution to the change in the ECM is proportional to

$$
\Delta ECM_i^{(EC)} = \beta_i\alpha_i ECM_t = \kappa_i ECM_t .
$$

Under compatible rescaling $\beta\mapsto q\beta$ and $\alpha\mapsto\alpha/q$, $\kappa_i$ is
unchanged. This makes $\kappa$ a useful scale-invariant diagnostic of which assets structurally
restore the equilibrium. In the implemented sign convention:

- $\kappa_i<0$ is restoring;
- $\kappa_i>0$ is individually non-restoring/destabilizing;
- the system-level persistence is related to $1+\beta'\alpha$, with full $\Gamma$ dynamics retained
  for deterministic convergence simulation.

The **hypothesis** is that restoration may be concentrated in a small subset of assets. It is not an
assumption that concentration always exists, that a correcting asset is profitable to trade, or that
$\beta$ supplies final trade weights.

## How the Hypothesis Has Evolved

The original minimal expression was:

```text
dominant corrector  versus  weakly adjusting leader/anchor
```

The intended anchor had two properties: statistically weak participation in long-run adjustment
($\alpha$ not distinguishable from zero and low correction share), plus Gamma-based evidence that
its lagged return helped predict the dominant corrector. CP7 implements this conservative rule.

The recent six-fit result was:

- strongly asymmetric correction was visible;
- no asset was a weak adjuster;
- therefore no asset passed the leader/anchor rule.

The anchor hypothesis is **not validated**, and no tradable pair has been established.

### Open proposal: opposite structural pole

A current **research proposal**, not an implementation, is:

$$
C=\arg\min_i \kappa_i,
\qquad
A=\arg\max_{j\ne C}\kappa_j,
$$

subject to explicit credibility checks that have not yet been specified. Here $C$ is the strongest
restorer and $A$ is the opposite end of the response spectrum. If all $\kappa$ values are negative,
$A$ would be the weakest restorer, not a genuinely non-restoring asset. Gamma evidence might become
supporting evidence rather than mandatory eligibility.

Do not silently implement this rule. It changes the research definition of the counter-leg and must
be frozen in a new specification first.

## What Success and Failure Mean

Positive PnL is not the only valid outcome, and in-sample cointegration is not itself success.

### Outcome A — the hypothesis is supported

Rank-one regimes occur often enough; the fitted equilibrium and correction roles are sufficiently
stable; a small number of assets consistently perform restoration; and a pre-specified minimal
expression captures subsequent relative convergence after costs in walk-forward testing.

### Outcome B — the equilibrium exists but minimal expression fails

Johansen/VECM diagnostics find meaningful rank-one mean reversion, but no stable small expression
captures it. The full $\beta$ basket may be the more faithful representation.

### Outcome C — asymmetry exists but is not tradable

Correction roles are measurable, but the move is too small, slow, unstable, or expensive after
next-bar timing, fees, spread, slippage, and financing.

### Outcome D — the regime is too unstable

Rank changes, rotating $\beta$, unstable $\kappa$ shares, structural breaks, or invalid persistence
make systematic use inappropriate.

### Outcome E — no usable edge

The system is statistically interesting but has no robust out-of-sample trading value. Engineering
must not tune this conclusion away.

## Rank-One Regime Gate

V1 intentionally supports only one equilibrium:

- **rank 0:** no identified cointegrating equilibrium; the V1 ECM regime is inactive.
- **rank 1:** one equilibrium; VECM, ECM-driven state space, and role diagnostics may operate.
- **rank > 1:** multiple equilibria are retained diagnostically but are outside V1 scope.

The code never chooses a convenient vector from a higher-rank space merely to keep the system
active. `StructuralFitResult.rank_scope`, `build_vecm_state_space()`, and
`StructuralRegimeManager` enforce the gate. Leaving rank one deactivates the ECM-driven filter;
returning to rank one creates a new regime under a new immutable fit.

Do not write “the five cryptocurrencies are cointegrated.” The defensible statement is: **under the
current window, lag, deterministic specification, and significance level, the system sometimes
exhibits a usable rank-one equilibrium**. A fresh bounded run at 2026-09-13 00:00 UTC produced:

| Window | Observations | BIC-selected VAR order | Johansen trace rank |
|---:|---:|---:|---:|
| 14 days | 1,344 | 1 | 0 |
| 30 days | 2,880 | 2 | 1 |
| 60 days | 5,760 | 3 | 1 |

Across the eight daily 30-day fits from 2026-09-06 through 2026-09-13, ranks were `2, 2, 1, 1,
1, 1, 1, 1`. All eight selected VAR order 2. This is direct evidence of regime dependence.

## Data Architecture and Current Coverage

### Universe and symbol boundary

`ResearchUniverse` fixes the canonical asset order `(BTC, ETH, SOL, XRP, ADA)` and requires one
configurable quote currency. Current mappings are:

| Canonical asset | Research pair | Binance symbol |
|---|---|---|
| BTC | BTC/USDT | BTCUSDT |
| ETH | ETH/USDT | ETHUSDT |
| SOL | SOL/USDT | SOLUSDT |
| XRP | XRP/USDT | XRPUSDT |
| ADA | ADA/USDT | ADAUSDT |

Research USDT symbols are distinct from Roostoo’s documented USD examples. The exchange adapter,
not model code, owns symbol translation.

### Pipeline

The canonical source is completed Binance 5-minute OHLCV. It is normalized to UTC, validated, and
stored under `data/research/raw/<QUOTE>/`. Derived 15-minute bars live separately under
`data/research/derived/<QUOTE>/`; raw data remains the source of truth.

Each 15-minute interval is UTC aligned and contains exactly three consecutive completed 5-minute
bars. For `[10:00, 10:15)`, the bar is observable at 10:15. A missing or misaligned constituent
invalidates the model bar; partial aggregation and price interpolation are prohibited.

The synchronized panel retains a timestamp only when all five assets have the same completed
15-minute interval. There is no forward-fill. `ResearchPanel.window()` exposes bounded observation-
count or calendar windows ending at an explicit completed timestamp. Raw-source and panel SHA-256
fingerprints make the input set reproducible. Historical bootstrap and incremental overlap handling
must reconstruct the same final panel.

### Local coverage verified during this handoff

- requested raw interval: 2025-09-13 00:00 UTC through 2026-09-13 00:00 UTC;
- per asset: 105,120 completed 5-minute rows and 35,040 valid 15-minute rows;
- synchronized panel: 2025-09-13 00:15 UTC through 2026-09-13 00:00 UTC;
- synchronized rows: 35,040;
- missing 5-minute intervals: 0 for every asset;
- excluded/forming 15-minute bars: 0;
- dropped panel timestamps: 0;
- source fingerprint:
  `24d01c986815742b64424f46b88f373ecbad44bbd3be54b2f7b69f3db214d7dc`;
- panel fingerprint:
  `df1e01146f94ca6cbab3e14bc92a066a7c7401f9e1f4d9961315a2fd19dfc524`.

The `data/` tree is Git-ignored. These numbers describe this workspace, not data guaranteed to exist
in a fresh clone.

## Hard No-Lookahead Invariant

This is a global research control:

```text
information through close[k]
    -> model/filter/role/signal at k
    -> earliest execution in bar k+1
```

Formally, $S_k=f(I_k)$ where $I_k$ contains information no later than completed close $t_k$, and
execution must satisfy

$$
model\_fit\_end \le filter\_time \le signal\_time < execution\_time.
$$

An order generated after observing close $k$ may not use bar $k$ open, high, low, close, or volume
for execution. The default market fill is open $k+1$ plus configured costs. A newly created limit
order can use only bar $k+1$ or later ranges.

The boundary applies to research windows, integration tests, lag selection, Johansen, VECM, ECM
normalization, structural stability, Kalman predict/update, role labels, future signal logic, and
execution. Strategies receive `MarketHistoryView`, an immutable prefix ending at the current bar,
not the future-containing dataframe.

A permanent regression pattern constructs datasets A and B that are identical through $t_k$ but
materially different afterward. Bounded data, fit parameters, filter state, and role results at
$t_k$ must remain identical. Forming bars, unsynchronized timestamps, and future rows are separately
excluded.

## Completed Checkpoints

### CP1 — infrastructure, APIs, and market data

**Purpose:** create an independent TARGET package and migrate only proven infrastructure.

**Implemented:** Python 3.11/Poetry packaging; validated side-effect-free configuration; structured
logging; Roostoo signing, typed errors, safe query retry policy, metadata/account/order queries, and
typed v6 short methods; Binance endpoint fallback and pagination; UTC normalization; completed-bar,
gap, duplicate, order, staleness, and forming-bar validation; atomic CSV persistence; and a CLI with
no exchange mutation commands.

The V2 repository is read-only reference material. TARGET imports no V2 modules at runtime;
`test_source_has_no_runtime_dependency_on_v2` protects this boundary.

### Roostoo short semantics: documented versus unknown

`API.md` is the local authority. The current adapter normalizes exchange actions into
exchange-neutral `Fill` events; the portfolio never consumes raw Roostoo JSON.

Explicitly documented/implemented facts include:

- `/v6/short_open` opens collateral-sized short exposure; its request does not use ordinary spot
  side/quantity semantics.
- Exchange `ShortQty` is a **positive magnitude**. Internal inventory represents it as a negative
  signed quantity.
- Repeated opens merge into one position with a quantity-weighted average entry.
- `/v6/short_close` is reduce-only, accepts close quantity or percentage, caps at the open amount,
  and supports partial close.
- Partial close preserves average entry and releases collateral proportionally; dust may be closed
  fully under the documented behavior.
- Short opening reserves collateral rather than crediting freely available short-sale proceeds.
- Fees, collateral release, realized PnL, and exchange settlement observations remain separately
  visible for reconciliation.

Undocumented/unresolved behavior includes:

- simultaneous independent long and short inventory in one symbol / hedge mode;
- complete wallet-equity and available-balance field semantics after shorts;
- leverage, maintenance margin, liquidation, borrow, interest, funding, or financing rules;
- complete partial-fill behavior for short limit orders;
- one-order reversal across zero;
- some close timestamps/identifiers and exact rounding/hidden-precision behavior;
- a small arithmetic discrepancy in the documentation’s sample close response, which the adapter
  surfaces rather than silently “fixing.”

Do not infer these semantics or enable live trading until they are confirmed.

### CP2 — signed long/short accounting

**Purpose:** build an exchange-neutral, Decimal accounting core before execution or strategy logic.

`domain/execution.py` defines normalized `Side`, `Fill`, and `FinancingEntry`. `BUY` increases signed
inventory and `SELL` decreases it; the previous and resulting inventory determine open, close, or
reversal. `original_action` and exchange metadata preserve Roostoo-specific meaning without leaking
it into the portfolio.

`accounting/position.py` implements:

```text
quantity > 0  => long
quantity < 0  => short
quantity = 0  => flat
```

It covers weighted same-side additions, partial/full closes, transitions through zero, gross
realized PnL, symmetric unrealized PnL, fees, financing, marks, timestamps, and restricted short
collateral. `accounting/portfolio.py` applies each normalized event once and creates immutable
snapshots. `persistence/portfolio.py` uses atomic, versioned, Decimal-as-text JSON.

The implemented exposure identities are:

$$
NMV=\sum_i q_i m_i,
\qquad
GMV=\sum_i |q_i m_i|,
$$

where $q_i$ is signed quantity and $m_i$ is mark price. Short market value is signed negative;
`short_exposure` is its absolute magnitude.

Under the segregated-collateral internal convention,

$$
Equity = available\ cash + long\ market\ value
       + restricted\ short\ collateral + short\ unrealized\ PnL.
$$

Opening a short debits cash and credits restricted collateral, so it creates no profit at an
unchanged mark. Fees and signed financing entries already affect cash; they are not subtracted from
equity a second time. Diagnostic net PnL is

$$
realized\ gross + unrealized - fees + financing.
$$

`exchange/roostoo/accounting.py` maps documented responses into fills and reconciliation
observations without allowing exchange-reported values to overwrite deterministic internal state.

### CP3 — deterministic long/short backtester

**Purpose:** establish timing and simulated-execution standards before alpha research.

The pipeline is:

```text
StrategyContext
    -> OrderIntent
    -> Order
    -> SimulatedExecutor
    -> Fill
    -> Portfolio
```

`BacktestEngine` builds synchronized completed frames, executes only previously eligible orders,
marks positions, posts due financing, exposes a bounded history, queues new orders for the following
bar, and records structured results. Every run starts with fresh portfolio, pending-order, group,
financing, and strategy state.

Market orders use the next eligible bar open. Limit orders use only future ranges and fill at the
limit—not the most favorable intrabar price—because OHLC does not establish intrabar sequencing.
Explicit fee, spread, slippage, short-open/close fee, zero/configured financing, full-fill, and
volume-capacity policies exist.

Multi-leg groups are analytical groupings only. Legs fill independently; partial groups, rejection,
capacity-limited quantities, and failed hedge legs remain visible. The simulator does not pretend
crypto execution is atomic and does not synthesize hedge recovery. The permanent 100 signal-close
versus 110 next-open test requires the fill basis to be 110.

### CP4 — synchronized research-data layer

**Purpose:** make all later estimation consume one reproducible, chronology-safe dataset.

`research_data/` implements the fixed five-asset universe, raw quality diagnostics, exact 5m-to-15m
aggregation, conservative synchronization, price/log-price/log-return views, bounded count/calendar
windows, minimum-history and gap validation, metadata, fingerprints, atomic storage, coverage
reports, bootstrap, and incremental refresh. A clean rebuild and bootstrap-plus-incremental result
are tested for equality. Its close timestamp convention is aligned directly with `BarClock`.

### CP5 — structural statistical layer

**Purpose:** estimate and expose the system’s equilibrium without creating a trading rule.

`models/` implements structured ADF/KPSS level and difference diagnostics; finite/variance checks;
VAR orders 1–8 with AIC/BIC/HQIC and BIC as default; the tested mapping VAR($p$) to $p-1$ VECM
lagged differences; Johansen trace and max-eigen diagnostics; rank scope; rank-one VECM estimation;
native and reporting-normalized $\beta$; compatible $\alpha$; every $\Gamma_j$; coefficient
uncertainty; residual covariance; ECM statistics; empirical AR(1) persistence; and deterministic
full-VECM convergence simulation.

The default model is explicit:

- model frequency: synchronized 15 minutes;
- primary window: 30 calendar days;
- comparison windows: 14, 30, and 60 days;
- fit cadence: 24 hours;
- minimum observations: 200;
- VAR search: 1 through 8, BIC primary;
- significance: 5%;
- Johansen `det_order=0`;
- VECM deterministic specification `ci` (constant inside the cointegration relation, no linear
  trend in observed levels);
- ADF/KPSS deterministic term `c`.

Latest 30-day diagnostics support—but do not prove—an I(1)-style setup: all five level ADF tests
failed to reject at 5%, all level KPSS tests rejected stationarity, and all first-difference ADF
tests strongly rejected a unit root. Difference KPSS was non-rejecting for ETH/SOL/XRP/ADA but
rejected for BTC at approximately $p=0.0228$. Contradictions remain visible rather than being
coerced into a binary truth.

The latest rank-one 30-day fit’s empirical ECM half-life was about 59.13 bars (14.78 hours); its
deterministic VECM convergence half-life was about 59.70 bars (14.92 hours). Neither is yet a holding
period or entry/exit rule.

#### Stability means three different things

Do not substitute one diagnostic for another:

1. **Cointegration validity:** rank, ECM ADF/persistence, and empirical/deterministic half-life.
2. **Equilibrium-vector stability:** sign-aligned unit-$\beta$ cosine similarity across fits,
   $\cos(\beta_t,\beta_{t-1})$, plus rank and lag stability.
3. **Correction-role stability:** persistence of $\kappa_i=\beta_i\alpha_i$, correction shares,
   dominant corrector, and secondary corrector.

Beta cosine is a direction-rotation metric, not a stationarity test. ECM stationarity, beta
stability, and role stability are not equivalent. A Menchero/Orr/Wang-style temporal factor-
stability concept has been discussed, but no such named metric is implemented in this repository.

Fits are immutable and get deterministic IDs/fingerprints. `StructuralFitScheduler` never
retrospectively mutates past fits. `StructuralModelStore` persists inspectable versioned JSON, not an
unsafe pickle-only artifact.

### CP6 — ECM-driven Kalman state space

**Purpose:** improve sequential state estimation while retaining the economics of the immutable
VECM. A pure random-walk state transition has no restoring force; this layer explicitly includes
$\alpha\beta'x_{t-1}$.

Between structural refits, $\beta$, $\alpha$, every $\Gamma_j$, deterministic terms, and
$\Sigma_\epsilon$ remain fixed. The filter estimates latent state only. It does not re-estimate a
time-varying $\beta$ or redefine the equilibrium to make the spread appear healthy.

For VAR order $p$ and five assets, the augmented state has dimension $5p$:

$$
s_t=[x_t,\ \Delta x_t,\ \Delta x_{t-1},\ldots,\Delta x_{t-p+2}]'.
$$

`models/state_space/representation.py` constructs

$$
s_t=Fs_{t-1}+b+G\epsilon_t,
\qquad Q=G\Sigma_\epsilon G',
$$

where the top state block uses $I+\alpha\beta'$, the lag blocks contain $\Gamma_j$ and deterministic
shift identities, and the same innovation enters both current price and newest difference blocks.
The measurement model is

$$
y_t=Hs_t+v_t,
\qquad H=[I_5,0,\ldots,0],
\qquad v_t\sim N(0,R).
$$

$R$ is explicit. The default is diagonal variance equal to 1% of fitted residual variance; explicit
diagonal or valid full covariance is also supported. This parameter has not been empirically
identified or tuned for PnL.

The filter predicts from the posterior through $k-1$, then consumes completed $y_k$, then publishes
the posterior at $k$. It reports predicted/filtered state and covariance, innovation, gain,
observed/filtered ECM under the same native $\beta$, separate $\alpha ECM$ and $\Gamma$ components,
one- and multi-step means/covariances, latent versus observed-relative changes, and the expected ECM
path. Structural refits rebuild lag state; compatible current-price posterior information may be
carried explicitly. Rank transitions deactivate/reactivate rather than retaining stale dynamics.

Random-walk KF and plain-VECM baselines exist for non-trading comparison. Filter and regime state
have atomic, versioned JSON persistence and deterministic restart tests.

#### CP6 empirical limitation

A fresh chronological replay of the eight recent daily fits (six active rank-one fits), with one
forecast origin per hour, produced the following aggregate absolute log-price RMSE:

| Horizon | Random-walk KF | Plain VECM | ECM-driven KF |
|---:|---:|---:|---:|
| 1 hour | 0.006218 | 0.006270 | 0.006275 |
| 4 hours | 0.011622 | 0.011931 | 0.011937 |
| 8 hours | 0.013427 | 0.014517 | 0.014518 |
| 16 hours | 0.017341 | 0.019326 | 0.019329 |
| 24 hours | 0.021810 | 0.023058 | 0.023064 |

The ECM-driven KF did not outperform the random-walk KF on this metric and was nearly identical to
the plain VECM because the default measurement noise is small. The study covered only 480 active
filter timestamps and 190 inactive timestamps; it is descriptive, not a tuned evaluation.

This result does not directly falsify the narrower trading hypothesis. That hypothesis concerns
relative equilibrium correction conditional on rank-one disequilibrium, not superior prediction of
each asset’s absolute USDT price. It nevertheless rules out claiming CP6 already improves aggregate
absolute-price forecasts.

### CP7 — structural role identification

**Purpose:** identify structural correctors and conservative leader/anchor candidates without
orders, sizing, thresholds, or PnL.

For restoring assets, the implementation defines

$$
restoring\_strength_i=\max(-\kappa_i,0),
$$

$$
correction\_share_i=
\frac{\max(-\kappa_i,0)}{\sum_j\max(-\kappa_j,0)}.
$$

Correction share is a **role diagnostic**, not a trade weight, hedge ratio, or capital allocation.
At the default 5% level, a credible corrector requires finite uncertainty, significant $\alpha_i$,
and $\kappa_i<0$. The dominant corrector is the credible restorer with the largest share. A secondary
corrector is admitted only if it is credible, has at least 50% of the dominant share, and its current
$\alpha_i ECM_t$ direction supports the same restoration story. V1 admits at most two correctors.

Weak-adjustment diagnostics test $H_0:\alpha_i=0$ without claiming causal or fundamental
exogeneity. Gamma leadership uses joint Wald restrictions over all fitted lag coefficients for each
ordered source-to-target pair, using the retained joint covariance block. Outgoing leadership and
incoming dependence remain separate and are called conditional predictive evidence, not causality.

The implemented anchor must not be a selected corrector, must have correction share at most 20%,
must show weak adjustment, and must significantly predict the dominant corrector through Gamma.
Deterministic tie-breaking uses p-value, correction share, then canonical universe order. No result
may use future returns or PnL.

`PAIR` means one corrector versus one anchor. `MINIMAL_CORRECTING_BASKET` means two correctors versus
one anchor. Both are membership only: there are no notionals. The separately reported
`FULL_COINTEGRATING_BASKET` retains native $\beta$ as a future benchmark and is never mixed into the
minimal role rule.

## Empirical CP7 Findings

These values were freshly recomputed during this handoff from the local fingerprinted panel and the
eight daily 30-day fits ending 2026-09-06 through 2026-09-13. Six were rank one.

### Correction roles

| Asset | Mean correction share across six rank-one fits |
|---|---:|
| BTC | 0.00% |
| ETH | 0.00% |
| SOL | 31.64% |
| XRP | 67.65% |
| ADA | 0.72% |

- XRP was the dominant corrector in 6/6 fits; adjacent-fit identity persistence was 100%.
- The median dominant share was 67.57%; all six exceeded the configured 40% asymmetry diagnostic.
- SOL met the secondary rule in 3/6 fits.
- BTC/ETH shares are zero because their $\kappa$ values were non-restoring in these fits, not
  because their $\alpha$ estimates were zero.
- All five $\alpha$ coefficients were significant in each fit; weak-adjustment candidate lists were
  empty.

The correct conclusion is **preliminary support for asymmetric error correction in this small
recent sample**. It is not proof of an edge, a position-sizing prescription, or evidence that XRP
will remain dominant in other periods.

### Gamma lead-lag evidence

The following ordered relationships were significant in all six rank-one fits:

```text
BTC -> ETH, XRP, ADA
ETH -> BTC, SOL, ADA
SOL -> ETH, XRP, ADA
XRP -> SOL
ADA -> BTC
```

`BTC -> SOL` and `ADA -> ETH` were significant in 3/6 fits. These are joint, conditional Gamma
tests under the active VECM. They are not causal price-discovery proof and currently have no
multiple-testing correction.

### Anchor and expression result

- weakly adjusting leader/anchors: 0/6;
- `PAIR_CANDIDATE`: 0/6;
- `MINIMAL_BASKET_CANDIDATE`: 0/6;
- `NO_CREDIBLE_ANCHOR`: 6/6;
- no valid expression: 6/6.

Therefore the weak-adjusting anchor premise was **not supported** in this sample. No profitability
test exists and no final trade membership has been frozen.

## Current Research Question

The project is paused before the first strategy backtest because the counter-leg definition is not
settled. If all assets adjust significantly, requiring $\alpha\approx0$ may be too restrictive. The
question is:

> What should form the opposite leg of a minimal convergence trade when there is a dominant
> corrector but no weakly adjusting anchor?

The opposite-$\kappa$-pole rule described earlier is the current proposal. It may choose a genuinely
non-restoring asset ($\kappa_A>0$) or merely the weakest restorer if every value is negative. Gamma
leadership may become supporting rather than gating evidence. None of these decisions is yet an
implemented/frozen rule.

The minimal expression is not currently required to be perfectly dollar neutral, beta neutral,
Johansen neutral, volatility neutral, or factor neutral. The proposed experiment asks whether two
ends of the equilibrium-response structure exhibit predictable relative convergence. Residual
exposures should eventually be measured and reported, not silently optimized away. Do not add
mean-variance optimization, sparse optimization, or beta-neutral hedge ratios under the guise of
cleanup.

## Full Johansen Basket Benchmark

The conventional benchmark is the full cointegrating vector traded opposite the ECM displacement,
conceptually

$$
w_\beta \propto -\operatorname{sign}(ECM_t)\beta,
$$

with a normalization that must be specified before backtesting. It is a future benchmark, not the
production strategy. The intended empirical comparison is:

```text
minimal role-derived expression  versus  full cointegrating basket
```

The comparison must use the same rank gate, information clock, next-bar fills, cost model, and
walk-forward structural fits.

## Permanent Correctness and Research Checks

The suite tests invariants, not just happy paths.

### Data correctness

- exact duplicate idempotence and conflicting-overlap rejection;
- gap, order, duplicate, staleness, interval, OHLCV, and forming-bar diagnostics;
- exact UTC 5m-to-15m alignment and no partial aggregation;
- all-five synchronization, deterministic order, and no forward-fill;
- bounded count/calendar windows and explicit insufficient-history/gap states;
- full rebuild versus bootstrap-plus-incremental equivalence;
- source/panel fingerprints and research/backtest clock alignment.

### Accounting

- every flat/long/short addition, partial close, full close, and reversal transition;
- weighted entry and remaining-entry preservation;
- long/short PnL symmetry and zero PnL on unchanged opening marks;
- collateral reservation/release, cash/equity identity, fees and financing exactly once;
- multi-asset GMV/NMV/equity reconciliation;
- immutable snapshots, Decimal serialization, and restart round trips;
- typed Roostoo short-open/close normalization and discrepancy surfacing.

### Backtesting

- signal bar cannot fill itself; close 100 followed by next open 110 fills from 110;
- bounded strategy history and future-data contamination invariance;
- forming-bar exclusion and multi-symbol synchronization;
- limit orders cannot use signal-bar high/low;
- model metadata chronology and every fill strictly after its signal;
- costs remain separately observable;
- grouped full, partial, failed, capacity-limited, and failed-close cases;
- deterministic reset/replay with no state leakage.

### Structural models

- fixed-seed no-cointegration and known-cointegration synthetic systems;
- rank-one subspace and known-adjuster recovery;
- rank > 1 retained without arbitrary reduction;
- VAR/VECM lag mapping and all criterion rows;
- structural-break beta instability;
- native/reporting beta sign/scale invariance and $\alpha\beta'$ equivalence;
- ECM persistence without clipping invalid half-life;
- residual-covariance shape, symmetry, and positive diagonal;
- bounded-fit and fingerprint contamination tests;
- safe, inspectable structural persistence.

### State space

- exact state-space/VECM recursion for $p=1$, $p=2$, and $p>2$;
- explicit $G\Sigma_\epsilon G'$ cross-covariance mapping;
- valid explicit measurement-noise modes and loud covariance failures;
- predict-before-update chronology and future-observation invariance;
- deterministic VECM convergence-path equivalence;
- synthetic restoring-force/state recovery and dominant-adjustment preservation;
- structural-break innovations worsen without changing $\beta$;
- rank/refit handoffs, incompatible-state rejection, and restart equivalence;
- chronological random-walk/VECM/ECM-KF diagnostics with no orders or PnL.

### Roles

- rank gate and no stale roles;
- $\kappa$, correction-share, alpha-credibility, and dominant-corrector recovery;
- explicit no-corrector and no-anchor systems;
- joint multi-lag Gamma Wald tests and covariance consistency;
- secondary-corrector threshold and three-leg membership cap;
- compatible beta/alpha scale/sign invariance;
- fit/filter ID compatibility and future-contamination invariance;
- history studies containing no PnL or orders.

## Verification Status

Freshly run during this handoff on the current working tree:

- research coverage: 35,040 synchronized rows, zero drops/gaps;
- latest cross-window fits and eight-fit CP6/CP7 diagnostic studies;
- `poetry run pytest`: **191 passed**;
- `poetry run ruff check .`: passed;
- `poetry run ruff format --check .`: passed;
- `poetry check`: passed;
- `poetry run pip check`: passed;
- `poetry build -f wheel`: passed;
- `poetry build -f sdist`: passed;
- `git diff --check`: passed.

No signed live exchange call was made.

### Repository-state warning

At inspection start, branch `main` was at `d5a5627` (`checkpoint 7`) and matched `origin/main`, but
the working tree was already not clean. It contained pre-existing CP7 modifications to `README.md`,
`models/__init__.py`, `models/vecm.py`, and structural-model tests, plus untracked role docs/source/
tests. `BRIEF.md` is the only file added by this handoff. A fresh clone of the current commit should
not be assumed to contain the exact CP7 working tree described here until those pre-existing changes
are reconciled and committed.

## Known Limitations and Unproven Claims

- Johansen rank is sensitive to window length, lag order, deterministic specification, significance
  level, and structural breaks.
- The recent role study has only six rank-one fits over a short, overlapping period.
- Finding in-sample cointegration is not evidence of a profitable strategy.
- $\beta$ describes an equilibrium; it is not automatically a forecast-optimal or tradable weight
  vector.
- $\alpha$ inference and Gamma Wald tests are asymptotic; no multiple-testing correction is applied.
- Gamma evidence is predictive/conditional, not causal leadership.
- The conservative weak-anchor rule failed in the current sample; its replacement is unresolved.
- Measurement covariance $R$ is explicit but not empirically identified. The default small $R$
  makes filtered observations track market closes closely.
- Structural parameter uncertainty is not propagated through state forecasts.
- The Kalman model is linear/Gaussian and has no generic missing-measurement update path.
- CP6 did not improve aggregate absolute-price RMSE over the random-walk baseline in the recent
  study.
- Financing/funding and several live short/margin behaviors remain undocumented.
- OHLC limit simulation cannot know intrabar order and uses a conservative documented convention.
- Simulated liquidity is full-fill or simple volume capacity; it is not a market-impact model.
- There is no final counter-leg, entry/exit threshold, holding rule, sizing, leverage/risk framework,
  live execution coordinator, or strategy profitability result.
- There is no causal price-leadership conclusion.

## Repository Map

| Path | Responsibility | Important abstractions |
|---|---|---|
| `src/stat_arb_bot/config/` | Explicit validated environment/runtime configuration | `Settings`, `load_settings` |
| `src/stat_arb_bot/domain/` | Exchange-neutral market and execution events | `Candle`, `Side`, `Fill`, `FinancingEntry` |
| `src/stat_arb_bot/exchange/roostoo/` | Signing, HTTP client, typed Roostoo responses, accounting normalization | `RoostooClient`, short result models, mapping/reconciliation functions |
| `src/stat_arb_bot/market_data/` | Binance retrieval, symbols, candle validation, atomic operational CSV | `BinanceData`, `CandleStore`, quality reports |
| `src/stat_arb_bot/research_data/` | Canonical history, aggregation, synchronization, windows, provenance | `ResearchUniverse`, `ResearchDataBuilder`, `ResearchPanel`, `ResearchWindow`, `ModelFitMetadata` |
| `src/stat_arb_bot/accounting/` | Signed position and portfolio economics | `Position`, `Portfolio`, immutable snapshots, reconciliation |
| `src/stat_arb_bot/persistence/` | Portfolio state durability | `PortfolioStateStore` |
| `src/stat_arb_bot/backtest/` | Information clock, bounded history, intents/orders, costs, fills, groups, replay | `BarClock`, `HistoricalPanel`, `OrderIntent`, `SimulatedExecutor`, `BacktestEngine`, `BacktestResult` |
| `src/stat_arb_bot/models/` | Integration diagnostics, lag selection, Johansen/VECM, stability, fit scheduling/persistence | `StructuralModelConfig`, `StructuralEstimator`, `StructuralFitResult`, `RankOneVECMResult` |
| `src/stat_arb_bot/models/state_space/` | Exact VECM state representation, Kalman filter, regime handoff, baselines, studies, persistence | `VECMStateSpace`, `ECMDrivenKalmanFilter`, `StructuralRegimeManager` |
| `src/stat_arb_bot/models/roles/` | Non-trading $\kappa$/corrector/anchor/Gamma diagnostics | `StructuralRoleProfile`, `RoleClassificationResult`, `RoleHistoryStudy` |
| `src/stat_arb_bot/cli.py` | Read-only exchange inspection and market/research-data operations | `stat-arb-bot` command parser |
| `docs/` | Detailed contracts for accounting, backtesting, data, modelling, state space, and roles | Architecture/reference documents |
| `tests/` | Permanent correctness and research-invariant suite | Synthetic systems, chronology, persistence, empirical plumbing |
| `API.md` | Local authoritative Roostoo short specification | v6 short open/close/position behavior |

## Suggested Onboarding Reading Order

1. `BRIEF.md` — research question, evidence, invariants, and current decision point.
2. `README.md` — setup and supported operational surface.
3. `docs/research-data.md` — timestamps, aggregation, synchronization, and bounded windows.
4. `docs/modelling.md` — structural specification and econometric interpretations.
5. `docs/state-space.md` — fixed-VECM state derivation and sequential chronology.
6. `docs/role-selection.md` — the implemented conservative CP7 rule.
7. `docs/backtesting.md` — information clock, costs, and non-atomic execution.
8. `docs/accounting.md` — signed accounting and short-collateral identity.
9. `API.md` — exactly what is and is not documented about Roostoo shorts.
10. Core source, in this order:
    - `research_data/panel.py`, `windows.py`, and `pipeline.py`;
    - `models/config.py`, `structural.py`, `vecm.py`, and `scheduler.py`;
    - `models/state_space/representation.py`, `filter.py`, and `regime.py`;
    - `models/roles/structural.py`, `gamma.py`, `classification.py`, and `history.py`;
    - `backtest/history.py`, `engine.py`, and `execution.py`;
    - `accounting/position.py` and `portfolio.py`;
    - `exchange/roostoo/accounting.py` and `client.py`.
11. Tests matching each layer, especially chronology, structural synthetic systems, state-space
    equivalence, and role contamination tests.

## Non-Negotiable Engineering Invariants

1. V2 remains read-only reference material.
2. TARGET must have no runtime dependency on V2.
3. Do not guess beyond `API.md`; unresolved exchange behavior stays unresolved.
4. Exchange quantities, prices, cash, fees, collateral, and accounting use explicit `Decimal`
   boundaries. Model/research arrays may use NumPy floating point.
5. Never construct `Decimal` directly from a binary float; conversion goes through deterministic
   text (`Decimal(str(value))`).
6. Information through close $k$ cannot use future observations.
7. A signal created at $k$ cannot execute on $k$.
8. Structural fits are immutable and retain their data-end timestamp and source fingerprint.
9. Rank other than one disables the V1 ECM-driven model and role regime.
10. $\beta$ is not automatically a trade-weight vector.
11. $\beta$, $\alpha$, and $\Gamma$ have distinct meanings and must remain separately inspectable.
12. Compatible beta normalization requires inverse alpha scaling so $\alpha\beta'$ and $\kappa$
    remain economically unchanged.
13. Correction shares are role diagnostics, not notionals.
14. Do not optimize trade construction before the counter-leg hypothesis is frozen and tested.
15. Future data and future returns must never influence role labels, tie-breaking, or model versions.
16. Multi-leg execution is not atomic; actual partial/failed exposure remains real.
17. Adverse and null empirical findings are evidence to preserve, not parameters to tune away.

## “Cleanup” That Can Change the Experiment

Do not casually:

- give strategies or estimators the full future-containing dataframe instead of a bounded view;
- allow close-$k$ information to fill at any bar-$k$ price;
- forward-fill a missing asset into a synchronized model row;
- recompute historical fits with newer data and overwrite their version;
- normalize $\beta$ without inversely rescaling $\alpha$;
- compare raw $|\alpha|$ across incompatible beta normalizations;
- collapse $\alpha ECM$ and Gamma dynamics into one opaque score;
- use correction shares as position weights;
- replace a missing anchor with BTC because it seems economically important;
- implement the opposite-$\kappa$ proposal without a frozen specification;
- use `Decimal(float_value)` in accounting;
- treat a multi-leg group as an atomic exchange fill;
- change collateral/margin economics based on undocumented assumptions;
- choose windows, thresholds, $R$, notionals, or exits by eventual backtest Sharpe without declaring
  and controlling a new research stage.

## Development and Verification Commands

The project targets Python 3.11.15 and Poetry 2.x.

```bash
poetry env use 3.11.15
poetry install --with dev
cp .env.example .env
```

Never commit `.env`. Signed read-only Roostoo queries require credentials; the verification suite
does not.

Run the local checks:

```bash
poetry run pytest
poetry run ruff check .
poetry run ruff format --check .
poetry check
poetry run pip check
poetry build -f wheel
poetry build -f sdist
git diff --check
```

Inspect current local research coverage without network access:

```bash
poetry run stat-arb-bot research-coverage --as-of 2026-09-13T00:00:00Z
```

Bootstrap or incrementally refresh the canonical raw store, then rebuild derived data:

```bash
poetry run stat-arb-bot research-fetch \
  --start 2025-09-13T00:00:00Z --end 2026-09-13T00:00:00Z

poetry run stat-arb-bot research-fetch \
  --start 2025-09-13T00:00:00Z --end 2026-09-14T00:00:00Z --incremental

poetry run stat-arb-bot research-build \
  --start 2025-09-13T00:00:00Z --as-of 2026-09-13T00:00:00Z
```

`research-fetch` performs public network reads and writes local canonical data. The CLI also exposes
`server-time`, `exchange-info`, `ticker`, `balance`, `pending-count`, `orders`, `short-positions`,
`collect`, `validate-candles`, and `smoke`; it intentionally exposes no place/open/close/cancel
command. Use `poetry run stat-arb-bot --help` for the actual argument surface.

There is currently no model/role-study CLI. CP5–CP7 diagnostics are library APIs exercised by tests;
do not invent a command in handoff instructions.

## Where We Are Now

- Independent infrastructure, market data, signed accounting, and deterministic simulation are
  mature.
- One year of complete local five-asset research data is available in this workspace.
- Bounded structural estimation and immutable daily fits exist.
- The rank-gated fixed-VECM state-space filter exists and is restartable.
- Scale-invariant asymmetric correction diagnostics and conservative role classification exist.
- Recent evidence says XRP and SOL account for almost all restoring share, with XRP dominant in all
  six recent rank-one fits.
- The conservative weak-adjustment-plus-Gamma anchor rule found no valid anchor.
- No final trade membership has been defined.
- No entry/exit or sizing rule exists.
- No strategy PnL test has been run, and no profitability or causal-leadership claim is justified.

## What Comes Next

Before implementing the first strategy backtest, freeze the minimal counter-leg rule in a new,
explicit research specification. The currently discussed opposite-$\kappa$-pole direction is a
proposal, not an instruction hidden in this brief. The specification should define credibility,
ties, behavior when all assets restore, treatment of Gamma evidence, and when to return “no trade.”

Only after membership is frozen should the first non-optimized walk-forward test be built. It should
use:

- rank-one regime gating;
- a pre-specified ECM displacement condition;
- immutable walk-forward structural fits and CP6 state chronology;
- the frozen minimal membership rule;
- simple transparent notionals, with residual exposures reported rather than optimized away;
- signal at close $k$ and execution no earlier than open $k+1$;
- the existing fees, spread, slippage, financing, collateral, and partial/failure semantics;
- exits on pre-specified convergence, rank/regime failure, and time stop;
- the full $\beta$ basket under identical conditions as the benchmark;
- no parameter search against the evaluation result.

The next stage must answer—not assume—the project’s core question: does a role-derived minimal
expression monetize rank-one equilibrium correction more effectively than the full Johansen basket?
