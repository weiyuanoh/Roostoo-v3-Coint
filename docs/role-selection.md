# Corrector and leader/anchor diagnostics

Checkpoint 7 is a non-trading research layer. It identifies structural roles and hypothetical
trade membership, but it has no entry or exit threshold, quantity, notional, leverage, risk
limit, order, execution, or PnL logic.

## Frozen hypothesis and rank gate

The five-asset system defines the equilibrium. In a rank-one regime, restoration may be
asymmetric: one or two assets can perform most long-run correction while a weakly adjusting
asset may lead them over shorter horizons. The intended later expression is the dominant
corrector (optionally one transparent secondary corrector) against one leader/anchor—not the
full beta portfolio by default.

Classification is active only for `rank == 1`. Rank zero returns `INACTIVE_RANK_ZERO`; higher
rank returns `INACTIVE_MULTIPLE_RANK`. No role from an earlier rank-one fit is carried across a
rank transition. Every new immutable fit receives a new fit-level classification.

## Three deliberately separate quantities

`beta` defines the equilibrium. `alpha` describes how each asset responds to disequilibrium.
`Gamma` describes lagged short-run changes. These produce three distinct diagnostics:

1. Structural equilibrium contribution: `kappa_i = beta_i * alpha_i`.
2. Current corrective price move: `alpha_i * ECM_t`, evaluated with both observed and filtered
   ECM.
3. Short-run predictive leadership: joint restrictions on the relevant Gamma coefficients.

They are never combined into an opaque score.

## Corrector rule

The contribution of asset `i` to the ECM change through error correction is
`kappa_i * ECM_t`. Because compatible rescaling uses `beta -> q beta` and
`alpha -> alpha / q`, `kappa_i` is invariant to beta scale and sign. A negative kappa is
restoring; a positive kappa is individually destabilizing even when the total
`beta' alpha` remains restoring.

For restoring assets:

```text
restoring_strength_i = max(-kappa_i, 0)
correction_share_i = restoring_strength_i / sum(restoring_strength)
```

Correction shares are role diagnostics, not notionals, hedge ratios, or trade weights. A
credible corrector must have negative kappa and an alpha coefficient significant under the
configured two-sided asymptotic test (5% by default). Restoring assets with imprecise alpha are
reported as `RESTORING_BUT_UNCERTAIN`; they are not silently discarded. If none is credible,
the result is `NO_CREDIBLE_CORRECTOR`.

The dominant corrector is the credible restoring asset with the largest share, with canonical
universe order as a deterministic exact-tie fallback. It is fixed for the life of that
structural fit. A second corrector is admitted only when it is credible, has at least 50% of
the dominant share by default, and its `alpha * ECM` price direction is the same as the
dominant corrector for every non-zero ECM. V1 never selects more than two correctors.

## Weak adjustment and Gamma leadership

An alpha coefficient that is not statistically distinguishable from zero supplies
weak-adjustment (or weak-exogeneity-within-this-VECM) evidence. It does not prove fundamental
or causal exogeneity.

For each ordered source→target pair, all coefficients
`Gamma_lag[target, source]` are tested jointly against zero with a Wald chi-square test using
the fitted joint coefficient covariance. The complete coefficient vector, marginal standard
errors, signs, test statistic, degrees of freedom, p-value, and validity status remain
inspectable. Incoming dependence and outgoing leadership are retained as separate
relationship collections. This is short-run predictive lead/lag evidence—not causal price
discovery.

For dominant corrector `C`, an anchor candidate `A` must:

* not be a selected corrector;
* have correction share at or below 20% by default;
* show weak-adjustment evidence; and
* have significant Gamma evidence `A -> C`.

If several qualify, selection uses lower Gamma p-value, then lower correction share, then
canonical universe order. No future return or PnL enters the tie-break. Only an asset satisfying
both weak long-run adjustment and short-run leadership is called a leader/anchor. Otherwise the
result is `NO_CREDIBLE_ANCHOR`.

## Current Kalman consistency

Structural roles are chosen before the Kalman forecast is consulted. At the active empirical
half-life horizon (falling back to the deterministic VECM half-life when unavailable), the
result reports observed and filtered ECM, per-asset `alpha * ECM`, latent and
market-relative forecast changes, and the dominant-corrector-minus-anchor latent forecast.

The current result reports `NO_EXPECTED_CONVERGENCE` when the expected absolute ECM does not
shrink. It reports `CORRECTOR_DIRECTION_CONFLICT` when the dominant corrector's filtered
`alpha * ECM` direction materially disagrees with its latent-state forecast direction. These
states do not erase the structural role diagnostics.

## Membership-only expressions

`PAIR` contains one dominant corrector and one leader/anchor.
`MINIMAL_CORRECTING_BASKET` contains a dominant corrector, one admitted secondary corrector,
and one leader/anchor. The hypothetical corrector direction is the sign of filtered
`alpha * ECM`; the anchor is the opposite side. These are memberships and directions only.

`FULL_COINTEGRATING_BASKET` separately records the rank-one beta benchmark and its convergence
directions. It is retained for a later controlled comparison. **Beta is not the default trading
portfolio.**

## Chronology and study outputs

A role result requires `structural_fit.data_end <= classification_timestamp`. A supplied Kalman
state must carry the same immutable fit ID. The history study initializes each fit only from
panel observations at or before its activation timestamp. It records shares, roles, Gamma
relationships, persistence, and candidate frequencies; it neither receives future returns nor
calculates PnL.

The asymmetry summary uses a pre-specified, non-trading diagnostic: median dominant correction
share at least 40%, compared with the equal-share reference of 20%. The raw shares and frequency
remain primary; this label is not an entry gate and is not calibrated against profitability.
