"""Pre-specified, non-trading configuration for structural role diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RoleSelectionConfig:
    """Transparent V1 role rules; none of these values are entry or sizing rules."""

    alpha_significance_level: float = 0.05
    gamma_significance_level: float = 0.05
    secondary_share_ratio: float = 0.50
    maximum_anchor_correction_share: float = 0.20
    asymmetry_dominant_share_threshold: float = 0.40
    numerical_zero_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("alpha_significance_level", "gamma_significance_level"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be strictly between zero and one")
        if not 0 <= self.secondary_share_ratio <= 1:
            raise ValueError("secondary_share_ratio must be between zero and one")
        if not 0 <= self.maximum_anchor_correction_share <= 1:
            raise ValueError("maximum_anchor_correction_share must be between zero and one")
        if not 0 <= self.asymmetry_dominant_share_threshold <= 1:
            raise ValueError("asymmetry_dominant_share_threshold must be between zero and one")
        if self.numerical_zero_tolerance <= 0:
            raise ValueError("numerical_zero_tolerance must be positive")
