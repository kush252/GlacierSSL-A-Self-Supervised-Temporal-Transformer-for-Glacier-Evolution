from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class EpochData:
    """Represents measurements for a single glacier at a single epoch."""
    date: datetime
    area_km2: float
    perimeter_km: float
    debris_fraction: float
    terminus_displacement_m: float  # Cumulative displacement from first epoch (positive = retreat up-valley)
    fragment_count: int
    min_bridge_width_m: float
    is_surge_active: bool = False
    is_snow_patch_likely: bool = False

@dataclass
class GlacierState:
    """Represents the entire observation sequence and computed metrics for a glacier."""
    glacier_id: str
    glims_id: str
    region: str
    
    epochs: List[EpochData] = field(default_factory=list)
    
    # Precomputed temporal features
    time_span_years: float = 0.0
    beta_hat: float = 0.0          # OLS slope of area vs year
    p_value: float = 1.0           # OLS p-value
    max_gap_years: float = 0.0
    max_area_km2: float = 0.0
    
    # Final outputs
    assigned_class: Optional[str] = None
    confidence: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)

    @property
    def num_epochs(self) -> int:
        return len(self.epochs)

    def get_areas(self) -> List[float]:
        return [e.area_km2 for e in self.epochs]
    
    def get_dates(self) -> List[datetime]:
        return [e.date for e in self.epochs]
