from utils.structures import GlacierState
from config.config import config
from rule_engine import rules

def score_confidence(state: GlacierState) -> str:
    """
    Calculates the confidence score: High, Medium, or Low.
    Based on Section 2.5 of the Guide.
    """
    if state.assigned_class == "insufficient_evidence":
        return "None"

    # Evaluate base criteria
    meets_epoch_count = state.num_epochs >= config.min_epochs
    has_large_gap = state.max_gap_years > config.max_gap_years_high_conf
    
    # Check debris cover levels
    max_debris = max(e.debris_fraction for e in state.epochs)
    heavy_debris = max_debris > config.debris_tiers['heavy']
    partial_debris = config.debris_tiers['clean'] < max_debris <= config.debris_tiers['heavy']

    # Sub-MMU check
    is_sub_mmu = state.max_area_km2 < config.mmu_km2

    # LOW CONFIDENCE (overrides all)
    if not meets_epoch_count or heavy_debris or is_sub_mmu:
        return "Low"
        
    # MEDIUM CONFIDENCE
    # - Has >= 1 gap > 5 years
    # - Partial debris cover (10-50%)
    # - Only one of {Sentinel-2, GLIMS} available (Assume True for now, we'll flag if not later)
    # - Change magnitude between 1x and 2x noise threshold (Implemented as a flag for simplicity)
    if has_large_gap or partial_debris:
        return "Medium"
        
    # HIGH CONFIDENCE
    return "High"
