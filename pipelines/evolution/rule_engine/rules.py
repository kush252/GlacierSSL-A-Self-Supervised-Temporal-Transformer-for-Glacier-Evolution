from config.config import config
from utils.structures import GlacierState
import math

def rule_2_3_A(state: GlacierState) -> bool:
    """Minimum temporal support: >= 4 independent epochs spanning >= 8 years."""
    if state.num_epochs < config.min_epochs:
        return False
    if state.time_span_years < config.min_span_years:
        return False
    return True

def rule_2_4_B(state: GlacierState) -> bool:
    """Minimum mapping unit (MMU): Max observed area across sequence is < 0.02 km2."""
    if state.max_area_km2 < config.mmu_km2:
        return True
    return False

def rule_2_4_C_exceeded(state: GlacierState, direction: str) -> bool:
    """
    Checks if the terminus displacement exceeds the threshold in the specified direction.
    direction: 'up-valley' (retreat) or 'down-valley' (advance).
    Retreat is defined as positive cumulative displacement in our tracker.
    """
    if state.num_epochs < 2:
        return False
        
    term_diff = state.epochs[-1].terminus_displacement_m - state.epochs[0].terminus_displacement_m
    
    # Check debris cover for threshold
    heavy_debris = any(e.debris_fraction > config.debris_tiers['partial'] for e in state.epochs)
    threshold = config.min_terminus_disp_debris_m if heavy_debris else config.min_terminus_disp_clean_m
    
    if direction == 'up-valley':
        return term_diff >= threshold
    elif direction == 'down-valley':
        return term_diff <= -threshold
    return False

def rule_5_5_B_fragmented(state: GlacierState) -> bool:
    """
    Treat as fragmented if the max path width is 0 across >= 2 independent, same-season epochs.
    Here we check if we have >= persistence_epochs where min_bridge_width_m == 0.
    """
    fragmented_epochs = sum(1 for e in state.epochs if e.min_bridge_width_m < config.min_bridge_width_m)
    return fragmented_epochs >= config.persistence_epochs

def rule_7_4_A(state: GlacierState) -> bool:
    """
    Disappearance: Most recent >= 2 same-season epochs show A_t < 0.02 km2 or A_t = 0,
    AND immediately preceding trend was significantly negative.
    """
    if state.num_epochs < 2:
        return False
        
    recent_epochs = state.epochs[-config.persistence_epochs:]
    all_below_mmu = all(e.area_km2 < config.mmu_km2 for e in recent_epochs)
    
    if not all_below_mmu:
        return False
        
    # Check preceding trend (Rule 3.4-A/4.4-A logic)
    if state.p_value < config.p_value_threshold and state.beta_hat < 0:
        return True
    return False

def rule_3_4_A(state: GlacierState) -> bool:
    """
    Stable:
    1. |beta_hat| * (year_N - year_1) < dA_noise * 2
    2. trend slope not statistically significant (p >= 0.05)
    3. terminus displacement below Rule 2.4-C threshold
    4. no fragmentation event recorded
    """
    if not rule_2_3_A(state):
        return False
        
    # We approximate dA_noise using the average perimeter
    avg_perimeter = sum(e.perimeter_km for e in state.epochs) / state.num_epochs
    # Assuming clean ice (10m) for approximation, config defines pixel_width_clean_ice_m in meters.
    # Convert to km
    w_km = config.pixel_width_clean_ice_m / 1000.0
    dA_noise = avg_perimeter * w_km
    
    cond1 = (abs(state.beta_hat) * state.time_span_years) < (dA_noise * config.safety_factor)
    cond2 = state.p_value >= config.p_value_threshold
    cond3 = not (rule_2_4_C_exceeded(state, 'up-valley') or rule_2_4_C_exceeded(state, 'down-valley'))
    cond4 = not rule_5_5_B_fragmented(state)
    
    return cond1 and cond2 and cond3 and cond4

def rule_4_4_A(state: GlacierState) -> bool:
    """
    Retreat:
    (p < 0.05 with beta_hat < 0) OR (terminus displacement exceeds Rule 2.4-C up-valley)
    AND no fragmentation applies (checked in hierarchy).
    """
    if not rule_2_3_A(state):
        return False
        
    cond1 = (state.p_value < config.p_value_threshold) and (state.beta_hat < 0)
    cond2 = rule_2_4_C_exceeded(state, 'up-valley')
    
    return cond1 or cond2

def rule_6_5_A(state: GlacierState) -> bool:
    """
    Surge Exclusion: positive trend concentrated in <= 2 consecutive epochs and visible surface disturbance.
    """
    # For now, rely on visual heuristic flags stored in epoch data.
    surge_epochs = sum(1 for e in state.epochs if e.is_surge_active)
    return surge_epochs > 0

def rule_6_4_A(state: GlacierState) -> bool:
    """
    Advance:
    (p < 0.05 with beta_hat > 0) OR (terminus displacement exceeds Rule 2.4-C down-valley)
    AND surge exclusion checked and does not apply.
    """
    if not rule_2_3_A(state):
        return False
        
    cond1 = (state.p_value < config.p_value_threshold) and (state.beta_hat > 0)
    cond2 = rule_2_4_C_exceeded(state, 'down-valley')
    cond3 = not rule_6_5_A(state)
    
    return (cond1 or cond2) and cond3
