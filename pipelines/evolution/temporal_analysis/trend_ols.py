import numpy as np
from scipy import stats
from utils.structures import GlacierState

def compute_temporal_features(state: GlacierState):
    """
    Computes OLS trend (beta_hat), p-value, time span, and max epoch gap.
    Mutates the GlacierState object with these precomputed values.
    """
    if state.num_epochs < 2:
        state.beta_hat = 0.0
        state.p_value = 1.0
        state.time_span_years = 0.0
        state.max_gap_years = 0.0
        state.max_area_km2 = max(state.get_areas()) if state.num_epochs > 0 else 0.0
        return

    # Extract years (floating point representation)
    years = np.array([e.date.year + e.date.timetuple().tm_yday / 365.25 for e in state.epochs])
    areas = np.array(state.get_areas())

    # Calculate gaps
    gaps = np.diff(years)
    state.max_gap_years = float(np.max(gaps)) if len(gaps) > 0 else 0.0
    state.time_span_years = float(years[-1] - years[0])
    state.max_area_km2 = float(np.max(areas))

    # OLS Regression for Area Trend (Rule 3.4-A, 4.4-A, 6.4-A)
    # Using scipy.stats.linregress to get slope and p-value
    res = stats.linregress(years, areas)
    
    # If the standard error is exactly 0 (e.g. all points on a line), p-value is 0 or nan.
    # Handle NaNs
    state.beta_hat = float(res.slope) if not np.isnan(res.slope) else 0.0
    state.p_value = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
