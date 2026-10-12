import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="config/thresholds.yaml"):
        # We assume the script is run from the project root.
        base_path = Path(__file__).parent.parent
        full_path = base_path / config_path
        
        with open(full_path, "r", encoding="utf-8") as f:
            self.settings = yaml.safe_load(f)

    @property
    def min_epochs(self):
        return self.settings["temporal"]["min_epochs"]

    @property
    def min_span_years(self):
        return self.settings["temporal"]["min_span_years"]

    @property
    def max_gap_years_high_conf(self):
        return self.settings["temporal"]["max_gap_years_high_conf"]

    @property
    def persistence_epochs(self):
        return self.settings["temporal"]["persistence_epochs"]

    @property
    def mmu_km2(self):
        return self.settings["spatial"]["mmu_km2"]

    @property
    def pixel_width_clean_ice_m(self):
        return self.settings["spatial"]["pixel_width_clean_ice_m"]

    @property
    def pixel_width_swir_m(self):
        return self.settings["spatial"]["pixel_width_swir_m"]

    @property
    def safety_factor(self):
        return self.settings["spatial"]["safety_factor"]

    @property
    def min_terminus_disp_clean_m(self):
        return self.settings["spatial"]["min_terminus_disp_clean_m"]

    @property
    def min_terminus_disp_debris_m(self):
        return self.settings["spatial"]["min_terminus_disp_debris_m"]

    @property
    def min_bridge_width_m(self):
        return self.settings["spatial"]["min_bridge_width_m"]

    @property
    def p_value_threshold(self):
        return self.settings["statistics"]["p_value_threshold"]

    @property
    def debris_tiers(self):
        return self.settings["classes"]["debris_tiers"]

# Singleton instance for easy access
config = Config()
