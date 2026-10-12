import os
import json
from pathlib import Path
from utils.structures import GlacierState

class OutputFormatter:
    def __init__(self, output_base_dir="output"):
        self.output_base = Path(__file__).parent.parent / output_base_dir
        
    def _create_class_dir(self, assigned_class: str) -> Path:
        class_dir = self.output_base / assigned_class
        os.makedirs(class_dir, exist_ok=True)
        return class_dir
        
    def export_glacier(self, state: GlacierState):
        """
        Exports the glacier metadata to its respective class folder.
        In a real scenario, this would also copy imagery/shapefiles.
        """
        if not state.assigned_class:
            state.assigned_class = "Unclassified"
            
        class_dir = self._create_class_dir(state.assigned_class)
        glacier_dir = class_dir / state.glacier_id
        os.makedirs(glacier_dir, exist_ok=True)
        
        metadata = {
            "Glacier ID": state.glacier_id,
            "GLIMS ID": state.glims_id,
            "Assigned class": state.assigned_class,
            "Confidence": state.confidence,
            "Observation years": [e.date.year for e in state.epochs],
            "Triggered rules": state.triggered_rules,
            "Computed metrics": {
                "max_area_km2": state.max_area_km2,
                "area_trend_beta_hat": state.beta_hat,
                "area_trend_p_value": state.p_value,
                "max_fragments": max((e.fragment_count for e in state.epochs), default=1)
            },
            "Warnings": state.flags,
            "Reasoning": state.notes
        }
        
        with open(glacier_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
