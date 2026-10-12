import csv
import os
from pathlib import Path
from utils.structures import GlacierState

class CSVLogger:
    def __init__(self, log_dir="output"):
        # We assume the script is run from the project root.
        base_path = Path(__file__).parent.parent
        self.log_dir = base_path / log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file = self.log_dir / "classification_log.csv"
        self._initialize_csv()

    def _initialize_csv(self):
        if not self.log_file.exists():
            with open(self.log_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Glacier ID",
                    "GLIMS ID",
                    "Class",
                    "Confidence",
                    "Triggered Rules",
                    "Max Area (km2)",
                    "Area Trend (beta_hat)",
                    "Terminus Change (m)",
                    "Max Fragment Count",
                    "Min Bridge Width (m)",
                    "Reason/Notes",
                    "Warnings/Flags"
                ])

    def log_glacier(self, state: GlacierState):
        with open(self.log_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Calculate aggregations safely
            max_area = max(state.get_areas()) if state.num_epochs > 0 else 0
            term_change = (state.epochs[-1].terminus_displacement_m - state.epochs[0].terminus_displacement_m) if state.num_epochs > 0 else 0
            max_fragments = max((e.fragment_count for e in state.epochs), default=1)
            min_bridge = min((e.min_bridge_width_m for e in state.epochs), default=0.0)
            
            writer.writerow([
                state.glacier_id,
                state.glims_id,
                state.assigned_class,
                state.confidence,
                "; ".join(state.triggered_rules),
                f"{max_area:.4f}",
                f"{state.beta_hat:.4f}",
                f"{term_change:.2f}",
                max_fragments,
                f"{min_bridge:.2f}",
                "; ".join(state.notes),
                "; ".join(state.flags)
            ])

logger = CSVLogger()
