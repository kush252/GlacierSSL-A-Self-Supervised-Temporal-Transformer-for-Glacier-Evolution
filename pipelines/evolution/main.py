import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from dataset_loader.loader import DatasetLoader
from temporal_analysis.trend_ols import compute_temporal_features
from classification.hierarchy import classify_glacier
from export.output_formatter import OutputFormatter
from pipeline_logging.csv_logger import logger
from export.validation import validate_outputs

def run_pipeline(data_dir: str):
    print("Initializing Glacier Evolution Annotation Pipeline...")
    loader = DatasetLoader(data_dir)
    formatter = OutputFormatter()
    
    print("Loading glaciers and computing features...")
    for state in loader.load_glaciers():
        # Phase 4: Compute all required metrics (temporal features based on extracted epochs)
        compute_temporal_features(state)
        
        # Phase 5 & 6: Run rule engine and classification hierarchy
        classify_glacier(state)
        
        # Phase 7: Export to specific folder
        formatter.export_glacier(state)
        
        # Phase 8: Log to CSV
        logger.log_glacier(state)
        
        print(f"Processed {state.glacier_id} -> {state.assigned_class} (Confidence: {state.confidence})")
        
    print("Pipeline execution completed.")
    print("Running validation checks (Phase 9)...")
    validate_outputs()
    print("Validation finished.")

if __name__ == "__main__":
    # In a real scenario, argument parser would get the data directory
    data_dir = r"D:\Kush\2nd Year\projects\glacier\implementation\glacier_evolution_pipeline\temp_dataset\test"
    run_pipeline(data_dir)
