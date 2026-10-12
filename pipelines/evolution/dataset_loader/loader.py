import os
from datetime import datetime
from typing import Generator
from utils.structures import GlacierState, EpochData

class DatasetLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_glaciers(self) -> Generator[GlacierState, None, None]:
        
        # 1. Loop through your glacier folders in the data directory
        for glacier_name in os.listdir(self.data_dir):
            glacier_path = os.path.join(self.data_dir, glacier_name)
            if not os.path.isdir(glacier_path):
                continue
                
            # The next level down is the GLIMS ID (there can be multiple)
            for glims_id in os.listdir(glacier_path):
                glims_path = os.path.join(glacier_path, glims_id)
                if not os.path.isdir(glims_path):
                    continue
                
                # Create the main container for this specific glacier lineage
                state = GlacierState(
                    glacier_id=f"{glacier_name}_{glims_id}",
                    glims_id=glims_id,
                    region="Unknown" # Update this if you have region metadata
                )
                
                # 2. Loop through all the years/seasons you have for this specific GLIMS ID
                for season_folder in os.listdir(glims_path):
                    if not os.path.isdir(os.path.join(glims_path, season_folder)):
                        continue
                        
                    # Parse '2016_JJA' to year and a rough month estimate
                    parts = season_folder.split('_')
                    if len(parts) != 2:
                        continue
                    
                    year = int(parts[0])
                    season = parts[1]
                    
                    # Map season to a rough month for the datetime object
                    season_to_month = {
                        "DJF": 1,  # Winter
                        "MAM": 4,  # Spring
                        "JJA": 7,  # Summer
                        "SON": 10  # Autumn
                    }
                    month = season_to_month.get(season, 1)
                    
                    # --- THIS IS WHERE YOU DO THE ACTUAL IMAGE PROCESSING ---
                    season_path = os.path.join(glims_path, season_folder)
                    
                    try:
                        from feature_extraction.extractor import FeatureExtractor
                        extractor = FeatureExtractor(season_path)
                        features = extractor.extract_features()
                    except Exception as e:
                        print(f"Error extracting features for {season_path}: {e}")
                        continue
                    # --------------------------------------------------------
                    
                    # 3. Create an EpochData object with the numbers you calculated
                    epoch = EpochData(
                        date=datetime(year=year, month=month, day=15),
                        area_km2=features["area_km2"],
                        perimeter_km=features["perimeter_km"],
                        debris_fraction=features["debris_fraction"],
                        terminus_displacement_m=features["terminus_y_geo"], # Store the raw Y geographic coordinate
                        fragment_count=features["fragment_count"],
                        min_bridge_width_m=features["min_bridge_width_m"]
                    )
                    
                    # Add this year to the glacier's history
                    state.epochs.append(epoch)
                    
                # 4. Once all years are added, yield the glacier to the main pipeline
                # (Sort epochs by date first so the temporal analysis works correctly)
                if state.epochs:
                    state.epochs.sort(key=lambda x: x.date)
                    yield state
