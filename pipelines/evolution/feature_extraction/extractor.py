import os
import rasterio
import numpy as np
from scipy import ndimage
from skimage import measure, morphology
from config.config import config

class FeatureExtractor:
    def __init__(self, season_dir: str):
        self.season_dir = season_dir
        self.pixel_size = config.pixel_width_clean_ice_m  # 10m typically
        
    def _get_ice_mask(self) -> np.ndarray:
        """
        Creates a binary ice/snow mask.
        Attempts to use SCL.tif first (Class 11 = Snow/Ice).
        If SCL is missing or entirely empty, falls back to NDSI if B03 and B11 are present.
        """
        scl_path = os.path.join(self.season_dir, "SCL.tif")
        if os.path.exists(scl_path):
            with rasterio.open(scl_path) as src:
                scl = src.read(1)
                # SCL Class 11 is Snow/Ice
                mask = (scl == 11)
                if np.any(mask):
                    return mask
                    
        # Fallback to NDSI: (Green - SWIR) / (Green + SWIR)
        b03_path = os.path.join(self.season_dir, "B03.tif")
        b11_path = os.path.join(self.season_dir, "B11.tif")
        
        if os.path.exists(b03_path) and os.path.exists(b11_path):
            with rasterio.open(b03_path) as src:
                green = src.read(1).astype(float)
            
            with rasterio.open(b11_path) as src:
                # B11 might be 20m, meaning it's half the size of B03.
                # In a robust implementation, we would reproject/resample.
                # For this simplified version, we assume they are already resampled or we resize them.
                swir = src.read(1, out_shape=green.shape).astype(float)
                
            ndsi = (green - swir) / (green + swir + 1e-10)
            mask = (ndsi > 0.4)
            return mask
            
        # If no data is found, return empty mask
        return np.zeros((10, 10), dtype=bool)

    def extract_features(self) -> dict:
        mask = self._get_ice_mask()
        
        # 1. Area (km2)
        pixel_area_km2 = (self.pixel_size / 1000.0) ** 2
        area_km2 = np.sum(mask) * pixel_area_km2
        
        if area_km2 == 0:
            return {
                "area_km2": 0.0,
                "perimeter_km": 0.0,
                "debris_fraction": 0.0,
                "terminus_y_geo": 0.0,
                "fragment_count": 0,
                "min_bridge_width_m": 0.0
            }
            
        # 2. Perimeter (km)
        contours = measure.find_contours(mask, 0.5)
        perimeter_pixels = sum(contour.shape[0] for contour in contours)
        perimeter_km = perimeter_pixels * (self.pixel_size / 1000.0)
        
        # 3. Fragment Count
        labeled_array, num_features = ndimage.label(mask)
        
        # Filter fragments below MMU
        fragment_count = 0
        for i in range(1, num_features + 1):
            frag_area = np.sum(labeled_array == i) * pixel_area_km2
            if frag_area >= config.mmu_km2:
                fragment_count += 1
                
        # 4. Bridge Width
        # Calculate distance transform from background
        dist_transform = ndimage.distance_transform_edt(mask)
        # In a real scenario, bridge width requires topological skeletonization.
        # Here we approximate the "narrowest point" by finding the minimum non-zero
        # distance from the skeleton. For simplicity, if fragment_count == 1 but there are
        # bottlenecks, we look at the skeleton.
        skeleton = morphology.skeletonize(mask)
        skeleton_distances = dist_transform[skeleton]
        
        min_bridge_width_m = float(np.min(skeleton_distances) * 2 * self.pixel_size) if skeleton_distances.size > 0 else 0.0
        
        # 5. Terminus Displacement fallback (using geographic Y-coordinate of the lowest ice pixel)
        y_indices, x_indices = np.where(mask)
        terminus_y_geo = 0.0
        
        if len(y_indices) > 0:
            terminus_y_pixel = int(np.max(y_indices))
            # Take median X at the lowest Y to find the center of the terminus
            xs_at_bottom = x_indices[y_indices == terminus_y_pixel]
            terminus_x_pixel = int(np.median(xs_at_bottom))
            
            # Project to geographic coordinates using the spatial transform of a reference TIF
            ref_path = os.path.join(self.season_dir, "SCL.tif")
            if not os.path.exists(ref_path):
                ref_path = os.path.join(self.season_dir, "B04.tif")
                
            if os.path.exists(ref_path):
                with rasterio.open(ref_path) as src:
                    x_geo, y_geo = src.xy(terminus_y_pixel, terminus_x_pixel)
                    terminus_y_geo = float(y_geo)
            else:
                terminus_y_geo = float(terminus_y_pixel * 10) # Crude fallback
        
        # 6. Debris Fraction
        # Count non-ice pixels (e.g. Bare Soil SCL=5 or Dark Area SCL=2) inside a convex hull
        debris_fraction = 0.0
        scl_path = os.path.join(self.season_dir, "SCL.tif")
        if os.path.exists(scl_path):
            with rasterio.open(scl_path) as src:
                scl = src.read(1)
                
                # Simple heuristic: debris is class 5 (Bare Soil) near the ice
                # Better: dilate the ice mask slightly to get the "glacier footprint" and check debris
                dilated_mask = ndimage.binary_dilation(mask, iterations=5)
                footprint_debris = np.sum(scl[dilated_mask] == 5)
                footprint_ice = np.sum(mask)
                total_footprint = footprint_ice + footprint_debris
                
                if total_footprint > 0:
                    debris_fraction = footprint_debris / total_footprint
                    
        return {
            "area_km2": float(area_km2),
            "perimeter_km": float(perimeter_km),
            "debris_fraction": float(debris_fraction),
            "terminus_y_geo": terminus_y_geo,
            "fragment_count": fragment_count,
            "min_bridge_width_m": min_bridge_width_m
        }
