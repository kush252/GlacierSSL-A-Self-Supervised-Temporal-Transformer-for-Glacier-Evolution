# 🛰️ Satellite Image Processing — The Complete Guide

> **A comprehensive, beginner-to-advanced reference on how satellite images work, what spectral bands are, why they exist, how to read them, how to visualise them, and how all of this was applied in the `data_understanding.ipynb` notebook for glacier analysis.**

---

## Table of Contents

1. [What Was Done in the Notebook — A Complete Walkthrough](#1-what-was-done-in-the-notebook--a-complete-walkthrough)
2. [The Two Worlds of Geospatial Data: Vector vs. Raster](#2-the-two-worlds-of-geospatial-data-vector-vs-raster)
3. [What Is a Satellite Image, Really?](#3-what-is-a-satellite-image-really)
4. [The Electromagnetic Spectrum and Why Satellites See More Than We Do](#4-the-electromagnetic-spectrum-and-why-satellites-see-more-than-we-do)
5. [Spectral Bands Explained — Every Band, What It Sees, and Why It Matters](#5-spectral-bands-explained--every-band-what-it-sees-and-why-it-matters)
6. [Complete Sentinel-2 Band Reference Table](#6-complete-sentinel-2-band-reference-table)
7. [How Your Eyes vs. How Satellites See the World](#7-how-your-eyes-vs-how-satellites-see-the-world)
8. [How to Visualise Satellite Data — True Color, False Color, and Single Bands](#8-how-to-visualise-satellite-data--true-color-false-color-and-single-bands)
9. [Pixel Values, Bit Depth, and Why You Must Normalise]V(#9-pixel-values-bit-depth-and-why-you-must-normalise)
10. [Percentile Stretching — The Professional Way to Normalise](#10-percentile-stretching--the-professional-way-to-normalise)
11. [How to Handle an Unknown Band](#11-how-to-handle-an-unknown-band)
12. [Spectral Indices — Doing Math with Bands](#12-spectral-indices--doing-math-with-bands)
13. [Coordinate Reference Systems (CRS) and Why They Matter](#13-coordinate-reference-systems-crs-and-why-they-matter)
14. [Windowed Reading — How to Crop a Satellite Image to Your Area of Interest](#14-windowed-reading--how-to-crop-a-satellite-image-to-your-area-of-interest)
15. [Multi-Resolution Bands — Why Bands Have Different Pixel Sizes](#15-multi-resolution-bands--why-bands-have-different-pixel-sizes)
16. [Time-Series Analysis — Watching a Glacier Change Over Years](#16-time-series-analysis--watching-a-glacier-change-over-years)
17. [Cloud-Optimised GeoTIFFs and STAC — Modern Satellite Data Access](#17-cloud-optimised-geotiffs-and-stac--modern-satellite-data-access)
18. [Scene Classification Layer (SCL) — Automatic Pixel Classification](#18-scene-classification-layer-scl--automatic-pixel-classification)
19. [Common False Color Composite Recipes](#19-common-false-color-composite-recipes)
20. [Summary and Key Takeaways](#20-summary-and-key-takeaways)

---

## 1. What Was Done in the Notebook — A Complete Walkthrough

The `data_understanding.ipynb` notebook is a multi-stage exploration that goes from **vector data** (glacier boundary polygons) all the way through to **raster satellite imagery** (actual pixel-based images from space). Here is exactly what was done, step by step:

### Phase 1: Vector Data Exploration (GLIMS Glacier Boundaries)

**What was loaded:** The GLIMS (Global Land Ice Measurements from Space) shapefile — `glims_polygons.shp` — which contains the **outlines** of 617,777 glaciers worldwide as vector polygons.

**Libraries used:** `geopandas`

**What was explored:**
- The shapefile was loaded and its 36 columns inspected (`glac_id`, `glac_name`, `area`, `geog_area`, `geometry`, etc.)
- The `geometry` column contains the actual polygon shapes (sequences of lat/lon coordinates that trace the boundary of each glacier)
- **Centroids** (centre-point latitude and longitude) were calculated from each polygon using `gdf.geometry.centroid`
- Glaciers were sorted by area to find the **largest glaciers** globally — Grosser Aletschgletscher in the European Alps, Gornergletscher, Bezengi in the Caucasus, etc.
- Glaciers were filtered by geographic region using string matching on the `geog_area` column:
  - `"Greenland"` — to find Arctic glaciers
  - `"Himal"` — to find Himalayan glaciers like Khumbu, Ngojumba, Yalung, Langtang, Dhaulagiri
  - `"Cordillera|Alaska|Juneau|Chugach|Kenai"` — to find North American glaciers like Taku Glacier

**Why this matters:** These polygons define *where* the glaciers are. The polygon bounds (bounding box coordinates) are later used to search for and crop satellite images over those exact locations.

### Phase 2: Connecting to a Satellite Data Catalog (Microsoft Planetary Computer)

**What was done:** The notebook connects to Microsoft's Planetary Computer STAC (SpatioTemporal Asset Catalog) API to search for real Sentinel-2 satellite images.

**Libraries used:** `pystac_client`, `planetary_computer`

**The search:**
```python
search = catalog.search(
    collections=["sentinel-2-l2a"],    # Sentinel-2 Level-2A (atmospherically corrected)
    bbox=bbox,                          # Bounding box from the Aletsch glacier polygon
    datetime="2024-07-01/2024-09-30",  # Summer months (minimal snow cover beyond glacier)
    query={"eo:cloud_cover": {"lt": 20}}  # Only images with <20% cloud cover
)
```

**What was discovered:** Each Sentinel-2 scene contains many **assets** (individual band files):
```
dict_keys(['AOT', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
           'B09', 'B11', 'B12', 'B8A', 'SCL', 'WVP', 'visual',
           'safe-manifest', 'granule-metadata', ...])
```

Each `B__` key is a separate spectral band stored as its own GeoTIFF file. This is a critical insight: **a single satellite "image" is actually a collection of many separate single-band image files**.

### Phase 3: Downloading and Visualising RGB Bands (True Color)

**What was done:** Bands B04 (Red), B03 (Green), and B02 (Blue) were downloaded individually using `rasterio` and stacked into an RGB image.

**Key code pattern:**
```python
bands = {}
for band in ["B02", "B03", "B04"]:
    with rasterio.open(item.assets[band].href) as src:
        bands[band] = src.read(1)  # read(1) = read the first (and only) band in the file

rgb = np.dstack([bands["B04"], bands["B03"], bands["B02"]])  # Stack as R, G, B
```

**Critical step — Percentile Stretch normalisation:**
```python
rgb = rgb.astype(np.float32)
p2, p98 = np.percentile(rgb, (2, 98))
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
```

The raw pixel values are 16-bit integers (0–10000 for Sentinel-2 reflectance). Matplotlib expects float values between 0–1. Simple min-max normalisation fails because a few extreme outlier pixels (very bright clouds, sensor noise) would compress the rest of the image into near-darkness. **Percentile stretching** uses the 2nd and 98th percentiles instead, ignoring outliers and producing a visually appealing image.

### Phase 4: Inspecting Raster Metadata (CRS, Resolution, Bounds)

**What was done:**
```python
with rasterio.open(item.assets["B04"].href) as src:
    print("CRS:", src.crs)          # EPSG:32632 (UTM Zone 32N)
    print("Width:", src.width)      # 10980 pixels
    print("Height:", src.height)    # 10980 pixels
    print("Bounds:", src.bounds)    # geographic extent in metres
    print("Resolution:", src.res)   # (10.0, 10.0) — 10m per pixel
```

This revealed that the full Sentinel-2 tile is **10980×10980 pixels** — that's roughly a 110km × 110km area on the ground.

### Phase 5: CRS Reprojection and Windowed Cropping

**The problem:** The glacier polygons from GLIMS are in **EPSG:4326** (latitude/longitude in degrees), but the Sentinel-2 images are in **EPSG:32632** (UTM projection in metres). You cannot directly compare coordinates between different CRS.

**The solution:**
```python
aletsch = aletsch.to_crs(target_crs)  # Reproject polygon to match the satellite image CRS
```

Then, instead of downloading the entire 10980×10980 tile, a **window** was used to read only the pixels covering the glacier:
```python
from rasterio.windows import from_bounds
window = from_bounds(minx, miny, maxx, maxy, src.transform)
red_crop = src.read(1, window=window)
```

This is dramatically faster and more memory-efficient than downloading the full tile.

### Phase 6: Visualising Non-Visible Bands (NIR and SWIR)

**What was done:** Bands B08 (Near-Infrared), B11 (SWIR-1), and B12 (SWIR-2) were downloaded and each was plotted individually in **grayscale**:

```python
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, band in enumerate(["B08", "B11", "B12"]):
    img = bands_extra[band]
    p2, p98 = np.percentile(img, (2, 98))
    img = np.clip((img - p2) / (p98 - p2), 0, 1)
    ax[i].imshow(img, cmap="gray")  # Single band = grayscale
    ax[i].set_title(band)
```

**Why grayscale?** A single band is a 2D array of intensity values — it has no inherent "colour." We use `cmap="gray"` to map low values to black and high values to white. This lets you see the spatial patterns: in SWIR bands, snow/ice appears **dark** (it absorbs SWIR energy), while bare rock and clouds appear bright.

### Phase 7: Checking Resolution Differences Across Bands

**What was discovered:**
```
B02  (10.0, 10.0)  10980  10980   # 10m resolution
B03  (10.0, 10.0)  10980  10980   # 10m resolution
B04  (10.0, 10.0)  10980  10980   # 10m resolution
B08  (10.0, 10.0)  10980  10980   # 10m resolution
B11  (20.0, 20.0)  5490   5490    # 20m resolution!
B12  (20.0, 20.0)  5490   5490    # 20m resolution!
```

B11 and B12 have **half the resolution** of the visible bands. They have different pixel sizes (20m vs 10m), which means when you stack them for composite images or index calculation, you need to **resample** one to match the other. This is a fundamental challenge in multispectral satellite imagery.

### Phase 8: Time-Series Analysis (Tracking Glacier Change Over Years)

**What was done:**
1. All Sentinel-2 scenes from **2017–2025** were searched
2. For each year, the **best summer scene** (July–September, lowest cloud cover) was selected
3. An RGB crop of the Aletsch Glacier was generated for each year
4. All 9 years were plotted in a **3×3 grid** to visually compare glacier retreat

This is the core of temporal glacier analysis — watching the same glacier from the same viewpoint across multiple years to observe changes.

### Phase 9: Working with Pre-Downloaded Local Band Files

**What was done:** The notebook also loads individual band `.tif` files from a local dataset (`new_pipeline_dataset/train/BONDHUSBREA/G006322E60028N/2016_DJF/`), where each band is stored as a separate file (B02.tif, B03.tif, B04.tif, B08.tif, B11.tif, B12.tif, SCL.tif).

The same RGB stacking and percentile stretching techniques were applied to this local data, and band metadata (width, height, transform, CRS) was inspected for all bands.

---

## 2. The Two Worlds of Geospatial Data: Vector vs. Raster

All geographic data falls into one of two fundamental types:

### Vector Data
- **What it is:** Mathematical shapes — points, lines, and polygons — defined by sequences of coordinates
- **File formats:** `.shp` (Shapefile), `.geojson`, `.gpkg` (GeoPackage)
- **Example in your project:** The GLIMS shapefile contains 617,777 glacier polygons. Each glacier's boundary is a list of (longitude, latitude) coordinate pairs that trace its outline
- **What it's good for:** Precise boundaries, administrative regions, roads, building footprints
- **What it's NOT:** It contains zero information about what the surface *looks like*. A glacier polygon tells you where the glacier is, but not whether it's covered in fresh snow, dirty ice, or meltwater
- **How you read it:** `geopandas.read_file("file.shp")` → returns a GeoDataFrame (table with a geometry column)

### Raster Data (Satellite Images)
- **What it is:** A regular grid of pixels (like a digital photo), where each pixel stores a number representing the amount of energy reflected or emitted from that tiny patch of Earth
- **File formats:** `.tif` / `.tiff` (GeoTIFF — the most common), `.nc` (NetCDF), `.hdf` (HDF5), `.jp2` (JPEG2000)
- **Example in your project:** Sentinel-2 band files like `B04.tif` — a 10980×10980 grid where each pixel covers a 10m × 10m patch of ground
- **What it's good for:** Actual visual and spectral analysis — seeing what the surface looks like, measuring reflectance, detecting changes
- **How you read it:** `rasterio.open("file.tif")` → returns a dataset object; use `.read(1)` to get a 2D numpy array of pixel values

### How They Work Together

In your notebook, you combined both:
1. **Vector** → Identified the Aletsch Glacier polygon, extracted its bounding box coordinates
2. **Raster** → Used those coordinates to search for and crop satellite images covering that exact area

This vector-guided-raster-analysis workflow is the standard approach in remote sensing.

---

## 3. What Is a Satellite Image, Really?

A satellite image is fundamentally different from a photograph taken with your phone:

### A Phone Camera
- Has a single sensor with a **Bayer filter** that captures Red, Green, and Blue simultaneously
- Produces one file with 3 channels (RGB)
- Pixel values represent colour intensity (0–255 for 8-bit JPEG)
- The image "just works" on your screen

### A Satellite Sensor (like Sentinel-2's MSI)
- Has **13 separate detectors**, each with a different optical filter
- Each detector captures a different slice of the electromagnetic spectrum
- Each detector produces a **separate image file** (one `.tif` per band)
- Pixel values represent **surface reflectance** — the fraction of incoming sunlight reflected back to the satellite (typically scaled as integers: 0–10000 for Sentinel-2, 0–65535 for Landsat)
- The raw data looks like nothing on your screen — you must manually combine and stretch bands to create a viewable image

### The Data Structure

A single Sentinel-2 "scene" (one overpass of one location) is not a single file. It's a **folder of files**:
```
S2B_MSIL2A_20240823T101559_...
├── B01.tif    (Coastal Aerosol, 60m resolution)
├── B02.tif    (Blue, 10m)
├── B03.tif    (Green, 10m)
├── B04.tif    (Red, 10m)
├── B05.tif    (Vegetation Red Edge 1, 20m)
├── B06.tif    (Vegetation Red Edge 2, 20m)
├── B07.tif    (Vegetation Red Edge 3, 20m)
├── B08.tif    (NIR, 10m)
├── B8A.tif    (Narrow NIR, 20m)
├── B09.tif    (Water Vapour, 60m)
├── B11.tif    (SWIR-1, 20m)
├── B12.tif    (SWIR-2, 20m)
├── SCL.tif    (Scene Classification Layer, 20m)
├── AOT.tif    (Aerosol Optical Thickness)
├── WVP.tif    (Water Vapour Pressure)
└── metadata.xml
```

Each `.tif` is a grayscale image — a 2D grid of numbers. To create something you can see on screen, you pick 3 bands and assign them to the Red, Green, and Blue channels of your monitor.

---

## 4. The Electromagnetic Spectrum and Why Satellites See More Than We Do

### What Is Light?

Light is electromagnetic radiation — waves of energy travelling through space. These waves come in different **wavelengths**. Human eyes can only detect a tiny range of wavelengths:

```
← Shorter wavelength                                    Longer wavelength →
← Higher energy                                         Lower energy →

┌─────────┬─────────┬──────┬───────┬─────┬──────────┬───────────┬──────────┐
│  Gamma  │  X-Ray  │  UV  │Visible│ NIR │   SWIR   │  Thermal  │Microwave │
│  Rays   │         │      │ Light │     │          │ Infrared  │          │
└─────────┴─────────┴──────┴───────┴─────┴──────────┴───────────┴──────────┘
                           │       │
                           │ HUMAN │
                           │ EYES  │
                           │ ONLY  │
                           │ SEE   │
                           │ THIS  │
                           │ PART  │
                           └───────┘
                        ~400nm to ~700nm

Satellite sensors see from ~440nm to ~2200nm and beyond!
```

### Why Does This Matter?

Different materials on Earth's surface **reflect each wavelength differently**. Think of it like a fingerprint:

- **Healthy vegetation:** Absorbs Red and Blue (for photosynthesis), but reflects a HUGE amount of Near-Infrared. In NIR images, forests glow bright white.
- **Snow and ice:** Reflects almost all visible light (looks white to our eyes), but **strongly absorbs SWIR** (appears very dark in SWIR images)
- **Clouds:** Reflect both visible AND SWIR light (appears bright in both)
- **Water:** Absorbs almost everything beyond Blue. In NIR and SWIR, water bodies appear jet black
- **Bare soil/rock:** Moderate and relatively flat reflectance across all bands

This is why a simple RGB image is not enough for scientific analysis. In an RGB image:
- Snow and clouds both look white → impossible to tell apart
- Healthy and stressed vegetation both look green-ish → hard to distinguish
- Shallow and deep water both look dark → no way to separate

But add NIR and SWIR bands, and suddenly each material has a unique spectral signature that a computer can identify.

---

## 5. Spectral Bands Explained — Every Band, What It Sees, and Why It Matters

### Visible Bands (What your eyes can see)

**Band 2 — Blue (~490nm)**
- Penetrates water better than any other visible band
- Useful for: bathymetry (water depth mapping), distinguishing soil from vegetation
- Downside: most affected by atmospheric scattering (why the sky is blue), so images can look hazy

**Band 3 — Green (~560nm)**
- Peak reflectance of healthy vegetation (chlorophyll reflects green light — that's why plants look green to us)
- Used in: NDSI calculation for snow mapping, water quality assessment
- Appears brightest over dense vegetation and shallow turbid water

**Band 4 — Red (~665nm)**
- Strongly absorbed by chlorophyll in plants (plants eat red light for photosynthesis)
- Used in: NDVI calculation (vegetation health), distinguishing vegetation from bare soil
- Vegetation appears DARK in this band; bare soil appears bright

### Vegetation Red Edge Bands (The boundary between visible and invisible)

**Bands 5, 6, 7 — Red Edge (~705nm, ~740nm, ~783nm)** *(Sentinel-2 specific)*
- These capture the dramatic transition zone where vegetation reflectance skyrockets from low (red) to high (NIR)
- The exact position of this "red edge" shifts based on plant health, chlorophyll content, and species
- Think of it as a "stress detector" for plants — the red edge shifts when vegetation is under drought, disease, or pollution stress
- Resolution: 20m (lower than visible bands)

### Near-Infrared (NIR) Bands

**Band 8 — NIR (~842nm)**
- THE most important band for vegetation analysis
- Healthy green vegetation reflects ~50% of NIR light (vs ~10% of visible light)
- Water absorbs almost all NIR → appears jet black
- Used in: NDVI, vegetation mapping, water body delineation
- Resolution: 10m (same as visible bands — this is a huge advantage of Sentinel-2)

**Band 8A — Narrow NIR (~865nm)**
- Narrower wavelength range than B08, less affected by water vapour absorption
- Better for vegetation studies where atmospheric effects matter
- Resolution: 20m

### Shortwave Infrared (SWIR) Bands — Critical for Glaciers!

**Band 11 — SWIR-1 (~1610nm)**
- Sensitive to leaf water content and soil moisture
- Snow/ice **strongly absorbs** this wavelength → appears dark
- Clouds reflect it → appear bright
- This is how you tell snow from clouds — the single most important trick in glacier remote sensing!
- Used in: NDSI (snow detection), burn scar mapping, geological mapping
- Resolution: 20m

**Band 12 — SWIR-2 (~2190nm)**
- Even more sensitive to moisture content
- Excellent for geological/mineral mapping
- Also useful for snow/cloud discrimination
- Can penetrate thin cirrus clouds
- Resolution: 20m

### Atmospheric Bands (Not about the surface — about the air above it)

**Band 1 — Coastal Aerosol (~443nm)**
- Designed to detect aerosols (dust, smoke, pollution) in the atmosphere
- Also useful for shallow coastal water mapping
- Resolution: 60m (very coarse)

**Band 9 — Water Vapour (~945nm)**
- Measures water vapour content in the atmosphere
- Used for atmospheric correction (removing the effect of the atmosphere from other bands)
- Not typically used for surface analysis
- Resolution: 60m

---

## 6. Complete Sentinel-2 Band Reference Table

| Band | Name | Central Wavelength | Bandwidth | Resolution | Primary Use |
|------|------|-------------------|-----------|------------|-------------|
| B01 | Coastal Aerosol | 443 nm | 20 nm | 60m | Aerosol detection, shallow water |
| B02 | Blue | 490 nm | 65 nm | 10m | Water penetration, soil vs vegetation |
| B03 | Green | 560 nm | 35 nm | 10m | Vegetation peak, NDSI snow index |
| B04 | Red | 665 nm | 30 nm | 10m | Chlorophyll absorption, NDVI |
| B05 | Red Edge 1 | 705 nm | 15 nm | 20m | Vegetation stress detection |
| B06 | Red Edge 2 | 740 nm | 15 nm | 20m | Vegetation biophysical parameters |
| B07 | Red Edge 3 | 783 nm | 20 nm | 20m | Vegetation structure |
| B08 | NIR | 842 nm | 115 nm | 10m | Vegetation biomass, water boundaries |
| B8A | Narrow NIR | 865 nm | 20 nm | 20m | Vegetation, atmospheric correction |
| B09 | Water Vapour | 945 nm | 20 nm | 60m | Atmospheric water vapour |
| B10 | Cirrus | 1375 nm | 30 nm | 60m | Cirrus cloud detection (L1C only) |
| B11 | SWIR-1 | 1610 nm | 90 nm | 20m | Snow/ice, moisture, burn scars |
| B12 | SWIR-2 | 2190 nm | 180 nm | 20m | Geology, minerals, snow |

**For comparison — Landsat 8/9 Bands:**

| Band | Name | Wavelength | Resolution |
|------|------|-----------|------------|
| B1 | Coastal Aerosol | 435–451 nm | 30m |
| B2 | Blue | 452–512 nm | 30m |
| B3 | Green | 533–590 nm | 30m |
| B4 | Red | 636–673 nm | 30m |
| B5 | NIR | 851–879 nm | 30m |
| B6 | SWIR-1 | 1566–1651 nm | 30m |
| B7 | SWIR-2 | 2107–2294 nm | 30m |
| B8 | Panchromatic | 503–676 nm | 15m |
| B10 | TIR-1 | 10600–11190 nm | 100m |
| B11 | TIR-2 | 11500–12510 nm | 100m |

> **⚠️ Warning:** Band NUMBERS are NOT universal! Landsat's B5 is NIR, but Sentinel-2's B05 is Red Edge. Always check the satellite's documentation to know which wavelength each band number represents.

---

## 7. How Your Eyes vs. How Satellites See the World

Imagine looking at a glacier scene. Here's what different "viewers" would see:

### Human Eyes (RGB only)
- Snow: **white** (reflects all visible light equally)
- Clouds: **white** (also reflects all visible light)
- Water: **blue-ish to dark** (absorbs red, reflects some blue)
- Vegetation: **green** (reflects green, absorbs red and blue)
- Rock: **grey/brown** (moderate reflectance across visible bands)
- **Problem:** Snow and clouds are indistinguishable!

### Satellite with NIR (Band 8)
- Snow: **bright** (high NIR reflectance)
- Clouds: **bright** (high NIR reflectance)
- Water: **BLACK** (total NIR absorption — this is how you map water bodies)
- Vegetation: **EXTREMELY BRIGHT** (much brighter than in visible — this is the "secret" of NIR)
- Rock: **medium grey**

### Satellite with SWIR (Band 11)
- Snow/Ice: **DARK** (strong SWIR absorption — this is the key!)
- Clouds: **BRIGHT** (clouds still reflect SWIR)
- Water: **black** (absorbs everything)
- Vegetation: **medium-dark** (moderate SWIR reflectance)
- Rock: **bright** (rocks reflect SWIR well)
- **Solution:** Snow appears dark, clouds appear bright → perfectly separated!

### The Key Insight

No single band tells the whole story. Each band reveals different properties of the surface. By combining bands strategically — either as composite images (assigning bands to R/G/B channels) or through mathematical indices (like NDSI, NDVI) — we can extract specific information that is impossible to see with the naked eye.

---

## 8. How to Visualise Satellite Data — True Color, False Color, and Single Bands

### Method 1: Single Band — Grayscale

When you have just one band, you have a 2D array of numbers. You plot it using a **colormap** (cmap). This is exactly what your notebook did for B08, B11, and B12:

```python
# Read a single band
with rasterio.open("B08.tif") as src:
    nir = src.read(1)

# Normalise using percentile stretch
p2, p98 = np.percentile(nir, (2, 98))
nir_display = np.clip((nir - p2) / (p98 - p2), 0, 1)

# Plot in grayscale
plt.imshow(nir_display, cmap="gray")
plt.colorbar()
plt.title("Band 8 — Near Infrared")
```

Other useful colormaps for single bands:
- `cmap="gray"` — Standard grayscale (black = low, white = high)
- `cmap="viridis"` — Perceptually uniform, good for scientific plots
- `cmap="RdYlGn"` — Red to Green, great for NDVI (vegetation health)
- `cmap="coolwarm"` — Blue to Red, great for NDSI (snow index)
- `cmap="terrain"` — For elevation data (DEM)

### Method 2: True Color Composite (RGB)

Map the visible bands to your screen's RGB channels in the natural order:
- Red channel ← Band 4 (Red)
- Green channel ← Band 3 (Green)
- Blue channel ← Band 2 (Blue)

```python
rgb = np.dstack([bands["B04"], bands["B03"], bands["B02"]])
rgb = rgb.astype(np.float32)
p2, p98 = np.percentile(rgb, (2, 98))
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
plt.imshow(rgb)
```

This looks like what a human astronaut would see from space.

### Method 3: False Color Composite

Map NON-visible bands to R/G/B channels. The resulting image has "false" colours — they don't represent what things actually look like, but they reveal information that's invisible in true colour.

The process is identical to true colour — just swap which band goes into which channel:

```python
# Example: Color Infrared composite
false_color = np.dstack([
    bands["B08"],   # NIR → Red channel
    bands["B04"],   # Red → Green channel
    bands["B03"]    # Green → Blue channel
])
# Same normalisation as before
p2, p98 = np.percentile(false_color, (2, 98))
false_color = np.clip((false_color - p2) / (p98 - p2), 0, 1)
plt.imshow(false_color)
```

In this composite, healthy vegetation appears **bright red** (because vegetation reflects tons of NIR, which is mapped to the red channel). This makes it immediately obvious where vegetation is thriving vs stressed.

---

## 9. Pixel Values, Bit Depth, and Why You Must Normalise

### The Problem

Your computer screen expects colour values between **0 and 1** (float) or **0 and 255** (8-bit integer). But satellite data doesn't come in that range:

| Satellite | Data Type | Value Range | What the values mean |
|-----------|-----------|-------------|---------------------|
| Sentinel-2 L2A | uint16 | 0–10000 | Surface reflectance × 10000 (so 5000 = 50% reflectance) |
| Landsat 8/9 L2SP | uint16 | 0–65535 | Scaled surface reflectance |
| MODIS | int16 | -100 to 16000 | Reflectance × 10000 |
| Your phone | uint8 | 0–255 | "Display-ready" colour |

If you feed a uint16 array (values like 1200, 3500, 8000) directly to `plt.imshow()`, matplotlib will try to auto-scale it, but the result is often terrible — either washed out or way too dark.

### The Solution: Normalisation

**Naive approach — Min-Max scaling (don't use this):**
```python
normalised = (img - img.min()) / (img.max() - img.min())
```
Problem: If even one pixel has an extreme value (sensor error, very bright cloud), `img.max()` is dominated by that outlier, and 99.9% of your image gets compressed into near-zero values. The result looks almost completely black with a single white dot.

**Professional approach — Percentile stretching (use this):**
```python
p2, p98 = np.percentile(img, (2, 98))
normalised = np.clip((img - p2) / (p98 - p2), 0, 1)
```
This ignores the extreme top and bottom 2% of pixel values, stretching the "meaningful" range of values across the full 0–1 display range. This is what your notebook uses throughout.

---

## 10. Percentile Stretching — The Professional Way to Normalise

Let's break down what percentile stretching does with a concrete example:

```
Raw pixel values in a band: [0, 100, 200, 500, 800, 1000, 1200, 1500, 2000, 9500]
```

**Step 1: Calculate percentiles**
```python
p2  = np.percentile(data, 2)   # ~100  (value below which 2% of pixels fall)
p98 = np.percentile(data, 98)  # ~9500 (value below which 98% of pixels fall)
```

**Step 2: Stretch to 0–1 range**
```python
stretched = (data - p2) / (p98 - p2)
# [~0, 0, 0.01, 0.04, 0.07, 0.1, 0.12, 0.15, 0.2, 1.0]
```

**Step 3: Clip to [0, 1]**
```python
final = np.clip(stretched, 0, 1)
```

Any pixel darker than p2 becomes 0 (black). Any pixel brighter than p98 becomes 1 (white). Everything in between is linearly scaled.

### Per-Band vs. Global Stretching

**Global stretch (what your notebook does for RGB):**
```python
p2, p98 = np.percentile(rgb, (2, 98))  # single p2/p98 for all 3 bands
rgb = np.clip((rgb - p2) / (p98 - p2), 0, 1)
```
Simple and fast. Works well when bands have similar value ranges.

**Per-band stretch (more accurate colours):**
```python
for i in range(3):
    vals = rgb[:,:,i][mask]  # only use valid (non-background) pixels
    p2 = np.percentile(vals, 2)
    p98 = np.percentile(vals, 98)
    rgb[:,:,i] = np.clip((rgb[:,:,i] - p2) / (p98 - p2), 0, 1)
```
More work, but preserves the relative colour balance better. Your notebook uses this approach when processing local band files.

---

## 11. How to Handle an Unknown Band

You've downloaded a satellite image and it has a band you've never seen before. How do you figure out what it is and how to visualise it?

### Step 1: Read the Metadata

Every GeoTIFF has embedded metadata. Read it with rasterio:

```python
with rasterio.open("mystery_band.tif") as src:
    print("CRS:", src.crs)                    # Coordinate system
    print("Resolution:", src.res)             # Pixel size in CRS units
    print("Bounds:", src.bounds)              # Geographic extent
    print("Data type:", src.dtypes)           # uint8, uint16, float32, etc.
    print("NoData value:", src.nodata)        # Value used for "no data" pixels
    print("Band count:", src.count)           # Number of bands in this file
    print("Description:", src.descriptions)   # Sometimes contains band name
    print("Tags:", src.tags())                # Additional metadata
    print("Width × Height:", src.width, src.height)
```

Also check for accompanying XML or MTL files that describe the satellite product.

### Step 2: Look at the Statistics

```python
data = src.read(1)
valid = data[data != src.nodata] if src.nodata else data[data > 0]

print("Min:", valid.min())
print("Max:", valid.max())
print("Mean:", valid.mean())
print("Std:", valid.std())

plt.figure(figsize=(10, 4))
plt.hist(valid.ravel(), bins=200)
plt.title("Pixel Value Distribution")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()
```

The histogram tells you a lot:
- **Bimodal** (two peaks) → probably land and water, or surface and cloud
- **Values 0–10000** → likely Sentinel-2 reflectance
- **Values 0–255** → already 8-bit, possibly pre-processed
- **Floating point values 0–1** → likely a precomputed index (NDVI, NDSI, etc.)
- **Values 0–11** → likely a classification layer (SCL)

### Step 3: Plot It with Different Colormaps

```python
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Normalise once
p2, p98 = np.percentile(valid, (2, 98))
display = np.clip((data.astype(float) - p2) / (p98 - p2), 0, 1)

for ax, cmap in zip(axes, ["gray", "viridis", "terrain", "coolwarm"]):
    ax.imshow(display, cmap=cmap)
    ax.set_title(cmap)
    ax.axis("off")

plt.tight_layout()
plt.show()
```

### Step 4: Deduce the Band from Resolution and Visual Pattern

| Resolution | Likely Band (Sentinel-2) |
|-----------|--------------------------|
| 10m | B02, B03, B04, B08 |
| 20m | B05, B06, B07, B8A, B11, B12, SCL |
| 60m | B01, B09 |

Visual clues:
- **Water appears black** → probably NIR or SWIR
- **Vegetation appears very bright** → probably NIR (B08)
- **Snow appears dark but clouds bright** → probably SWIR (B11 or B12)
- **Image has discrete integer values 0–11** → Scene Classification Layer (SCL)
- **Vegetation appears dark** → probably Red (B04)

---

## 12. Spectral Indices — Doing Math with Bands

Instead of just looking at bands visually, we can perform pixel-by-pixel arithmetic to create new derived images that highlight specific phenomena:

### NDVI — Normalised Difference Vegetation Index

```python
# NDVI = (NIR - Red) / (NIR + Red)
ndvi = (B08.astype(float) - B04.astype(float)) / (B08 + B04 + 1e-10)
# Values range from -1 to +1
# -1 to 0: water, bare soil, snow, clouds
# 0 to 0.3: sparse vegetation, grassland
# 0.3 to 0.6: moderate vegetation
# 0.6 to 1.0: dense, healthy vegetation (forest, crops)

plt.imshow(ndvi, cmap="RdYlGn", vmin=-0.5, vmax=0.8)
plt.colorbar(label="NDVI")
```

### NDSI — Normalised Difference Snow Index (Critical for Glaciers!)

```python
# NDSI = (Green - SWIR) / (Green + SWIR)
ndsi = (B03.astype(float) - B11.astype(float)) / (B03 + B11 + 1e-10)
# Values range from -1 to +1
# > 0.4: Snow/ice (the standard threshold)
# < 0.4: Not snow

plt.imshow(ndsi, cmap="coolwarm", vmin=-0.5, vmax=1.0)
plt.colorbar(label="NDSI")
```

**Why NDSI works:** Green light is strongly reflected by snow AND clouds. But SWIR light is absorbed by snow and reflected by clouds. So:
- Snow: high Green, low SWIR → high NDSI
- Cloud: high Green, high SWIR → low NDSI
- Water: low Green, low SWIR → variable NDSI
- Rock: moderate both → low NDSI

### NDWI — Normalised Difference Water Index

```python
# NDWI = (Green - NIR) / (Green + NIR)
ndwi = (B03.astype(float) - B08.astype(float)) / (B03 + B08 + 1e-10)
# > 0.3: Water
# < 0.3: Not water
```

### The General Pattern

All normalised difference indices follow the same formula:
```
Index = (Band_that_reflects - Band_that_absorbs) / (Band_that_reflects + Band_that_absorbs)
```

The result is always between -1 and +1, making it easy to apply universal thresholds regardless of illumination conditions, sensor differences, or atmospheric state. This self-normalising property is why these indices are so powerful.

---

## 13. Coordinate Reference Systems (CRS) and Why They Matter

### The Problem Your Notebook Solved

The GLIMS glacier polygons are in **EPSG:4326** — latitude and longitude in degrees (e.g., lat=46.49, lon=8.02). The Sentinel-2 images are in **EPSG:32632** — UTM Zone 32N, which uses metres (e.g., x=427531, y=5151289).

These are completely different coordinate systems. If you use the lat/lon bounds from the polygon directly to crop the UTM raster, you get nonsense results (or an error).

### The Solution: Reprojection

```python
# Convert the glacier polygon from degrees to metres
aletsch = aletsch.to_crs(target_crs)  # target_crs = EPSG:32632 from the raster
```

Now the polygon coordinates are in the same coordinate system as the raster, and you can use them directly for cropping.

### Common CRS Systems

| EPSG Code | Name | Units | When Used |
|-----------|------|-------|-----------|
| 4326 | WGS84 | Degrees (lat/lon) | GPS, vector data, web maps |
| 32632 | UTM Zone 32N | Metres | Sentinel-2 tiles over central Europe |
| 32645 | UTM Zone 45N | Metres | Sentinel-2 tiles over Nepal/Himalayas |
| 3857 | Web Mercator | Metres | Google Maps, OpenStreetMap |

---

## 14. Windowed Reading — How to Crop a Satellite Image to Your Area of Interest

A full Sentinel-2 tile is 10980×10980 pixels (about 110×110 km). If you only need a small glacier, reading the entire tile wastes time and memory. **Windowed reading** solves this:

```python
from rasterio.windows import from_bounds

# Get glacier extent in the raster's CRS (after reprojection!)
minx, miny, maxx, maxy = glacier_polygon.bounds

with rasterio.open("B04.tif") as src:
    # Create a window object from geographic bounds
    window = from_bounds(minx, miny, maxx, maxy, src.transform)
    
    # Read ONLY the pixels within that window
    cropped = src.read(1, window=window)
    
print(cropped.shape)  # Much smaller than (10980, 10980)!
```

`src.transform` is the **affine transformation matrix** that maps pixel coordinates (row, column) to geographic coordinates (x, y). The `from_bounds` function uses this to calculate exactly which pixel rows and columns correspond to your geographic extent.

---

## 15. Multi-Resolution Bands — Why Bands Have Different Pixel Sizes

### The Physical Reason

There's a fundamental trade-off in sensor design:
- **Smaller pixels** (higher resolution) = each pixel collects less light = noisier signal
- **Larger pixels** (lower resolution) = each pixel collects more light = cleaner signal

SWIR detectors are inherently noisier than visible-light detectors (SWIR photons carry less energy). To get a usable signal-to-noise ratio, SWIR bands need bigger pixels. That's why B11 and B12 are 20m while B02–B04 are 10m.

### The Practical Problem

As your notebook discovered:
```
B04  10m  10980×10980
B11  20m   5490× 5490   ← Half the pixels in each dimension!
```

You cannot directly stack a 10980×10980 B04 array with a 5490×5490 B11 array. They have different shapes.

### Solutions

**Option 1: Resample the 20m band to 10m (upsampling)**
```python
from rasterio.enums import Resampling

with rasterio.open("B11.tif") as src:
    # Read at 2× the resolution (interpolate to fill gaps)
    swir = src.read(
        1,
        out_shape=(10980, 10980),  # Force output to 10m grid size
        resampling=Resampling.bilinear
    )
```

**Option 2: Resample the 10m band to 20m (downsampling)**
```python
with rasterio.open("B04.tif") as src:
    red = src.read(
        1,
        out_shape=(5490, 5490),
        resampling=Resampling.average
    )
```

**Option 3: Use windowed reading (they auto-align if using the same geographic bounds)**

When you use `from_bounds` with a window, rasterio reads whatever pixels fall within those bounds. The 10m band will produce more pixels than the 20m band for the same geographic area. Your notebook handled this implicitly because it was processing bands separately rather than stacking them.

---

## 16. Time-Series Analysis — Watching a Glacier Change Over Years

Your notebook built a time-series of the Aletsch Glacier from 2017 to 2025. Here's the methodology:

### Step 1: Search for All Available Scenes

```python
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2017-01-01/2025-12-31",
    query={"eo:cloud_cover": {"lt": 20}}
)
items = sorted(list(search.items()), key=lambda x: x.datetime)
```

### Step 2: Select the Best Scene Per Year

For glacier analysis, you want **summer images** (July–September) because:
- Seasonal snow has melted, revealing the true glacier extent
- Longer daylight hours = better illumination
- The glacier is at its minimum annual extent → the "true" size of the glacier body

```python
for scene in items:
    year = scene.datetime.year
    month = scene.datetime.month
    if month not in [7, 8, 9]:        # Summer only
        continue
    cloud = scene.properties["eo:cloud_cover"]
    if year not in best_per_year or cloud < best_per_year[year].properties["eo:cloud_cover"]:
        best_per_year[year] = scene    # Keep the least-cloudy summer scene
```

### Step 3: Generate Consistent Crops

A reusable function was created to download, stack, and normalise an RGB crop for any scene:

```python
def get_rgb_crop(scene, glacier_geom):
    scene = planetary_computer.sign(scene)  # Re-sign URLs (they expire!)
    minx, miny, maxx, maxy = glacier_geom.bounds
    bands = {}
    for band in ["B04", "B03", "B02"]:
        with rasterio.open(scene.assets[band].href) as src:
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            bands[band] = src.read(1, window=window)
    rgb = np.dstack([bands["B04"], bands["B03"], bands["B02"]]).astype(np.float32)
    p2, p98 = np.percentile(rgb, (2, 98))
    return np.clip((rgb - p2) / (p98 - p2), 0, 1)
```

### Step 4: Plot the Grid

```python
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for ax, year in zip(axes, sorted(yearly_images.keys())):
    ax.imshow(yearly_images[year])
    ax.set_title(str(year))
    ax.axis("off")
```

This produces a visual record of how the glacier's snow cover, surface, and extent changes year over year.

---

## 17. Cloud-Optimised GeoTIFFs and STAC — Modern Satellite Data Access

### The Old Way (Before ~2020)

1. Go to USGS EarthExplorer or Copernicus Open Access Hub
2. Search for your area and date
3. Download a **massive ZIP file** (700MB–1.5GB per scene)
4. Extract it locally
5. Open individual bands with rasterio

### The New Way (What Your Notebook Uses)

1. Connect to a **STAC API** (SpatioTemporal Asset Catalog)
2. Search programmatically with bounding box, date, and cloud cover filters
3. Get **direct URLs** to each band file stored in the cloud
4. **Read only the pixels you need** via windowed reading — no download required!

```python
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=bbox,
    datetime="2024-07-01/2024-09-30",
    query={"eo:cloud_cover": {"lt": 20}},
)
```

The band files are **Cloud-Optimised GeoTIFFs (COGs)** — specially structured TIFFs that allow rasterio to read arbitrary rectangular subsets over HTTP without downloading the full file. This is why windowed reading works directly with URLs.

### URL Signing

Microsoft Planetary Computer URLs require authentication tokens (SAS tokens) that **expire after ~24 hours**. That's why the notebook calls `planetary_computer.sign(scene)` before each access — it refreshes the authentication tokens on the URLs.

---

## 18. Scene Classification Layer (SCL) — Automatic Pixel Classification

Your dataset includes `SCL.tif` files. SCL is a pre-computed classification layer included with every Sentinel-2 L2A product. Each pixel is assigned an integer class:

| Value | Class | Description |
|-------|-------|-------------|
| 0 | No Data | Missing or invalid pixel |
| 1 | Saturated/Defective | Sensor issue |
| 2 | Dark Area Pixels | Topographic shadow, dark surfaces |
| 3 | Cloud Shadow | Detected cloud shadow |
| 4 | Vegetation | Green, photosynthetically active |
| 5 | Bare Soil | Non-vegetated land |
| 6 | Water | Open water |
| 7 | Unclassified | Couldn't be classified |
| 8 | Cloud (Medium Probability) | Probably cloud |
| 9 | Cloud (High Probability) | Definitely cloud |
| 10 | Thin Cirrus | Semi-transparent high cloud |
| 11 | Snow/Ice | Detected snow or ice |

**How to use it for glacier masking:**
```python
with rasterio.open("SCL.tif") as src:
    scl = src.read(1)

# Create a glacier/snow mask
snow_mask = (scl == 11)  # Snow/Ice pixels
cloud_mask = (scl >= 8) & (scl <= 10)  # Cloud pixels to exclude

plt.imshow(scl, cmap="tab20", vmin=0, vmax=11)
plt.colorbar()
plt.title("Scene Classification Layer")
```

---

## 19. Common False Color Composite Recipes

Here are the most useful band combinations for different purposes. For each, you assign three bands to the R, G, B channels of your display:

| Composite Name | R ← | G ← | B ← | What You See |
|---------------|------|------|------|-------------|
| **True Color (Natural)** | B04 | B03 | B02 | What the human eye would see |
| **Color Infrared (CIR)** | B08 | B04 | B03 | Vegetation → bright red |
| **False Color Urban** | B12 | B11 | B04 | Urban areas → white/cyan |
| **Agriculture** | B11 | B08 | B02 | Crops → bright green |
| **Atmospheric Penetration** | B12 | B11 | B8A | See through thin clouds/haze |
| **Healthy Vegetation** | B08 | B11 | B02 | Healthy plants → bright green |
| **Land/Water** | B08 | B11 | B04 | Water → black, land → coloured |
| **Shortwave Infrared** | B12 | B08 | B04 | Snow → cyan/blue, clouds → white |
| **Vegetation Index** | B08 | B04 | B03 | Dense vegetation → bright red |
| **Geology** | B12 | B11 | B02 | Rock types → distinct colours |
| **Snow/Ice (for Glaciers!)** | B03 | B11 | B12 | Snow → bright, rock → dark, clouds → white |

### Glacier-Specific Recommendation

For glacier analysis, the **SWIR composite** (B12, B08, B04 mapped to R, G, B) is extremely useful:
```python
glacier_composite = np.dstack([
    normalise(B12),  # SWIR-2 → Red
    normalise(B08),  # NIR → Green
    normalise(B04)   # Red → Blue
])
# Snow/ice appears cyan/blue (low SWIR, high NIR)
# Clouds appear white (high everything)
# Bare rock appears orange/brown
# Water appears black
```

---

## 20. Summary and Key Takeaways

### What You Did in Your Notebook

| Phase | Data Type | What You Did | Libraries |
|-------|-----------|-------------|-----------|
| 1 | Vector | Loaded 617K glacier polygons, explored metadata, filtered by region | geopandas |
| 2 | API | Connected to Planetary Computer STAC, searched for Sentinel-2 scenes | pystac_client, planetary_computer |
| 3 | Raster | Downloaded B02/B03/B04, stacked as RGB, normalised with percentile stretch | rasterio, numpy, matplotlib |
| 4 | Metadata | Inspected CRS (EPSG:32632), resolution (10m), and tile size (10980×10980) | rasterio |
| 5 | Raster + Vector | Reprojected glacier polygon CRS, used windowed reading to crop bands | rasterio, geopandas |
| 6 | Raster | Loaded NIR (B08) and SWIR (B11, B12), plotted each as grayscale | rasterio, matplotlib |
| 7 | Metadata | Discovered resolution differences: 10m vs 20m across bands | rasterio |
| 8 | Time-Series | Built yearly summer RGB crops from 2017–2025, plotted as 3×3 grid | all of the above |
| 9 | Local Raster | Loaded pre-downloaded per-band TIFs, applied same RGB + stretch workflow | rasterio, numpy |

### Core Concepts

1. **Satellite images are NOT photos** — they are multi-band rasters where each band captures a different wavelength of the electromagnetic spectrum
2. **Bands exist because different wavelengths reveal different things** — SWIR separates snow from clouds, NIR highlights vegetation, thermal measures temperature
3. **Band numbers are NOT universal** — always check the satellite's documentation
4. **Raw pixel values are NOT display-ready** — you must normalise (preferably using percentile stretching)
5. **To display any band, map it to a colormap** — single band → grayscale; three bands → assign to R/G/B channels
6. **Unknown bands can be identified** by: reading metadata, checking resolution, plotting histograms, and comparing visual patterns to known spectral signatures
7. **CRS must match** before combining vector and raster data — use `.to_crs()` to reproject
8. **Windowed reading** lets you crop satellite images to your area of interest without downloading the full tile
9. **Spectral indices** (NDVI, NDSI, NDWI) are normalised band ratios that quantify specific surface properties
10. **Time-series** analysis compares the same location across multiple dates to track change — essential for glacier monitoring
