# Glacier Evolution Annotation Pipeline

This directory contains the deterministic, rule-based pipeline for classifying glacier evolution. The pipeline implements the strict rules, thresholds, and hierarchies defined in the Glacier Evolution Annotation Guide. 

It takes time-series data of glaciers (such as Sentinel-2 imagery metrics and GLIMS polygons) and automatically classifies them into one of five evolutionary states: **Stable, Retreat, Advance, Fragmentation, or Disappearance**, along with a confidence score.

---

## 🏗️ Pipeline Architecture & Data Flow

The pipeline is designed to be highly modular. The execution flows sequentially through several distinct phases:

1. **Initialization & Configuration**: Thresholds are loaded from a configuration file.
2. **Data Loading (`dataset_loader`)**: Glacier data spanning multiple epochs is loaded. Currently implemented as a stub, this phase produces `GlacierState` objects populated with `EpochData`.
3. **Feature Extraction**: (Placeholder for future implementation) Computes raw metrics like area, perimeter, and debris fractions from satellite data.
4. **Temporal Analysis (`temporal_analysis`)**: Computes time spans, max gaps, and Ordinary Least Squares (OLS) regression on the area over time to determine statistical trends (slope and p-value).
5. **Rule Engine (`rule_engine`)**: Evaluates the precomputed features against strict, isolated boolean rules derived from the annotation guide.
6. **Classification Hierarchy (`classification`)**: Applies a strict precedence order (Rule 8-A) to assign the final evolutionary class based on which rules were triggered.
7. **Confidence Scoring (`confidence`)**: Assigns a High, Medium, or Low confidence score based on data quality, debris cover, and observation gaps.
8. **Export & Logging (`export`, `logging`)**: Saves the results into class-specific directories with detailed `metadata.json` files and logs a summary to a central CSV file.
9. **Validation (`export.validation`)**: A post-processing check to ensure deterministic behavior (no duplicates, valid confidence scores, and complete metadata).

---

## 📂 Directory Structure & File Explanations

### 1. The Orchestrator
* **`main.py`**
  * **Function**: The entry point of the pipeline.
  * **Working**: It initializes the `DatasetLoader` and `OutputFormatter`. It loops through every `GlacierState` yielded by the loader, sequentially calling the temporal analysis, classification hierarchy, export, and logging functions. Finally, it triggers the validation script.

### 2. Configuration (`config/`)
* **`thresholds.yaml`**
  * **Function**: Contains all the hardcoded constants and thresholds required by the rules (e.g., Minimum Mapping Unit of 0.02 km², minimum epochs = 4, p-value threshold = 0.05). Keeping these in a YAML file ensures the pipeline is configurable without touching the code.
* **`config.py`**
  * **Function**: A Python wrapper around the YAML file.
  * **Working**: Defines the `Config` class which reads the YAML file on initialization and exposes the thresholds as Python properties. It instantiates a singleton `config` object used by the rest of the pipeline.

### 3. Utilities & Data Structures (`utils/`)
* **`structures.py`**
  * **Function**: Defines the core data schemas that flow through the pipeline.
  * **Working**: 
    * `EpochData`: A dataclass storing metrics for a single point in time (date, area, perimeter, debris fraction, terminus displacement, bridge widths).
    * `GlacierState`: A dataclass storing the entire sequence for a single glacier. It holds a list of `EpochData`, precomputed temporal features (OLS slope, p-value), and the final assigned class, confidence, and triggered rules.

### 4. Data Loading (`dataset_loader/`)
* **`loader.py`**
  * **Function**: Interface for fetching glacier data.
  * **Working**: Currently contains a stub `DatasetLoader` class with a `load_glaciers()` generator function. In a production environment, this function would orchestrate the downloading/reading of Sentinel-2 images and GLIMS shapefiles, run the feature extraction, and yield populated `GlacierState` objects.

### 5. Temporal Analysis (`temporal_analysis/`)
* **`trend_ols.py`**
  * **Function**: Calculates statistical trends over time.
  * **Working**: Contains the `compute_temporal_features(state)` function. It extracts the dates and areas from the `GlacierState`'s epochs, calculates the max temporal gap between observations, and uses `scipy.stats.linregress` to compute the OLS regression slope (`beta_hat`) and its statistical significance (`p_value`). These are saved back into the `GlacierState`.

### 6. The Rule Engine (`rule_engine/`)
* **`rules.py`**
  * **Function**: The brain of the pipeline. It translates the textual rules from the Annotation Guide into isolated boolean Python functions.
  * **Working**:
    * `rule_2_3_A(state)`: Checks if the minimum temporal support (>= 4 epochs, >= 8 years) is met.
    * `rule_2_4_B(state)`: Checks if the maximum observed area is below the Minimum Mapping Unit (MMU).
    * `rule_2_4_C_exceeded(state, direction)`: Checks if the terminus displacement exceeds the noise threshold in a specific direction (taking debris cover into account).
    * `rule_5_5_B_fragmented(state)`: Checks if an ice bridge width has reached 0 for the required persistence duration.
    * `rule_7_4_A(state)`: Evaluates the conditions for Disappearance (recent epochs below MMU following a negative trend).
    * `rule_3_4_A(state)`: Evaluates the conditions for Stable (non-significant trend, low terminus displacement, no fragmentation).
    * `rule_4_4_A(state)`: Evaluates the conditions for Retreat (significant negative trend OR significant up-valley terminus displacement).
    * `rule_6_4_A(state)`: Evaluates the conditions for Advance (significant positive trend OR down-valley displacement, excluding surges).
    * `rule_6_5_A(state)`: Evaluates the conditions for a Glacier Surge (sudden advance with visible surface disturbance).

### 7. Classification Hierarchy (`classification/`)
* **`hierarchy.py`**
  * **Function**: Applies the rule engine functions in a strict precedence order.
  * **Working**: Contains `classify_glacier(state)`. It enforces Rule 8-A of the guide. It checks the rules in this exact order: **Disappearance > Fragmentation > Retreat > Advance > Stable**. The first rule that returns `True` determines the `assigned_class`. It then calls the confidence scorer and logs any relevant sub-flags (like `insufficient_size_mmu`).

### 8. Confidence Scoring (`confidence/`)
* **`scorer.py`**
  * **Function**: Evaluates the reliability of the classification.
  * **Working**: Contains `score_confidence(state)`. It checks for conditions that downgrade confidence, such as heavy debris cover, sub-MMU sizes, or large gaps between observation epochs (> 5 years). It returns "High", "Medium", or "Low".

### 9. Export & Logging (`export/` and `logging/`)
* **`export/output_formatter.py`**
  * **Function**: Saves the classification results to disk.
  * **Working**: The `OutputFormatter` creates directories for each evolutionary class (e.g., `output/Retreat/`). Inside, it creates a folder for the specific glacier and writes a `metadata.json` file containing all the decision reasoning, triggered rules, and computed metrics.
* **`export/validation.py`**
  * **Function**: Post-processing sanity check.
  * **Working**: Contains `validate_outputs()`. It scans the `output/` directory to ensure that no glacier ID appears in more than one class folder, that all `metadata.json` files are readable and complete, and that confidence values are valid.
* **`logging/csv_logger.py`**
  * **Function**: Maintains a tabular record of all classifications.
  * **Working**: The `CSVLogger` initializes a `classification_log.csv` file. The `log_glacier(state)` function appends a row for each processed glacier, recording its ID, class, confidence, area trend, max fragments, and reasoning.

---

## 🚀 How to Run

To run the pipeline with the current stubs:
```bash
python main.py
```
This will execute the pipeline using the mock data provided in the `DatasetLoader` and generate an `output/` folder containing the results and logs.
