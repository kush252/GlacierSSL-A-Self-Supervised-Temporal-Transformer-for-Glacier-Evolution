import os
import json
from pathlib import Path

def validate_outputs(output_dir="output"):
    """
    Validates that:
    - No glacier is classified twice (exists in multiple folders)
    - Metadata is complete
    - Confidence values are valid
    """
    base_path = Path(__file__).parent.parent / output_dir
    if not base_path.exists():
        print("No output directory found.")
        return

    seen_glaciers = set()
    duplicates = []
    invalid_confidences = []
    missing_metadata = []

    for class_folder in base_path.iterdir():
        if class_folder.is_dir():
            for glacier_folder in class_folder.iterdir():
                if glacier_folder.is_dir():
                    g_id = glacier_folder.name
                    if g_id in seen_glaciers:
                        duplicates.append(g_id)
                    seen_glaciers.add(g_id)

                    meta_file = glacier_folder / "metadata.json"
                    if not meta_file.exists():
                        missing_metadata.append(g_id)
                    else:
                        try:
                            with open(meta_file, "r") as f:
                                data = json.load(f)
                                conf = data.get("Confidence")
                                if conf not in ["High", "Medium", "Low", "None"]:
                                    invalid_confidences.append(f"{g_id}: {conf}")
                        except Exception as e:
                            missing_metadata.append(f"{g_id} (error: {str(e)})")

    print(f"Validation complete. Processed {len(seen_glaciers)} glaciers.")
    if duplicates:
        print(f"WARNING: Duplicates found! {duplicates}")
    if missing_metadata:
        print(f"WARNING: Missing/invalid metadata for! {missing_metadata}")
    if invalid_confidences:
        print(f"WARNING: Invalid confidences found! {invalid_confidences}")

if __name__ == "__main__":
    validate_outputs()
