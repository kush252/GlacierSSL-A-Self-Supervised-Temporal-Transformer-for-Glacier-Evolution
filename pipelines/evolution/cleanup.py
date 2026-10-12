import os
import shutil
from pathlib import Path

def clean_output():
    # The output directory is located in the same directory as this script
    output_dir = Path(__file__).parent / "output"
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"✅ Successfully deleted all previous logs and folders in: {output_dir}")
        print("Ready for a fresh run!")
    else:
        print(f"ℹ️ Output directory {output_dir} does not exist. Nothing to clean.")

if __name__ == "__main__":
    clean_output()
