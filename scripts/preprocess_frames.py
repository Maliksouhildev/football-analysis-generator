import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.ingestion.preprocess_frames import main

if __name__ == "__main__":
    main()
