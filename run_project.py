"""
Single entry point for the entire Network Traffic Preprocessing and Class-Imbalance Study.

PyCharm:
1. Open this repository as a project.
2. Create/select a Python 3.10+ virtual environment.
3. Install: pip install -r requirements.txt
4. Put your IDS CSV file(s), including benign/normal and attack traffic, in data/raw/.
5. Run this file.
"""
from pathlib import Path
import yaml
from network_imbalance.config import Config
from network_imbalance.experiment import run_experiment

def main():
    cfg_path=Path("configs/config.yaml")
    values={}
    if cfg_path.exists():
        with open(cfg_path,encoding="utf-8") as f:
            values=yaml.safe_load(f) or {}
    config=Config(**values)
    print("="*72)
    print("Network Traffic Preprocessing and Class-Imbalance Study")
    print("="*72)
    print(f"Dataset: {config.data_path}")
    print("Running strategies:", ", ".join(config.strategies))
    results=run_experiment(config)
    print("\nExperiment complete. Results:")
    print(results.round(4).to_string(index=False))
    print(f"\nOutputs saved to: {config.output_dir}/")
    print("Main table: outputs/tables/experiment_results.csv")

if __name__=="__main__":
    main()
