from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config():
    """Load application settings from config.yaml."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    config = load_config()

    print("Project:", config["project"]["name"])
    print("Products:", config["data"]["num_products"])
    print("Warehouses:", config["data"]["num_warehouses"])
    print("Service level:", config["inventory"]["service_level"])
