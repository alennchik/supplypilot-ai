from src.config import load_config


def test_project_name():
    config = load_config()
    assert config["project"]["name"] == "SupplyPilot AI"


def test_service_level():
    config = load_config()
    level = config["inventory"]["service_level"]

    assert 0 < level < 1


def test_data_dimensions():
    config = load_config()

    assert config["data"]["num_products"] == 20
    assert config["data"]["num_warehouses"] == 3
