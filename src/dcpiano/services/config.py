import yaml
from pathlib import Path
from dcpiano.types.config import Config


class ConfigService:
    @staticmethod
    def load_config(config_path: Path = Path("config/default.yaml")) -> Config:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        return Config(**config_data)