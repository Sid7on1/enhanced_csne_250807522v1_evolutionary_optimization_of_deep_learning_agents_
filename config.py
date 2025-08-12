import logging
import os
import yaml
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
CONFIG_FILE = 'config.yaml'
DEFAULT_CONFIG = {
    'agent': {
        'name': 'default_agent',
        'type': 'lstm',
        'params': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }
    },
    'environment': {
        'name': 'default_environment',
        'type': 'sparrow_mahjong',
        'params': {
            'board_size': 15,
            'num_players': 4
        }
    }
}

# Define enums
class AgentType(Enum):
    LSTM = 'lstm'
    CNN = 'cnn'

class EnvironmentType(Enum):
    SPARROW_MAHJONG = 'sparrow_mahjong'

# Define dataclasses
@dataclass
class AgentConfig:
    name: str
    type: AgentType
    params: Dict[str, float]

@dataclass
class EnvironmentConfig:
    name: str
    type: EnvironmentType
    params: Dict[str, int]

@dataclass
class Config:
    agent: AgentConfig
    environment: EnvironmentConfig

# Define functions
def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file or default values."""
    if config_file is None:
        config_file = CONFIG_FILE
    if not os.path.exists(config_file):
        logger.warning(f'Config file {config_file} not found. Using default values.')
        return Config(**DEFAULT_CONFIG)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return Config(**config)

def save_config(config: Config, config_file: Optional[str] = None) -> None:
    """Save configuration to file."""
    if config_file is None:
        config_file = CONFIG_FILE
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def validate_config(config: Config) -> None:
    """Validate configuration values."""
    if config.agent.type not in [agent_type.value for agent_type in AgentType]:
        raise ValueError(f'Invalid agent type: {config.agent.type}')
    if config.environment.type not in [env_type.value for env_type in EnvironmentType]:
        raise ValueError(f'Invalid environment type: {config.environment.type}')

def get_config() -> Config:
    """Get configuration instance."""
    config_file = CONFIG_FILE
    if not os.path.exists(config_file):
        logger.warning(f'Config file {config_file} not found. Using default values.')
        return Config(**DEFAULT_CONFIG)
    config = load_config(config_file)
    validate_config(config)
    return config

# Define main class
class ConfigManager:
    def __init__(self, config_file: Optional[str] = None):
        self.config = get_config()
        self.config_file = config_file

    def save(self) -> None:
        save_config(self.config, self.config_file)

    def update(self, agent: AgentConfig = None, environment: EnvironmentConfig = None) -> None:
        if agent is not None:
            self.config.agent = agent
        if environment is not None:
            self.config.environment = environment
        self.save()

# Create instance
config_manager = ConfigManager()

# Example usage
if __name__ == '__main__':
    config = config_manager.config
    print(config.agent.name)
    print(config.environment.name)
    config_manager.update(agent=AgentConfig(name='new_agent', type=AgentType.LSTM, params={'learning_rate': 0.01}))
    config_manager.save()