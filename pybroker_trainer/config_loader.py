"""
Handles loading and managing strategy configurations from JSON files.
This separates strategy parameters from the core application logic, making
the system more modular and easier to manage, similar to professional platforms.
"""

import os
import json
import logging

from load_cfg import WORKING_DIRECTORY

logger = logging.getLogger(__name__)
CONFIG_DIR = os.path.join(WORKING_DIRECTORY, "strategy_configs")


def load_strategy_config(strategy_type: str, base_params: dict) -> dict:
    """
    Loads a strategy's JSON configuration file and merges it with base defaults.

    Args:
        strategy_type: The name of the strategy, corresponding to the JSON filename.
        base_params: A dictionary of default parameters to fall back on.

    Returns:
        A dictionary containing the final, merged strategy parameters.
    """
    config_path = os.path.join(CONFIG_DIR, f"{strategy_type}.json")
    final_params = base_params.copy()
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            specific_params = json.load(f)
            final_params.update(specific_params)
            logger.info(f"Loaded and merged config from {config_path}")
    return final_params