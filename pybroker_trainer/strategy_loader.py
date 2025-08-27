"""
A utility for dynamically discovering and loading strategy modules.

This module allows the framework to be truly plug-and-play. To add a new
strategy, simply create a new Python file in the `strategies` directory that
contains a class inheriting from `BaseStrategy`. This loader will automatically
find and register it.
"""

import os
import importlib
import inspect
import pkgutil
from skopt.space import Real, Integer

from pybroker_trainer.strategy_sdk import BaseStrategy

def to_camel_case(snake_str: str) -> str:
    """Converts snake_case_string to CamelCaseString."""
    return "".join(x.capitalize() for x in snake_str.lower().split("_"))

def get_strategy_class_map():
    """
    Dynamically discovers and returns a map of all available strategy classes.
    It scans the 'strategies' directory for modules containing BaseStrategy subclasses.
    """
    strategy_map = {}
    strategies_path = os.path.join(os.path.dirname(__file__), '..', 'strategies')
    
    for _, name, _ in pkgutil.iter_modules([strategies_path]):
        try:
            module = importlib.import_module(f'strategies.{name}')
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                    # The key is the snake_case module name, e.g., 'uptrend_pullback'
                    strategy_map[name] = obj
        except Exception as e:
            print(f"Could not load strategy from {name}: {e}")
            
    return strategy_map

STRATEGY_CLASS_MAP = get_strategy_class_map()

def load_strategy_class(strategy_name: str):
    """Loads a strategy class by its snake_case name."""
    return STRATEGY_CLASS_MAP.get(strategy_name)

def get_strategy_defaults(strategy_class) -> dict:
    """
    Extracts the default parameter values from a strategy class's
    `define_parameters` method.
    """
    if not strategy_class or not hasattr(strategy_class, 'define_parameters'):
        return {}
    
    params = strategy_class.define_parameters()
    return {name: p_info['default'] for name, p_info in params.items()}

def get_strategy_tuning_space(strategy_class) -> list:
    """
    Constructs the search space for scikit-optimize from a strategy's
    `define_parameters` method.
    """
    if not strategy_class or not hasattr(strategy_class, 'define_parameters'):
        return []
        
    params = strategy_class.define_parameters()
    search_space = []
    for name, p_info in params.items():
        if 'tuning_range' in p_info:
            if p_info['type'] == 'int':
                search_space.append(Integer(*p_info['tuning_range'], name=name))
            elif p_info['type'] == 'float':
                search_space.append(Real(*p_info['tuning_range'], name=name))
    return search_space