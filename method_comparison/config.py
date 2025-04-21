from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
import os

@dataclass
class MethodConfig:
    """Configuration for a PEFT method."""
    name: str
    parameters: Dict[str, Any]
    memory_requirements: Dict[str, float]  # e.g., {"GPU": 8.0} for 8GB
    training_time: Optional[float] = None  # in hours
    hardware_requirements: Dict[str, str] = None  # e.g., {"GPU": "A100"}
    best_use_cases: list = None
    limitations: list = None

class MethodRegistry:
    """Registry for managing PEFT method configurations."""
    
    def __init__(self):
        self.methods: Dict[str, MethodConfig] = {}
        self.config_file = "method_configs.yaml"
    
    def add_new_method(self, method_name: str, config: MethodConfig) -> None:
        """Add a new method configuration to the registry.
        
        Args:
            method_name: Name of the PEFT method
            config: MethodConfig object containing the method's configuration
        """
        if method_name in self.methods:
            raise ValueError(f"Method {method_name} already exists in registry")
        
        self.methods[method_name] = config
        self._save_config()
    
    def get_method(self, method_name: str) -> MethodConfig:
        """Retrieve a method configuration by name.
        
        Args:
            method_name: Name of the PEFT method
            
        Returns:
            MethodConfig object for the specified method
        """
        if method_name not in self.methods:
            raise ValueError(f"Method {method_name} not found in registry")
        return self.methods[method_name]
    
    def _save_config(self) -> None:
        """Save the current configuration to a YAML file."""
        config_dict = {
            name: {
                "parameters": config.parameters,
                "memory_requirements": config.memory_requirements,
                "training_time": config.training_time,
                "hardware_requirements": config.hardware_requirements,
                "best_use_cases": config.best_use_cases,
                "limitations": config.limitations
            }
            for name, config in self.methods.items()
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(config_dict, f)
    
    def load_config(self) -> None:
        """Load configuration from the YAML file."""
        if not os.path.exists(self.config_file):
            return
            
        with open(self.config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        for name, config in config_dict.items():
            self.methods[name] = MethodConfig(
                name=name,
                parameters=config['parameters'],
                memory_requirements=config['memory_requirements'],
                training_time=config.get('training_time'),
                hardware_requirements=config.get('hardware_requirements'),
                best_use_cases=config.get('best_use_cases'),
                limitations=config.get('limitations')
            )

# Create a global registry instance
method_registry = MethodRegistry() 