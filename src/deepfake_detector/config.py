"""
Configuration management for the Deepfake Detection Agent.

This module provides utilities for loading configuration from:
- YAML files
- JSON files
- Environment variables
- Default values

The configuration system follows a layered approach:
1. Default values (hardcoded)
2. Configuration file (config.yaml)
3. Environment variables (override)
4. Runtime parameters (highest priority)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import ConfigurationError
from .models import AnalysisConfig


logger = logging.getLogger(__name__)


def load_config(
    config_path: str | Path | None = None,
    env_prefix: str = "DEEPFAKE_"
) -> AnalysisConfig:
    """
    Load configuration from file and environment variables.
    
    Configuration sources are merged in order of priority (lowest to highest):
    1. Default values
    2. Configuration file (if provided)
    3. Environment variables
    
    Args:
        config_path: Path to YAML or JSON configuration file
        env_prefix: Prefix for environment variables (default: DEEPFAKE_)
    
    Returns:
        AnalysisConfig with merged configuration
    
    Raises:
        ConfigurationError: If configuration file is invalid
    
    Example:
        >>> config = load_config("config/config.yaml")
        >>> detector = DeepfakeDetector(config=config)
    """
    # Start with defaults
    config_dict: dict[str, Any] = {}
    
    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {path}",
                config_key="config_path"
            )
        
        try:
            with open(path) as f:
                if path.suffix in (".yaml", ".yml"):
                    file_config = yaml.safe_load(f)
                elif path.suffix == ".json":
                    import json
                    file_config = json.load(f)
                else:
                    raise ConfigurationError(
                        f"Unsupported config file format: {path.suffix}",
                        config_key="config_path"
                    )
                
                if file_config:
                    config_dict = _flatten_config(file_config)
                    
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                config_key="config_path"
            ) from e
    
    # Override with environment variables
    env_config = _load_env_config(env_prefix)
    config_dict.update(env_config)
    
    # Build AnalysisConfig
    return _build_config(config_dict)


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested configuration dictionary."""
    result: dict[str, Any] = {}
    
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        
        if isinstance(value, dict):
            result.update(_flatten_config(value, f"{full_key}_"))
        else:
            result[full_key] = value
    
    return result


def _load_env_config(prefix: str) -> dict[str, Any]:
    """Load configuration from environment variables."""
    config: dict[str, Any] = {}
    
    # Define expected environment variables and their types
    env_mappings = {
        "THRESHOLD": ("deepfake_threshold", float),
        "UNCERTAIN_THRESHOLD": ("uncertain_threshold", float),
        "MAX_FRAMES": ("max_frames", int),
        "SAMPLE_RATE": ("sample_rate", int),
        "ENABLE_GPU": ("enable_gpu", bool),
        "ENABLE_AUDIO": ("enable_audio", bool),
        "LOG_LEVEL": ("log_level", str),
        "WEIGHT_VISUAL_ARTIFACTS": ("visual_artifacts_weight", float),
        "WEIGHT_TEMPORAL": ("temporal_weight", float),
        "WEIGHT_PHYSIOLOGICAL": ("physiological_weight", float),
        "WEIGHT_FREQUENCY": ("frequency_weight", float),
    }
    
    for env_suffix, (config_key, value_type) in env_mappings.items():
        env_key = f"{prefix}{env_suffix}"
        value = os.environ.get(env_key)
        
        if value is not None:
            try:
                if value_type == bool:
                    config[config_key] = value.lower() in ("true", "1", "yes")
                else:
                    config[config_key] = value_type(value)
            except ValueError as e:
                logger.warning(
                    f"Invalid value for {env_key}: {value} (expected {value_type.__name__})"
                )
    
    return config


def _build_config(config_dict: dict[str, Any]) -> AnalysisConfig:
    """Build AnalysisConfig from dictionary."""
    # Extract known fields
    kwargs: dict[str, Any] = {}
    
    field_mappings = {
        "max_frames": "max_frames",
        "analysis_max_frames": "max_frames",
        "sample_rate": "sample_rate",
        "analysis_sample_rate": "sample_rate",
        "deepfake_threshold": "deepfake_threshold",
        "thresholds_deepfake": "deepfake_threshold",
        "uncertain_threshold": "uncertain_threshold",
        "thresholds_uncertain": "uncertain_threshold",
        "min_face_size": "min_face_size",
        "analysis_min_face_size": "min_face_size",
    }
    
    for config_key, field_name in field_mappings.items():
        if config_key in config_dict:
            kwargs[field_name] = config_dict[config_key]
    
    # Handle skill weights
    weights = {}
    weight_prefix = "weights_"
    for key, value in config_dict.items():
        if key.startswith(weight_prefix):
            skill_name = key[len(weight_prefix):]
            weights[skill_name] = value
    
    if weights:
        kwargs["skill_weights"] = weights
    
    return AnalysisConfig(**kwargs)


def configure_logging(
    level: str = "INFO",
    format: str = "standard",
    output: str | Path | None = None,
    json_format: bool = False
) -> None:
    """
    Configure logging for the deepfake detector.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format ("standard" or "detailed")
        output: Optional file path for log output
        json_format: Whether to output logs as JSON
    
    Example:
        >>> configure_logging(level="DEBUG", output="analysis.log")
    """
    log_format = {
        "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    }.get(format, format)
    
    handlers: list[logging.Handler] = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if output specified
    if output:
        file_handler = logging.FileHandler(output)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger for deepfake_detector
    root_logger = logging.getLogger("deepfake_detector")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    logger.info(f"Logging configured: level={level}, output={output or 'console'}")


@dataclass
class RuntimeConfig:
    """
    Runtime configuration container.
    
    This class holds all configuration values and provides
    convenient access methods.
    """
    
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    log_level: str = "INFO"
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    
    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        """Load runtime configuration from file."""
        analysis = load_config(path)
        return cls(analysis=analysis)
    
    def validate(self) -> list[str]:
        """
        Validate configuration values.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []
        
        if self.analysis.max_frames < 10:
            errors.append("max_frames must be at least 10")
        
        if self.analysis.sample_rate < 1:
            errors.append("sample_rate must be at least 1")
        
        if not (0 < self.analysis.deepfake_threshold <= 1):
            errors.append("deepfake_threshold must be between 0 and 1")
        
        if not (0 < self.analysis.uncertain_threshold <= 1):
            errors.append("uncertain_threshold must be between 0 and 1")
        
        if self.analysis.uncertain_threshold >= self.analysis.deepfake_threshold:
            errors.append("uncertain_threshold must be less than deepfake_threshold")
        
        return errors

