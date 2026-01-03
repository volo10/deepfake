"""
Unit tests for configuration management.

Tests cover:
- Configuration loading from files
- Environment variable handling
- Configuration validation
- Default value handling
"""

import pytest
import os
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepfake_detector.config import (
    load_config,
    configure_logging,
    RuntimeConfig,
    _flatten_config,
    _load_env_config,
)
from deepfake_detector.models import AnalysisConfig
from deepfake_detector.exceptions import ConfigurationError


class TestLoadConfig:
    """Tests for load_config function."""
    
    def test_load_default_config(self):
        """Test loading configuration with defaults only."""
        config = load_config()
        assert isinstance(config, AnalysisConfig)
        assert config.max_frames > 0
        assert config.sample_rate >= 1
    
    def test_load_yaml_config(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
        analysis:
          max_frames: 500
          sample_rate: 3
        thresholds:
          deepfake: 0.45
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                config = load_config(f.name)
                assert config.max_frames == 500
                assert config.sample_rate == 3
            finally:
                os.unlink(f.name)
    
    def test_load_json_config(self):
        """Test loading configuration from JSON file."""
        import json
        json_content = {
            "analysis": {
                "max_frames": 400,
                "sample_rate": 2
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_content, f)
            f.flush()
            
            try:
                config = load_config(f.name)
                assert config.max_frames == 400
                assert config.sample_rate == 2
            finally:
                os.unlink(f.name)
    
    def test_config_file_not_found(self):
        """Test error handling for missing config file."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config("/nonexistent/path/config.yaml")
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_invalid_yaml_config(self):
        """Test error handling for invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: {")
            f.flush()
            
            try:
                with pytest.raises(ConfigurationError):
                    load_config(f.name)
            finally:
                os.unlink(f.name)


class TestFlattenConfig:
    """Tests for configuration flattening."""
    
    def test_flatten_simple_dict(self):
        """Test flattening a simple dictionary."""
        config = {"key1": "value1", "key2": "value2"}
        flattened = _flatten_config(config)
        
        assert flattened["key1"] == "value1"
        assert flattened["key2"] == "value2"
    
    def test_flatten_nested_dict(self):
        """Test flattening a nested dictionary."""
        config = {
            "analysis": {
                "max_frames": 300,
                "sample_rate": 2
            },
            "thresholds": {
                "deepfake": 0.35
            }
        }
        flattened = _flatten_config(config)
        
        assert flattened["analysis_max_frames"] == 300
        assert flattened["analysis_sample_rate"] == 2
        assert flattened["thresholds_deepfake"] == 0.35
    
    def test_flatten_deeply_nested(self):
        """Test flattening deeply nested configuration."""
        config = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }
        flattened = _flatten_config(config)
        
        assert flattened["level1_level2_level3"] == "value"


class TestEnvConfig:
    """Tests for environment variable configuration."""
    
    def test_load_env_threshold(self):
        """Test loading threshold from environment."""
        os.environ["DEEPFAKE_THRESHOLD"] = "0.5"
        
        try:
            env_config = _load_env_config("DEEPFAKE_")
            assert env_config.get("deepfake_threshold") == 0.5
        finally:
            del os.environ["DEEPFAKE_THRESHOLD"]
    
    def test_load_env_max_frames(self):
        """Test loading max_frames from environment."""
        os.environ["DEEPFAKE_MAX_FRAMES"] = "400"
        
        try:
            env_config = _load_env_config("DEEPFAKE_")
            assert env_config.get("max_frames") == 400
        finally:
            del os.environ["DEEPFAKE_MAX_FRAMES"]
    
    def test_load_env_boolean(self):
        """Test loading boolean values from environment."""
        os.environ["DEEPFAKE_ENABLE_GPU"] = "true"
        
        try:
            env_config = _load_env_config("DEEPFAKE_")
            assert env_config.get("enable_gpu") is True
        finally:
            del os.environ["DEEPFAKE_ENABLE_GPU"]
    
    def test_load_env_boolean_false(self):
        """Test loading false boolean value."""
        os.environ["DEEPFAKE_ENABLE_GPU"] = "false"
        
        try:
            env_config = _load_env_config("DEEPFAKE_")
            assert env_config.get("enable_gpu") is False
        finally:
            del os.environ["DEEPFAKE_ENABLE_GPU"]
    
    def test_env_overrides_file(self):
        """Test that environment variables override file config."""
        yaml_content = """
        analysis:
          max_frames: 300
        """
        
        os.environ["DEEPFAKE_MAX_FRAMES"] = "600"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                config = load_config(f.name)
                # Environment should override file
                assert config.max_frames == 600
            finally:
                del os.environ["DEEPFAKE_MAX_FRAMES"]
                os.unlink(f.name)


class TestConfigureLogging:
    """Tests for logging configuration."""
    
    def test_configure_logging_default(self):
        """Test default logging configuration."""
        configure_logging()
        
        import logging
        logger = logging.getLogger("deepfake_detector")
        assert logger.level == logging.INFO
    
    def test_configure_logging_debug(self):
        """Test debug level configuration."""
        configure_logging(level="DEBUG")
        
        import logging
        logger = logging.getLogger("deepfake_detector")
        assert logger.level == logging.DEBUG
    
    def test_configure_logging_with_file(self):
        """Test logging to file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            try:
                configure_logging(level="INFO", output=f.name)
                
                import logging
                logger = logging.getLogger("deepfake_detector")
                
                # Check that file handler was added
                file_handlers = [
                    h for h in logger.handlers 
                    if isinstance(h, logging.FileHandler)
                ]
                assert len(file_handlers) > 0
            finally:
                os.unlink(f.name)


class TestRuntimeConfig:
    """Tests for RuntimeConfig class."""
    
    def test_runtime_config_defaults(self):
        """Test RuntimeConfig default values."""
        rc = RuntimeConfig()
        assert isinstance(rc.analysis, AnalysisConfig)
        assert rc.log_level == "INFO"
    
    def test_runtime_config_validate_success(self):
        """Test validation with valid config."""
        rc = RuntimeConfig(
            analysis=AnalysisConfig(
                max_frames=200,
                sample_rate=2,
                deepfake_threshold=0.5,
                uncertain_threshold=0.2
            )
        )
        errors = rc.validate()
        assert len(errors) == 0
    
    def test_runtime_config_validate_max_frames(self):
        """Test validation catches low max_frames."""
        rc = RuntimeConfig(
            analysis=AnalysisConfig(max_frames=5)
        )
        errors = rc.validate()
        assert any("max_frames" in e for e in errors)
    
    def test_runtime_config_validate_sample_rate(self):
        """Test validation catches zero sample_rate."""
        rc = RuntimeConfig(
            analysis=AnalysisConfig(sample_rate=0)
        )
        errors = rc.validate()
        assert any("sample_rate" in e for e in errors)
    
    def test_runtime_config_validate_thresholds(self):
        """Test validation catches inverted thresholds."""
        rc = RuntimeConfig(
            analysis=AnalysisConfig(
                deepfake_threshold=0.3,
                uncertain_threshold=0.5  # Should be less than deepfake
            )
        )
        errors = rc.validate()
        assert any("threshold" in e.lower() for e in errors)


class TestConfigEdgeCases:
    """Tests for configuration edge cases."""
    
    def test_empty_yaml_file(self):
        """Test handling of empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            
            try:
                config = load_config(f.name)
                # Should return defaults
                assert isinstance(config, AnalysisConfig)
            finally:
                os.unlink(f.name)
    
    def test_unsupported_file_format(self):
        """Test error handling for unsupported file format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            f.flush()
            
            try:
                with pytest.raises(ConfigurationError) as exc_info:
                    load_config(f.name)
                assert "unsupported" in str(exc_info.value).lower()
            finally:
                os.unlink(f.name)
    
    def test_invalid_env_value(self):
        """Test handling of invalid environment variable value."""
        os.environ["DEEPFAKE_MAX_FRAMES"] = "not_a_number"
        
        try:
            # Should not crash, just skip invalid value
            env_config = _load_env_config("DEEPFAKE_")
            assert "max_frames" not in env_config
        finally:
            del os.environ["DEEPFAKE_MAX_FRAMES"]

