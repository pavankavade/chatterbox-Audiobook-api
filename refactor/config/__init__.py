"""
Configuration module for Chatterbox Audiobook Studio.

This module provides centralized configuration management for the refactored
audiobook studio system.
"""

from .settings import *
from .device_config import *

__all__ = ['get_system_config', 'REFACTORED_PORT', 'get_device_configuration'] 