"""
Configuration Management Module

This module centralizes access to the application's configuration.
By importing 'config' from this module, other parts of the application
can access all settings and paths in a consistent way.
"""

from .settings import config, AppConfig

__all__ = [
    'config',
    'AppConfig'
] 