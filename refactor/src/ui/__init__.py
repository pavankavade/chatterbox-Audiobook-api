"""
UI Module

This module provides the Gradio web interface for the Chatterbox Audiobook Studio,
integrating with all the refactored modules to provide a seamless user experience.
"""

from .gradio_interface import ChatterboxGradioApp

__all__ = [
    'ChatterboxGradioApp'
] 