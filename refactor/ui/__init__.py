"""
User Interface module for Chatterbox Audiobook Studio.

This module provides enhanced Gradio interface components, real-time monitoring,
advanced settings panels, and workflow optimization for the refactored audiobook
studio system with integrated audio processing capabilities.
"""

from .enhanced_interface import *
from .monitoring_dashboard import *
from .settings_panels import *
from .workflow_optimizer import *
from .audio_integration import *

__all__ = [
    'EnhancedGradioInterface', 'MonitoringDashboard', 'SettingsPanel', 
    'WorkflowOptimizer', 'AudioIntegration', 'create_enhanced_interface',
    'setup_real_time_monitoring', 'create_audio_processing_ui'
] 