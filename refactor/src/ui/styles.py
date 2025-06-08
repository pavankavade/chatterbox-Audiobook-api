"""
Centralized CSS Styling System for Chatterbox Audiobook Studio

This module provides centralized CSS management for all UI components,
ensuring consistent styling across the entire application and eliminating
duplication of CSS code.
"""

from typing import Dict, List, Optional
from pathlib import Path

class ChatterboxStyles:
    """
    Centralized CSS styling system for all Chatterbox UI components.
    
    This class provides a single source of truth for all CSS styling,
    allowing for consistent design across all interfaces and easy
    maintenance of the visual design system.
    """
    
    def __init__(self):
        """Initialize the styling system with all CSS definitions."""
        self._load_base_styles()
        self._load_component_styles()
        self._load_theme_variants()
    
    def _load_base_styles(self) -> None:
        """Load base CSS styles used across all components."""
        self.base_styles = """
        /* ==================================================================== */
        /* CHATTERBOX AUDIOBOOK STUDIO - BASE STYLES */
        /* ==================================================================== */
        
        /* Global Typography */
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
        }
        
        /* Header Styles */
        .voice-library-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .voice-library-header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }
        
        .voice-library-header p {
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        
        /* Section Headers */
        .section-header {
            border-bottom: 2px solid #007bff;
            padding-bottom: 5px;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
            color: #007bff;
        }
        
        /* Layout Components */
        .compact-row {
            gap: 10px !important;
        }
        
        /* Interactive Elements */
        .gradio-button {
            transition: all 0.3s ease;
        }
        
        .gradio-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        """
    
    def _load_component_styles(self) -> None:
        """Load component-specific CSS styles."""
        self.component_styles = """
        /* ==================================================================== */
        /* COMPONENT STYLES */
        /* ==================================================================== */
        
        /* Information Boxes */
        .instruction-box {
            background: #3A3939;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }
        
        .instruction-box h4 {
            margin-top: 0;
            color: #007bff;
        }
        
        /* Status Indicators */
        .voice-status {
            padding: 10px;
            border-radius: 5px;
            background: #e3f2fd;
            border: 1px solid #2196f3;
            color: #1976d2;
            margin: 10px 0;
        }
        
        .config-status {
            padding: 8px;
            background: #f1f8e9;
            border: 1px solid #8bc34a;
            border-radius: 4px;
            color: #33691e;
            margin: 5px 0;
            font-size: 0.9em;
        }
        
        .config-status.success {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .config-status.error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .status-panel {
            background: #3A3939;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        /* Status Message Types */
        .status-box {
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .status-success {
            background: #3A3939;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        
        .status-warning {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        
        .status-error {
            background: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        
        .status-info {
            background: #cce7ff;
            border: 1px solid #b3d9ff;
            color: #004085;
        }
        
        .success-indicator {
            color: #28a745;
            font-weight: bold;
        }
        
        .warning-indicator {
            color: #ffc107;
            font-weight: bold;
        }
        
        .error-indicator {
            color: #dc3545;
            font-weight: bold;
        }
        
        /* Project Statistics */
        .project-stats {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .project-stats h4 {
            margin-top: 0;
            color: #495057;
        }
        
        /* Multi-Voice Character Assignment */
        .voice-assignment {
            margin: 10px 0;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #dee2e6;
        }
        
        .voice-assignment label {
            display: block;
            font-weight: bold;
            margin-bottom: 5px;
            color: #495057;
        }
        
        .voice-assignment select {
            width: 100%;
            padding: 8px 12px;
            background: #3A3939 !important;
            color: white !important;
            border: 1px solid #555;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .voice-assignment select option {
            background: #3A3939 !important;
            color: white !important;
            padding: 5px;
        }
        
        .voice-assignment select:focus {
            outline: none;
            border-color: #007bff;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
        }
        """
    
    def _load_theme_variants(self) -> None:
        """Load theme variant styles (light, dark, etc.)."""
        self.theme_variants = {
            'light': """
            /* Light Theme Variants */
            .theme-light .panel {
                background: #ffffff;
                color: #212529;
                border: 1px solid #dee2e6;
            }
            """,
            
            'dark_purple': """
            /* Dark Purple Theme for Production Studio */
            .theme-dark-purple .panel {
                background: #2b11c6;
                color: #ffffff;
                border: 1px solid #4a2fc7;
            }
            
            .theme-dark-purple .sub-panel {
                background: #3d1dd4;
                color: #ffffff;
            }
            
            .theme-dark-purple .item {
                background: #4a2fc7;
                color: #ffffff;
                border-left: 3px solid #7c3aed;
            }
            
            .theme-dark-purple .text-secondary {
                color: #e2e8f0;
            }
            
            .theme-dark-purple .text-muted {
                color: #cbd5e0;
            }
            """,
            
                         'audio_processing': """
             /* Audio Processing Theme */
             .audio-controls {
                 background: #e3f2fd;
                 border-radius: 8px;
                 padding: 15px;
                 margin: 10px 0;
             }
             
             .processing-panel {
                 background: #f3e5f5;
                 border-radius: 8px;
                 padding: 15px;
                 margin: 10px 0;
             }
             
             /* Enhanced Interface Components */
             .main-header {
                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white;
                 padding: 20px;
                 border-radius: 10px;
                 margin-bottom: 20px;
                 text-align: center;
             }
             
             .status-panel {
                 background: #3A3939;
                 border: 1px solid #dee2e6;
                 border-radius: 8px;
                 padding: 15px;
                 margin: 10px 0;
             }
             """
        }
    
    def get_complete_css(self, theme: str = 'light', additional_styles: Optional[List[str]] = None) -> str:
        """
        Get complete CSS with specified theme and additional styles.
        
        Args:
            theme: Theme variant to include ('light', 'dark_purple', 'audio_processing')
            additional_styles: List of additional style keys to include
            
        Returns:
            Complete CSS string ready for use in Gradio
        """
        css_parts = [
            self.base_styles,
            self.component_styles
        ]
        
        # Add theme variant
        if theme in self.theme_variants:
            css_parts.append(self.theme_variants[theme])
        
        # Add additional styles
        if additional_styles:
            for style_key in additional_styles:
                if style_key in self.theme_variants:
                    css_parts.append(self.theme_variants[style_key])
        
        return '\n'.join(css_parts)
    
    def get_inline_style(self, component_type: str, theme: str = 'light', **kwargs) -> str:
        """
        Get inline style for specific component types.
        
        Args:
            component_type: Type of component ('panel', 'status_success', etc.)
            theme: Theme to apply
            **kwargs: Additional style parameters
            
        Returns:
            Inline style string for HTML elements
        """
        styles = {
            'panel_light': "padding: 10px; background: #3A3939; border-radius: 5px; border: 1px solid #dee2e6; color: #212529;",
            'panel_dark_purple': "padding: 10px; background: #2b11c6; border-radius: 5px; border: 1px solid #4a2fc7; color: #ffffff;",
            'status_success': "padding: 10px; background: #3A3939; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724;",
            'status_error': "padding: 10px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px; color: #721c24;",
            'status_warning': "padding: 15px; background: #fff3cd; border: 1px solid #ffeeba; border-radius: 5px; color: #856404;",
            'status_info': "padding: 10px; background: #cce7ff; border: 1px solid #b3d9ff; border-radius: 5px; color: #004085;"
        }
        
        # Build component key
        component_key = f"{component_type}_{theme}" if f"{component_type}_{theme}" in styles else component_type
        
        base_style = styles.get(component_key, "")
        
        # Apply any custom kwargs
        if kwargs:
            style_additions = "; ".join([f"{k.replace('_', '-')}: {v}" for k, v in kwargs.items()])
            base_style += f"; {style_additions}"
        
        return base_style

# Global instance for easy access
chatterbox_styles = ChatterboxStyles()

# Convenience functions for easy usage
def get_css(theme: str = 'light', additional_themes: Optional[List[str]] = None) -> str:
    """Get complete CSS for Gradio interface."""
    return chatterbox_styles.get_complete_css(theme, additional_themes)

def get_inline_style(component_type: str, theme: str = 'light', **kwargs) -> str:
    """Get inline style for HTML components."""
    return chatterbox_styles.get_inline_style(component_type, theme, **kwargs)

def get_production_studio_css() -> str:
    """Get CSS specifically for Production Studio with dark purple theme."""
    return chatterbox_styles.get_complete_css('light', ['dark_purple'])

def get_audio_processing_css() -> str:
    """Get CSS specifically for Audio Processing interfaces."""
    return chatterbox_styles.get_complete_css('light', ['audio_processing']) 