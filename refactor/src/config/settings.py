import os
from pathlib import Path
from typing import Optional

def get_project_root() -> Path:
    """
    Finds the project root directory by searching upwards for a sentinel file.
    A sentinel file like 'pyproject.toml' or '.git' indicates the root.
    """
    current_path = Path(__file__).resolve()
    search_path = current_path.parent
    while search_path != search_path.parent:
        if (search_path / 'pyproject.toml').exists() or (search_path / '.git').exists():
            return search_path
        search_path = search_path.parent
    return Path(__file__).parent.parent.parent

class AppConfig:
    """
    Centralized configuration management for the Chatterbox application.
    Resolves paths relative to the project root.
    """
    def __init__(self):
        self._project_root = get_project_root()
        print(f"✅ Project root detected: {self._project_root}")
        self.settings = self._load_settings()

    def _load_settings(self):
        """
        Loads settings programmatically. Paths are built from the detected project root
        to ensure they are always correct, regardless of where the script is run from.
        """
        return {
            "app_path": str(self._project_root),
            "voices_path": str(self._project_root / "speakers"),
            "projects_path": str(self._project_root / "audiobook_projects"),
            "default_theme": "light",
            "supported_text_formats": [".txt", ".md", ".rtf"],
            "supported_audio_formats": [".wav", ".mp3", ".flac", ".ogg"],
            "max_file_size_mb": 100,
            "temp_file_prefix": "chatterbox_temp_",
            "default_volume_target": -18.0,
            "sample_rate": 44100
        }

    def get_setting(self, key: str, default=None):
        """Safely retrieves a setting by key."""
        return self.settings.get(key, default)

    def get_app_path(self) -> str:
        """Returns the absolute path to the application's root directory."""
        return self.get_setting("app_path")

    def get_voices_path(self) -> str:
        """Returns the absolute path to the voice library."""
        return self.get_setting("voices_path")
    
    def get_projects_path(self) -> str:
        """Returns the absolute path to the audiobook projects directory."""
        return self.get_setting("projects_path")

# --- Global Instance ---
config = AppConfig()

# Volume normalization presets
VOLUME_PRESETS = {
    "audiobook": -18.0,
    "podcast": -16.0,
    "broadcast": -23.0,
    "custom": None
}

def get_volume_preset(preset_name: str) -> Optional[float]:
    """Get volume normalization preset value."""
    return VOLUME_PRESETS.get(preset_name) 