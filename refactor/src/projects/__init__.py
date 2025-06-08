"""
Project Management Module

This module handles project-related functionality including:
- Project creation and loading
- Project metadata management
- Chunk regeneration and updates
- Project export and import
- Project backup and versioning
"""

# Import metadata functions
from .metadata import (
    save_project_metadata,
    load_project_metadata,
    update_project_metadata,
    get_project_status,
    set_project_status,
    get_project_info,
    validate_project_metadata,
    backup_project_metadata
)

# Import management functions
from .management import (
    get_existing_projects,
    get_project_choices,
    load_project_for_regeneration,
    create_project_directory,
    delete_project,
    rename_project,
    get_project_by_name,
    cleanup_project_temp_files,
    get_project_statistics
)

import os
from pathlib import Path

# Centralized location for the projects directory path
PROJECTS_DIR = "audiobook_projects"

def get_projects_path() -> Path:
    """Returns the absolute path to the audiobook projects directory."""
    # Note: This assumes the app is run from the 'refactor' directory.
    # A more robust solution might use a config file or environment variable.
    return Path(os.getcwd()) / PROJECTS_DIR

__all__ = [
    # Metadata functions
    'save_project_metadata',
    'load_project_metadata',
    'update_project_metadata',
    'get_project_status',
    'set_project_status',
    'get_project_info',
    'validate_project_metadata',
    'backup_project_metadata',
    
    # Management functions
    'get_existing_projects',
    'get_project_choices',
    'load_project_for_regeneration',
    'create_project_directory',
    'delete_project',
    'rename_project',
    'get_project_by_name',
    'cleanup_project_temp_files',
    'get_project_statistics'
] 