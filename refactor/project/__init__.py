"""
Project management module for Chatterbox Audiobook Studio.

This module provides comprehensive project and chunk management operations
for the refactored audiobook studio system.
"""

from .project_manager import *
from .chunk_processor import *
from .metadata_manager import *

__all__ = [
    'ProjectManager', 'ChunkProcessor', 'MetadataManager',
    'create_project', 'load_project', 'save_project',
    'process_chunks', 'get_project_status'
] 