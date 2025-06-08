"""Project metadata management for audiobook generation.

Handles project metadata creation, loading, and persistence operations.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime


def save_project_metadata(
    project_dir: str,
    metadata_or_project_name,
    text_content: str = None,
    voice_info: dict = None,
    chunks: list = None,
    project_type: str = "single_voice"
) -> None:
    """Save project metadata to JSON file.
    
    Args:
        project_dir: Project directory path
        metadata_or_project_name: Either a complete metadata dict OR project name string
        text_content: Original text content (if using individual params)
        voice_info: Voice configuration information (if using individual params)
        chunks: List of text chunks (if using individual params)
        project_type: Type of project (single_voice or multi_voice)
    """
    if isinstance(metadata_or_project_name, dict):
        # Called with complete metadata dictionary
        metadata = metadata_or_project_name
    else:
        # Called with individual parameters (legacy format)
        project_name = metadata_or_project_name
        metadata = {
            'project_name': project_name,
            'project_type': project_type,
            'created_at': datetime.now().isoformat(),
            'text_content': text_content,
            'voice_info': voice_info,
            'chunks': chunks,
            'total_chunks': len(chunks) if chunks else 0,
            'status': 'in_progress'
        }
    
    metadata_path = os.path.join(project_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def load_project_metadata(project_dir: str) -> dict:
    """Load project metadata from JSON file.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        Project metadata dictionary
    """
    metadata_path = os.path.join(project_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata for {project_dir}: {e}")
    return {}


def update_project_metadata(project_dir: str, updates: Dict[str, Any]) -> bool:
    """Update existing project metadata with new values.
    
    Args:
        project_dir: Project directory path
        updates: Dictionary of metadata fields to update
        
    Returns:
        bool: True if update was successful
    """
    try:
        metadata = load_project_metadata(project_dir)
        metadata.update(updates)
        metadata['updated_at'] = datetime.now().isoformat()
        
        metadata_path = os.path.join(project_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error updating metadata for {project_dir}: {e}")
        return False


def get_project_status(project_dir: str) -> str:
    """Get the current status of a project.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        str: Project status ('in_progress', 'completed', 'error', etc.)
    """
    metadata = load_project_metadata(project_dir)
    return metadata.get('status', 'unknown')


def set_project_status(project_dir: str, status: str) -> bool:
    """Set the status of a project.
    
    Args:
        project_dir: Project directory path
        status: New status to set
        
    Returns:
        bool: True if status was updated successfully
    """
    return update_project_metadata(project_dir, {'status': status})


def get_project_info(project_dir: str) -> Dict[str, Any]:
    """Get comprehensive project information.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        Dict containing project information
    """
    metadata = load_project_metadata(project_dir)
    
    if metadata:
        # Use metadata information
        return {
            'name': metadata.get('project_name', os.path.basename(project_dir)),
            'path': project_dir,
            'type': metadata.get('project_type', 'unknown'),
            'created_at': metadata.get('created_at', ''),
            'updated_at': metadata.get('updated_at', ''),
            'total_chunks': metadata.get('total_chunks', 0),
            'status': metadata.get('status', 'unknown'),
            'voice_info': metadata.get('voice_info', {}),
            'text_content': metadata.get('text_content', '')
        }
    else:
        # Fallback to directory scanning for legacy projects
        audio_files = []
        if os.path.isdir(project_dir):
            try:
                audio_files = [f for f in os.listdir(project_dir) if f.endswith('.wav')]
            except:
                pass
        
        return {
            'name': os.path.basename(project_dir),
            'path': project_dir,
            'type': 'legacy',
            'created_at': '',
            'updated_at': '',
            'total_chunks': len(audio_files),
            'status': 'completed' if audio_files else 'empty',
            'voice_info': {},
            'text_content': ''
        }


def validate_project_metadata(metadata: dict) -> Tuple[bool, str]:
    """Validate project metadata structure and required fields.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    required_fields = ['project_name', 'project_type', 'created_at']
    
    for field in required_fields:
        if field not in metadata:
            return False, f"Missing required field: {field}"
    
    # Validate project type
    valid_types = ['single_voice', 'multi_voice', 'legacy']
    if metadata.get('project_type') not in valid_types:
        return False, f"Invalid project type: {metadata.get('project_type')}"
    
    # Validate date format
    try:
        datetime.fromisoformat(metadata['created_at'])
    except ValueError:
        return False, f"Invalid date format: {metadata['created_at']}"
    
    return True, ""


def backup_project_metadata(project_dir: str) -> bool:
    """Create a backup of project metadata.
    
    Args:
        project_dir: Project directory path
        
    Returns:
        bool: True if backup was created successfully
    """
    try:
        metadata_path = os.path.join(project_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(project_dir, f'metadata_backup_{timestamp}.json')
        
        import shutil
        shutil.copy2(metadata_path, backup_path)
        return True
        
    except Exception as e:
        print(f"Error creating metadata backup: {e}")
        return False 