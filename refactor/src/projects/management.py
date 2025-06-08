"""Project management operations for audiobook generation.

Handles project discovery, UI integration, and CRUD operations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

from .metadata import load_project_metadata, get_project_info

# Default projects directory
PROJECTS_DIR = "audiobook_projects"

def get_projects_path() -> str:
    """Returns the path to the audiobook projects directory."""
    return PROJECTS_DIR


def get_existing_projects() -> List[Dict[str, Any]]:
    """Get list of existing audiobook projects.
        
    Returns:
        List of project information dictionaries
    """
    projects = []
    output_dir = get_projects_path()
    
    if not os.path.exists(output_dir):
        return projects
    
    try:
        for item in os.listdir(output_dir):
            project_dir = os.path.join(output_dir, item)
            if os.path.isdir(project_dir):
                project_info = get_project_info(project_dir)
                projects.append(project_info)
                
    except Exception as e:
        print(f"Warning: Error scanning projects directory: {e}")
    
    # Sort by creation date (newest first)
    def get_sort_key(project):
        created_at = project.get('created_at', '')
        if created_at:
            try:
                return datetime.fromisoformat(created_at)
            except:
                pass
        return datetime.min
    
    projects.sort(key=get_sort_key, reverse=True)
    return projects


def get_project_choices() -> List[str]:
    """Get project names for UI dropdowns.
    
    Returns:
        List of project names formatted for UI display
    """
    projects = get_existing_projects()
    if not projects:
        return ["No projects found"]
    
    # Format: "project_name (type - chunks)"
    choices = []
    for project in projects:
        name = project['name']
        project_type = project['type']
        chunk_count = project['total_chunks']
        formatted = f"{name} ({project_type} - {chunk_count} chunks)"
        choices.append(formatted)
    
    return choices


def load_project_for_regeneration(project_name: str) -> Tuple[str, str, str, str]:
    """Load project data for regeneration interface.
    
    Args:
        project_name: Name of the project to load (may be formatted string)
        
    Returns:
        tuple: (text_content, voice_name, project_type, status_message)
    """
    if not project_name or project_name == "No projects found":
        return "", "", "", "No project selected"
    
    # Extract actual project name from formatted string
    actual_name = project_name.split(' (')[0] if ' (' in project_name else project_name
    
    projects = get_existing_projects()
    project_info = None
    
    for project in projects:
        if project['name'] == actual_name:
            project_info = project
            break
    
    if not project_info:
        return "", "", "", f"❌ Project '{actual_name}' not found"
    
    # Load project metadata
    metadata = load_project_metadata(project_info['path'])
    
    if not metadata:
        return "", "", "", f"❌ Could not load project metadata for '{actual_name}'"
    
    text_content = metadata.get('text_content', '')
    voice_info = metadata.get('voice_info', {})
    project_type = metadata.get('project_type', 'single_voice')
    
    # Extract voice name based on project type
    if project_type == 'single_voice':
        voice_name = voice_info.get('voice_name', '')
    else:
        # For multi-voice, return the first voice or indicate multi-voice
        voice_assignments = voice_info.get('voice_assignments', {})
        if voice_assignments:
            voice_name = f"Multi-voice: {len(voice_assignments)} characters"
        else:
            voice_name = "Multi-voice (no assignments)"
    
    return text_content, voice_name, project_type, f"✅ Loaded project '{actual_name}'"


def create_project_directory(project_name: str, output_dir: str = "audiobook_projects") -> str:
    """Create a new project directory.
    
    Args:
        project_name: Name of the project
        output_dir: Base directory for projects
        
    Returns:
        str: Path to the created project directory
    """
    # Sanitize project name
    safe_project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_project_name = safe_project_name.replace(' ', '_')
    
    if not safe_project_name:
        safe_project_name = "untitled_project"
    
    # Create unique directory name if needed
    base_dir = os.path.join(output_dir, safe_project_name)
    project_dir = base_dir
    counter = 1
    
    while os.path.exists(project_dir):
        project_dir = f"{base_dir}_{counter}"
        counter += 1
    
    # Create the directory
    os.makedirs(project_dir, exist_ok=True)
    return project_dir


def delete_project(project_name: str, output_dir: str = "audiobook_projects") -> Tuple[bool, str]:
    """Delete a project and all its files.
    
    Args:
        project_name: Name of the project to delete
        output_dir: Base directory for projects
        
    Returns:
        tuple: (success, status_message)
    """
    try:
        # Find the project
        projects = get_existing_projects()
        project_to_delete = None
        
        for project in projects:
            if project['name'] == project_name:
                project_to_delete = project
                break
        
        if not project_to_delete:
            return False, f"❌ Project '{project_name}' not found"
        
        project_path = project_to_delete['path']
        
        # Create backup before deletion (optional)
        # backup_path = f"{project_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # shutil.copytree(project_path, backup_path)
        
        # Delete the project directory
        shutil.rmtree(project_path)
        
        return True, f"✅ Project '{project_name}' deleted successfully"
        
    except Exception as e:
        return False, f"❌ Error deleting project: {str(e)}"


def rename_project(old_name: str, new_name: str, output_dir: str = "audiobook_projects") -> Tuple[bool, str]:
    """Rename a project.
    
    Args:
        old_name: Current project name
        new_name: New project name
        output_dir: Base directory for projects
        
    Returns:
        tuple: (success, status_message)
    """
    try:
        # Find the existing project
        projects = get_existing_projects()
        project_to_rename = None
        
        for project in projects:
            if project['name'] == old_name:
                project_to_rename = project
                break
        
        if not project_to_rename:
            return False, f"❌ Project '{old_name}' not found"
        
        old_path = project_to_rename['path']
        
        # Create new project directory
        new_project_dir = create_project_directory(new_name, output_dir)
        
        # Move all files to new directory
        for item in os.listdir(old_path):
            shutil.move(os.path.join(old_path, item), os.path.join(new_project_dir, item))
        
        # Update metadata with new name
        from .metadata import update_project_metadata
        update_project_metadata(new_project_dir, {'project_name': new_name})
        
        # Remove old directory
        os.rmdir(old_path)
        
        return True, f"✅ Project renamed from '{old_name}' to '{new_name}'"
        
    except Exception as e:
        return False, f"❌ Error renaming project: {str(e)}"


def get_project_by_name(project_name: str, output_dir: str = "audiobook_projects") -> Optional[Dict[str, Any]]:
    """Get project information by name.
    
    Args:
        project_name: Name of the project
        output_dir: Base directory for projects
        
    Returns:
        Project information dictionary or None if not found
    """
    projects = get_existing_projects()
    
    for project in projects:
        if project['name'] == project_name:
            return project
    
    return None


def cleanup_project_temp_files(project_name: str, output_dir: str = "audiobook_projects") -> str:
    """Clean up temporary files in a project directory.
    
    Args:
        project_name: Name of the project
        output_dir: Base directory for projects
        
    Returns:
        str: Status message
    """
    try:
        project = get_project_by_name(project_name, output_dir)
        if not project:
            return f"❌ Project '{project_name}' not found"
        
        project_dir = project['path']
        temp_patterns = ['temp_', 'tmp_', '.tmp', '~', '_backup_']
        
        cleaned_files = []
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            
            # Check if it's a temporary file
            is_temp = any(pattern in item.lower() for pattern in temp_patterns)
            
            if is_temp and os.path.isfile(item_path):
                os.remove(item_path)
                cleaned_files.append(item)
        
        if cleaned_files:
            return f"✅ Cleaned {len(cleaned_files)} temporary files: {', '.join(cleaned_files[:3])}{'...' if len(cleaned_files) > 3 else ''}"
        else:
            return "✅ No temporary files found to clean"
            
    except Exception as e:
        return f"❌ Error cleaning temporary files: {str(e)}"


def get_project_statistics(output_dir: str = "audiobook_projects") -> Dict[str, Any]:
    """Get overall project statistics.
    
    Args:
        output_dir: Base directory for projects
        
    Returns:
        Dictionary containing project statistics
    """
    projects = get_existing_projects()
    
    stats = {
        'total_projects': len(projects),
        'by_type': {},
        'by_status': {},
        'total_chunks': 0,
        'completed_projects': 0,
        'in_progress_projects': 0
    }
    
    for project in projects:
        # Count by type
        project_type = project.get('type', 'unknown')
        stats['by_type'][project_type] = stats['by_type'].get(project_type, 0) + 1
        
        # Count by status
        status = project.get('status', 'unknown')
        stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
        
        # Accumulate chunks
        stats['total_chunks'] += project.get('total_chunks', 0)
        
        # Count completion status
        if status == 'completed':
            stats['completed_projects'] += 1
        elif status == 'in_progress':
            stats['in_progress_projects'] += 1
    
    return stats 