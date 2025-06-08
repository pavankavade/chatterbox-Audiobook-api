"""
# ==============================================================================
# PROJECT MANAGEMENT MODULE
# ==============================================================================
# 
# This module provides comprehensive project management for the Chatterbox
# Audiobook Studio refactored system. It handles project creation, loading,
# saving, and coordination of all project-level operations.
# 
# **Key Features:**
# - **Project Lifecycle**: Complete create, load, save, delete operations
# - **Chunk Management**: Integration with chunk processing systems
# - **Metadata Handling**: Project-level metadata and configuration
# - **Voice Coordination**: Integration with voice management systems
# - **Original Compatibility**: Full compatibility with existing project formats
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict

# Import other modules
from voice.voice_manager import VoiceManager
from core.tts_engine import RefactoredTTSEngine

# ==============================================================================
# PROJECT DATA STRUCTURES
# ==============================================================================

@dataclass
class ProjectMetadata:
    """
    Project metadata structure for audiobook projects.
    """
    project_name: str
    created_date: str
    modified_date: str
    author: str = ""
    title: str = ""
    description: str = ""
    version: str = "1.0"
    total_chunks: int = 0
    processed_chunks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectMetadata':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class ProjectSettings:
    """
    Project-specific settings and configuration.
    """
    default_voice: str = ""
    narrator_voice: str = ""
    target_sample_rate: int = 24000
    audio_format: str = "wav"
    output_directory: str = ""
    auto_save_enabled: bool = True
    chunk_size_limit: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectSettings':
        """Create from dictionary."""
        return cls(**data)

# ==============================================================================
# PROJECT MANAGER CLASS
# ==============================================================================

class ProjectManager:
    """
    Professional project management system for audiobook projects.
    
    This class provides comprehensive project-level operations including
    project creation, loading, saving, and coordination with all other
    system components.
    
    **Management Features:**
    - **Project CRUD**: Complete create, read, update, delete operations
    - **Chunk Integration**: Seamless chunk processing coordination
    - **Voice Integration**: Project-level voice management
    - **Metadata Management**: Comprehensive project metadata handling
    - **Original Compatibility**: Full compatibility with existing projects
    """
    
    def __init__(self, projects_directory: str = "projects"):
        """
        Initialize the project manager.
        
        Args:
            projects_directory (str): Directory for storing projects
        """
        self.projects_directory = Path(projects_directory)
        self.projects_directory.mkdir(parents=True, exist_ok=True)
        
        self.current_project: Optional[Dict[str, Any]] = None
        self.current_project_path: Optional[Path] = None
        
        # Initialize component managers
        self.voice_manager = VoiceManager()
        self.tts_engine = RefactoredTTSEngine()
        
        print(f"✅ Project Manager initialized - Projects: {self.projects_directory}")
    
    def create_project(
        self,
        project_name: str,
        title: str = "",
        author: str = "",
        description: str = "",
        default_voice: str = "",
        narrator_voice: str = ""
    ) -> Tuple[bool, str]:
        """
        Create a new audiobook project.
        
        Args:
            project_name (str): Unique project identifier
            title (str): Project title
            author (str): Project author
            description (str): Project description
            default_voice (str): Default voice for the project
            narrator_voice (str): Narrator voice for the project
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Validate project name
            if not project_name or not project_name.strip():
                return False, "Project name cannot be empty"
            
            # Create project directory
            project_path = self.projects_directory / project_name
            if project_path.exists():
                return False, f"Project '{project_name}' already exists"
            
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Create project metadata
            now = datetime.now().isoformat()
            metadata = ProjectMetadata(
                project_name=project_name,
                created_date=now,
                modified_date=now,
                author=author,
                title=title or project_name,
                description=description
            )
            
            # Create project settings
            settings = ProjectSettings(
                default_voice=default_voice,
                narrator_voice=narrator_voice or default_voice,
                output_directory=str(project_path / "output")
            )
            
            # Create project structure
            project_data = {
                'metadata': metadata.to_dict(),
                'settings': settings.to_dict(),
                'chunks': [],
                'voice_assignments': {},
                'processing_status': {
                    'total_chunks': 0,
                    'processed_chunks': 0,
                    'failed_chunks': 0,
                    'last_processed': None
                }
            }
            
            # Save project file
            project_file = project_path / "project.json"
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(project_data, f, indent=2, ensure_ascii=False)
            
            # Create output directory
            output_dir = project_path / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Create chunks directory
            chunks_dir = project_path / "chunks"
            chunks_dir.mkdir(exist_ok=True)
            
            return True, f"✅ Project '{project_name}' created successfully"
            
        except Exception as e:
            return False, f"❌ Error creating project: {str(e)}"
    
    def load_project(self, project_name: str) -> Tuple[bool, str]:
        """
        Load an existing project.
        
        Args:
            project_name (str): Name of project to load
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            project_path = self.projects_directory / project_name
            project_file = project_path / "project.json"
            
            if not project_file.exists():
                return False, f"Project '{project_name}' not found"
            
            # Load project data
            with open(project_file, 'r', encoding='utf-8') as f:
                project_data = json.load(f)
            
            # Validate project data structure
            required_keys = ['metadata', 'settings', 'chunks']
            for key in required_keys:
                if key not in project_data:
                    return False, f"Invalid project file: missing '{key}'"
            
            # Set current project
            self.current_project = project_data
            self.current_project_path = project_path
            
            # Update modified date
            project_data['metadata']['modified_date'] = datetime.now().isoformat()
            
            return True, f"✅ Project '{project_name}' loaded successfully"
            
        except Exception as e:
            return False, f"❌ Error loading project: {str(e)}"
    
    def save_project(self) -> Tuple[bool, str]:
        """
        Save the current project.
        
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not self.current_project or not self.current_project_path:
                return False, "No project currently loaded"
            
            # Update modified date
            self.current_project['metadata']['modified_date'] = datetime.now().isoformat()
            
            # Save project file
            project_file = self.current_project_path / "project.json"
            with open(project_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_project, f, indent=2, ensure_ascii=False)
            
            project_name = self.current_project['metadata']['project_name']
            return True, f"✅ Project '{project_name}' saved successfully"
            
        except Exception as e:
            return False, f"❌ Error saving project: {str(e)}"
    
    def get_project_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all available projects.
        
        Returns:
            List[Dict[str, Any]]: List of project information
        """
        projects = []
        
        try:
            if not self.projects_directory.exists():
                return projects
            
            for item in self.projects_directory.iterdir():
                if item.is_dir():
                    project_file = item / "project.json"
                    if project_file.exists():
                        try:
                            with open(project_file, 'r', encoding='utf-8') as f:
                                project_data = json.load(f)
                            
                            metadata = project_data.get('metadata', {})
                            projects.append({
                                'name': item.name,
                                'title': metadata.get('title', item.name),
                                'author': metadata.get('author', ''),
                                'description': metadata.get('description', ''),
                                'created_date': metadata.get('created_date', ''),
                                'modified_date': metadata.get('modified_date', ''),
                                'total_chunks': metadata.get('total_chunks', 0),
                                'processed_chunks': metadata.get('processed_chunks', 0)
                            })
                        except Exception as e:
                            print(f"⚠️  Error reading project {item.name}: {e}")
            
            # Sort by modified date (newest first)
            projects.sort(key=lambda x: x.get('modified_date', ''), reverse=True)
            
        except Exception as e:
            print(f"❌ Error listing projects: {e}")
        
        return projects
    
    def delete_project(self, project_name: str) -> Tuple[bool, str]:
        """
        Delete a project.
        
        Args:
            project_name (str): Name of project to delete
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            project_path = self.projects_directory / project_name
            
            if not project_path.exists():
                return False, f"Project '{project_name}' not found"
            
            # Close current project if it's the one being deleted
            if (self.current_project and 
                self.current_project['metadata']['project_name'] == project_name):
                self.current_project = None
                self.current_project_path = None
            
            # Delete project directory
            shutil.rmtree(project_path)
            
            return True, f"✅ Project '{project_name}' deleted successfully"
            
        except Exception as e:
            return False, f"❌ Error deleting project: {str(e)}"
    
    def get_current_project_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the currently loaded project.
        
        Returns:
            Optional[Dict[str, Any]]: Current project information or None
        """
        if not self.current_project:
            return None
        
        return {
            'metadata': self.current_project.get('metadata', {}),
            'settings': self.current_project.get('settings', {}),
            'chunk_count': len(self.current_project.get('chunks', [])),
            'processing_status': self.current_project.get('processing_status', {}),
            'project_path': str(self.current_project_path) if self.current_project_path else None
        }
    
    def update_project_metadata(
        self,
        title: Optional[str] = None,
        author: Optional[str] = None,
        description: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Update project metadata.
        
        Args:
            title (Optional[str]): New title
            author (Optional[str]): New author
            description (Optional[str]): New description
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.current_project:
            return False, "No project currently loaded"
        
        try:
            metadata = self.current_project['metadata']
            
            if title is not None:
                metadata['title'] = title
            if author is not None:
                metadata['author'] = author
            if description is not None:
                metadata['description'] = description
            
            metadata['modified_date'] = datetime.now().isoformat()
            
            return True, "✅ Project metadata updated successfully"
            
        except Exception as e:
            return False, f"❌ Error updating metadata: {str(e)}"
    
    def update_project_settings(
        self,
        default_voice: Optional[str] = None,
        narrator_voice: Optional[str] = None,
        audio_format: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Update project settings.
        
        Args:
            default_voice (Optional[str]): New default voice
            narrator_voice (Optional[str]): New narrator voice
            audio_format (Optional[str]): New audio format
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        if not self.current_project:
            return False, "No project currently loaded"
        
        try:
            settings = self.current_project['settings']
            
            if default_voice is not None:
                settings['default_voice'] = default_voice
            if narrator_voice is not None:
                settings['narrator_voice'] = narrator_voice
            if audio_format is not None:
                settings['audio_format'] = audio_format
            
            # Update metadata modified date
            self.current_project['metadata']['modified_date'] = datetime.now().isoformat()
            
            return True, "✅ Project settings updated successfully"
            
        except Exception as e:
            return False, f"❌ Error updating settings: {str(e)}"
    
    def get_project_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive project statistics.
        
        Returns:
            Dict[str, Any]: Project statistics
        """
        stats = {
            'total_projects': 0,
            'current_project_loaded': self.current_project is not None,
            'projects_directory': str(self.projects_directory)
        }
        
        # Count total projects
        if self.projects_directory.exists():
            project_dirs = [
                item for item in self.projects_directory.iterdir()
                if item.is_dir() and (item / "project.json").exists()
            ]
            stats['total_projects'] = len(project_dirs)
        
        # Current project stats
        if self.current_project:
            current_info = self.get_current_project_info()
            if current_info:
                stats['current_project'] = {
                    'name': current_info['metadata'].get('project_name', ''),
                    'title': current_info['metadata'].get('title', ''),
                    'chunk_count': current_info['chunk_count'],
                    'processing_status': current_info['processing_status']
                }
        
        return stats

# ==============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# ==============================================================================

# Global project manager for convenience
_global_project_manager: Optional[ProjectManager] = None

def get_global_project_manager() -> ProjectManager:
    """Get or create the global project manager instance."""
    global _global_project_manager
    if _global_project_manager is None:
        _global_project_manager = ProjectManager()
    return _global_project_manager

def create_project(
    project_name: str,
    title: str = "",
    author: str = "",
    description: str = ""
) -> Tuple[bool, str]:
    """Create project (backward compatibility)."""
    manager = get_global_project_manager()
    return manager.create_project(project_name, title, author, description)

def load_project(project_name: str) -> Tuple[bool, str]:
    """Load project (backward compatibility)."""
    manager = get_global_project_manager()
    return manager.load_project(project_name)

def save_project() -> Tuple[bool, str]:
    """Save current project (backward compatibility)."""
    manager = get_global_project_manager()
    return manager.save_project()

def get_project_status() -> Dict[str, Any]:
    """Get project status (backward compatibility)."""
    manager = get_global_project_manager()
    return manager.get_project_statistics()

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Project Management module loaded")
print("📁 Project lifecycle management ready for audiobook projects") 