"""
# ==============================================================================
# METADATA MANAGER MODULE
# ==============================================================================
# 
# This module provides comprehensive metadata management for the Chatterbox
# Audiobook Studio refactored system. It handles project metadata, configuration
# management, and data persistence operations.
# 
# **Key Features:**
# - **Metadata CRUD**: Complete metadata create, read, update, delete operations
# - **Configuration Management**: Project-level configuration handling
# - **Data Validation**: Comprehensive metadata validation and integrity checking
# - **Export/Import**: Metadata backup and restoration capabilities
# - **Original Compatibility**: Full compatibility with existing metadata formats
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict, field

# ==============================================================================
# METADATA DATA STRUCTURES
# ==============================================================================

@dataclass
class AudiobookMetadata:
    """
    Comprehensive audiobook metadata structure.
    """
    # Basic Information
    title: str = ""
    author: str = ""
    narrator: str = ""
    description: str = ""
    genre: str = ""
    language: str = "en"
    
    # Publication Information
    isbn: str = ""
    publisher: str = ""
    publication_date: str = ""
    series: str = ""
    series_number: int = 0
    
    # Technical Information
    total_duration: float = 0.0  # in seconds
    chapter_count: int = 0
    total_words: int = 0
    total_characters: int = 0
    
    # Project Information
    project_version: str = "1.0"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_date: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    
    # Voice and Audio Settings
    primary_voice: str = ""
    narrator_voice: str = ""
    voice_assignments: Dict[str, str] = field(default_factory=dict)
    audio_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Custom Fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudiobookMetadata':
        """Create from dictionary."""
        # Handle missing fields with defaults
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

@dataclass
class ProcessingMetadata:
    """
    Processing-specific metadata for tracking generation progress.
    """
    total_chunks: int = 0
    processed_chunks: int = 0
    failed_chunks: int = 0
    processing_start_time: str = ""
    processing_end_time: str = ""
    processing_duration: float = 0.0
    
    # Chunk-level statistics
    average_chunk_duration: float = 0.0
    total_audio_duration: float = 0.0
    processing_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    chunks_per_minute: float = 0.0
    audio_minutes_per_hour: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingMetadata':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

# ==============================================================================
# METADATA MANAGER CLASS
# ==============================================================================

class MetadataManager:
    """
    Professional metadata management system for audiobook projects.
    
    This class provides comprehensive metadata operations including creation,
    validation, persistence, and integration with all project components.
    
    **Management Features:**
    - **Metadata CRUD**: Complete metadata lifecycle management
    - **Validation**: Comprehensive data validation and integrity checking
    - **Persistence**: Reliable metadata saving and loading
    - **Export/Import**: Complete metadata backup and restoration
    - **Integration**: Seamless integration with project and chunk systems
    """
    
    def __init__(self, project_directory: Optional[Union[str, Path]] = None):
        """
        Initialize the metadata manager.
        
        Args:
            project_directory (Optional[Union[str, Path]]): Project directory path
        """
        self.project_directory = Path(project_directory) if project_directory else None
        self.audiobook_metadata = AudiobookMetadata()
        self.processing_metadata = ProcessingMetadata()
        
        print("✅ Metadata Manager initialized")
    
    def create_new_metadata(
        self,
        title: str,
        author: str = "",
        narrator: str = "",
        description: str = "",
        **kwargs
    ) -> AudiobookMetadata:
        """
        Create new audiobook metadata.
        
        Args:
            title (str): Audiobook title
            author (str): Author name
            narrator (str): Narrator name
            description (str): Book description
            **kwargs: Additional metadata fields
            
        Returns:
            AudiobookMetadata: New metadata object
        """
        now = datetime.now().isoformat()
        
        self.audiobook_metadata = AudiobookMetadata(
            title=title,
            author=author,
            narrator=narrator,
            description=description,
            created_date=now,
            modified_date=now,
            **kwargs
        )
        
        return self.audiobook_metadata
    
    def update_metadata(self, **updates) -> Tuple[bool, str]:
        """
        Update audiobook metadata fields.
        
        Args:
            **updates: Fields to update
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            for key, value in updates.items():
                if hasattr(self.audiobook_metadata, key):
                    setattr(self.audiobook_metadata, key, value)
                else:
                    # Add to custom fields
                    self.audiobook_metadata.custom_fields[key] = value
            
            # Update modification date
            self.audiobook_metadata.modified_date = datetime.now().isoformat()
            
            return True, "✅ Metadata updated successfully"
            
        except Exception as e:
            return False, f"❌ Error updating metadata: {str(e)}"
    
    def validate_metadata(self) -> Tuple[bool, List[str]]:
        """
        Validate metadata for completeness and consistency.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields validation
        if not self.audiobook_metadata.title.strip():
            errors.append("Title is required")
        
        if not self.audiobook_metadata.author.strip():
            errors.append("Author is required")
        
        # Data type validation
        if not isinstance(self.audiobook_metadata.total_duration, (int, float)):
            errors.append("Total duration must be a number")
        
        if not isinstance(self.audiobook_metadata.chapter_count, int):
            errors.append("Chapter count must be an integer")
        
        # Range validation
        if self.audiobook_metadata.total_duration < 0:
            errors.append("Total duration cannot be negative")
        
        if self.audiobook_metadata.chapter_count < 0:
            errors.append("Chapter count cannot be negative")
        
        # Language code validation (basic)
        valid_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
        if self.audiobook_metadata.language not in valid_languages:
            errors.append(f"Language '{self.audiobook_metadata.language}' not in supported list")
        
        # ISBN validation (basic format check)
        if self.audiobook_metadata.isbn:
            isbn_clean = self.audiobook_metadata.isbn.replace("-", "").replace(" ", "")
            if not (len(isbn_clean) in [10, 13] and isbn_clean.isdigit()):
                errors.append("ISBN format is invalid")
        
        return len(errors) == 0, errors
    
    def save_metadata(self, filename: str = "metadata.json") -> Tuple[bool, str]:
        """
        Save metadata to a JSON file.
        
        Args:
            filename (str): Filename for the metadata file
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not self.project_directory:
                return False, "No project directory set"
            
            metadata_path = self.project_directory / filename
            
            # Combine all metadata
            complete_metadata = {
                'audiobook_metadata': self.audiobook_metadata.to_dict(),
                'processing_metadata': self.processing_metadata.to_dict(),
                'saved_date': datetime.now().isoformat(),
                'metadata_version': '1.0'
            }
            
            # Ensure directory exists
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(complete_metadata, f, indent=2, ensure_ascii=False)
            
            return True, f"✅ Metadata saved to {metadata_path}"
            
        except Exception as e:
            return False, f"❌ Error saving metadata: {str(e)}"
    
    def load_metadata(self, filename: str = "metadata.json") -> Tuple[bool, str]:
        """
        Load metadata from a JSON file.
        
        Args:
            filename (str): Filename of the metadata file
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            if not self.project_directory:
                return False, "No project directory set"
            
            metadata_path = self.project_directory / filename
            
            if not metadata_path.exists():
                return False, f"Metadata file not found: {metadata_path}"
            
            # Load from file
            with open(metadata_path, 'r', encoding='utf-8') as f:
                complete_metadata = json.load(f)
            
            # Extract metadata components
            if 'audiobook_metadata' in complete_metadata:
                self.audiobook_metadata = AudiobookMetadata.from_dict(
                    complete_metadata['audiobook_metadata']
                )
            
            if 'processing_metadata' in complete_metadata:
                self.processing_metadata = ProcessingMetadata.from_dict(
                    complete_metadata['processing_metadata']
                )
            
            return True, f"✅ Metadata loaded from {metadata_path}"
            
        except Exception as e:
            return False, f"❌ Error loading metadata: {str(e)}"
    
    def export_metadata(
        self,
        export_path: Union[str, Path],
        format: str = "json",
        include_processing: bool = True
    ) -> Tuple[bool, str]:
        """
        Export metadata to various formats.
        
        Args:
            export_path (Union[str, Path]): Export file path
            format (str): Export format ("json", "csv", "xml")
            include_processing (bool): Include processing metadata
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            export_path = Path(export_path)
            
            if format.lower() == "json":
                return self._export_json(export_path, include_processing)
            elif format.lower() == "csv":
                return self._export_csv(export_path, include_processing)
            elif format.lower() == "xml":
                return self._export_xml(export_path, include_processing)
            else:
                return False, f"Unsupported export format: {format}"
            
        except Exception as e:
            return False, f"❌ Error exporting metadata: {str(e)}"
    
    def update_processing_statistics(
        self,
        total_chunks: int = 0,
        processed_chunks: int = 0,
        failed_chunks: int = 0,
        processing_duration: float = 0.0,
        **kwargs
    ) -> None:
        """
        Update processing metadata with statistics.
        
        Args:
            total_chunks (int): Total number of chunks
            processed_chunks (int): Number of processed chunks
            failed_chunks (int): Number of failed chunks
            processing_duration (float): Total processing duration
            **kwargs: Additional processing metrics
        """
        self.processing_metadata.total_chunks = total_chunks
        self.processing_metadata.processed_chunks = processed_chunks
        self.processing_metadata.failed_chunks = failed_chunks
        self.processing_metadata.processing_duration = processing_duration
        
        # Calculate derived metrics
        if processing_duration > 0:
            self.processing_metadata.chunks_per_minute = (processed_chunks / processing_duration) * 60
        
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(self.processing_metadata, key):
                setattr(self.processing_metadata, key, value)
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive metadata summary.
        
        Returns:
            Dict[str, Any]: Metadata summary
        """
        return {
            'basic_info': {
                'title': self.audiobook_metadata.title,
                'author': self.audiobook_metadata.author,
                'narrator': self.audiobook_metadata.narrator,
                'language': self.audiobook_metadata.language,
                'genre': self.audiobook_metadata.genre
            },
            'technical_info': {
                'total_duration': self.audiobook_metadata.total_duration,
                'chapter_count': self.audiobook_metadata.chapter_count,
                'total_words': self.audiobook_metadata.total_words,
                'total_characters': self.audiobook_metadata.total_characters
            },
            'processing_info': {
                'total_chunks': self.processing_metadata.total_chunks,
                'processed_chunks': self.processing_metadata.processed_chunks,
                'failed_chunks': self.processing_metadata.failed_chunks,
                'processing_duration': self.processing_metadata.processing_duration
            },
            'voice_info': {
                'primary_voice': self.audiobook_metadata.primary_voice,
                'narrator_voice': self.audiobook_metadata.narrator_voice,
                'voice_assignments_count': len(self.audiobook_metadata.voice_assignments)
            },
            'project_info': {
                'created_date': self.audiobook_metadata.created_date,
                'modified_date': self.audiobook_metadata.modified_date,
                'project_version': self.audiobook_metadata.project_version,
                'created_by': self.audiobook_metadata.created_by
            }
        }
    
    def _export_json(self, export_path: Path, include_processing: bool) -> Tuple[bool, str]:
        """Export metadata to JSON format."""
        export_data = {
            'audiobook_metadata': self.audiobook_metadata.to_dict(),
            'export_date': datetime.now().isoformat(),
            'export_format': 'json'
        }
        
        if include_processing:
            export_data['processing_metadata'] = self.processing_metadata.to_dict()
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return True, f"✅ Metadata exported to JSON: {export_path}"
    
    def _export_csv(self, export_path: Path, include_processing: bool) -> Tuple[bool, str]:
        """Export metadata to CSV format."""
        import csv
        
        # Flatten metadata for CSV
        flat_data = {}
        
        # Add audiobook metadata
        for key, value in self.audiobook_metadata.to_dict().items():
            if isinstance(value, (dict, list)):
                flat_data[key] = json.dumps(value)
            else:
                flat_data[key] = value
        
        # Add processing metadata if requested
        if include_processing:
            for key, value in self.processing_metadata.to_dict().items():
                if isinstance(value, (dict, list)):
                    flat_data[f"processing_{key}"] = json.dumps(value)
                else:
                    flat_data[f"processing_{key}"] = value
        
        # Write CSV
        with open(export_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=flat_data.keys())
            writer.writeheader()
            writer.writerow(flat_data)
        
        return True, f"✅ Metadata exported to CSV: {export_path}"
    
    def _export_xml(self, export_path: Path, include_processing: bool) -> Tuple[bool, str]:
        """Export metadata to XML format."""
        import xml.etree.ElementTree as ET
        
        # Create root element
        root = ET.Element("audiobook_metadata")
        
        # Add audiobook metadata
        audiobook_elem = ET.SubElement(root, "audiobook")
        for key, value in self.audiobook_metadata.to_dict().items():
            elem = ET.SubElement(audiobook_elem, key)
            if isinstance(value, (dict, list)):
                elem.text = json.dumps(value)
            else:
                elem.text = str(value)
        
        # Add processing metadata if requested
        if include_processing:
            processing_elem = ET.SubElement(root, "processing")
            for key, value in self.processing_metadata.to_dict().items():
                elem = ET.SubElement(processing_elem, key)
                if isinstance(value, (dict, list)):
                    elem.text = json.dumps(value)
                else:
                    elem.text = str(value)
        
        # Write XML
        tree = ET.ElementTree(root)
        tree.write(export_path, encoding='utf-8', xml_declaration=True)
        
        return True, f"✅ Metadata exported to XML: {export_path}"

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_metadata(title: str, author: str = "", **kwargs) -> AudiobookMetadata:
    """Create audiobook metadata (convenience function)."""
    manager = MetadataManager()
    return manager.create_new_metadata(title, author, **kwargs)

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Metadata Manager module loaded")
print("📋 Project metadata management ready for audiobook projects") 