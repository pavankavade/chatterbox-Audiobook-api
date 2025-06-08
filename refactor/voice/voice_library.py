"""
# ==============================================================================
# VOICE LIBRARY MODULE
# ==============================================================================
# 
# This module provides comprehensive voice library management for the Chatterbox
# Audiobook Studio refactored system. It handles voice discovery, organization,
# import/export operations, and library maintenance.
# 
# **Key Features:**
# - **Library Discovery**: Automatic voice profile scanning and indexing
# - **Import/Export**: Voice library backup and restoration
# - **Organization**: Voice categorization and search capabilities
# - **Maintenance**: Library validation and cleanup operations
# - **Original Compatibility**: Full compatibility with existing voice libraries
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from datetime import datetime

# Import voice management
from .voice_manager import VoiceManager, VoiceProfile

# Import configuration
from config.settings import get_voice_library_path

# ==============================================================================
# VOICE LIBRARY CLASS
# ==============================================================================

class VoiceLibrary:
    """
    Professional voice library management system.
    
    This class provides comprehensive library-level operations for voice
    management including discovery, organization, and maintenance.
    
    **Library Features:**
    - **Auto-Discovery**: Automatic voice profile scanning
    - **Import/Export**: Complete library backup and restoration
    - **Validation**: Library integrity checking and repair
    - **Search**: Advanced voice profile search and filtering
    - **Statistics**: Comprehensive library analytics
    """
    
    def __init__(self, voice_library_path: Optional[str] = None):
        """
        Initialize the voice library manager.
        
        Args:
            voice_library_path (Optional[str]): Path to voice library directory
        """
        self.voice_library_path = voice_library_path or get_voice_library_path()
        self.voice_manager = VoiceManager(self.voice_library_path)
        print(f"✅ Voice Library initialized - Path: {self.voice_library_path}")
    
    def discover_voices(self, scan_subdirectories: bool = True) -> Dict[str, Any]:
        """
        Discover and catalog all voices in the library.
        
        Args:
            scan_subdirectories (bool): Whether to scan subdirectories
            
        Returns:
            Dict[str, Any]: Complete discovery results
            
        **Discovery Results:**
        - **total_voices**: Total number of voice profiles found
        - **valid_voices**: Number of valid voice profiles
        - **invalid_voices**: Number of invalid voice profiles
        - **voice_profiles**: List of all valid voice profiles
        - **errors**: List of errors encountered during discovery
        """
        results = {
            'total_voices': 0,
            'valid_voices': 0,
            'invalid_voices': 0,
            'voice_profiles': [],
            'errors': [],
            'scan_time': datetime.now().isoformat()
        }
        
        library_path = Path(self.voice_library_path)
        
        if not library_path.exists():
            results['errors'].append(f"Voice library path does not exist: {library_path}")
            return results
        
        try:
            # Scan for voice directories
            voice_dirs = []
            
            if scan_subdirectories:
                # Recursive scan
                for item in library_path.rglob("*/"):
                    if item.is_dir() and (item / "config.json").exists():
                        voice_dirs.append(item)
            else:
                # Direct children only
                for item in library_path.iterdir():
                    if item.is_dir() and (item / "config.json").exists():
                        voice_dirs.append(item)
            
            results['total_voices'] = len(voice_dirs)
            
            # Process each voice directory
            for voice_dir in voice_dirs:
                try:
                    # Load voice profile
                    profile, message = self.voice_manager.load_voice_profile(voice_dir.name)
                    
                    if profile:
                        results['valid_voices'] += 1
                        results['voice_profiles'].append({
                            'name': profile.voice_name,
                            'display_name': profile.display_name,
                            'description': profile.description,
                            'audio_file': profile.audio_file,
                            'directory': str(voice_dir),
                            'exaggeration': profile.exaggeration,
                            'cfg_weight': profile.cfg_weight,
                            'temperature': profile.temperature
                        })
                    else:
                        results['invalid_voices'] += 1
                        results['errors'].append(f"Invalid voice in {voice_dir}: {message}")
                        
                except Exception as e:
                    results['invalid_voices'] += 1
                    results['errors'].append(f"Error processing {voice_dir}: {str(e)}")
            
        except Exception as e:
            results['errors'].append(f"Error during library discovery: {str(e)}")
        
        return results
    
    def search_voices(
        self,
        query: str = "",
        search_fields: List[str] = None,
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search voice profiles based on query criteria.
        
        Args:
            query (str): Search query string
            search_fields (List[str]): Fields to search in ['name', 'display_name', 'description']
            case_sensitive (bool): Whether search is case sensitive
            
        Returns:
            List[Dict[str, Any]]: Matching voice profiles
        """
        if search_fields is None:
            search_fields = ['name', 'display_name', 'description']
        
        discovery = self.discover_voices()
        all_voices = discovery['voice_profiles']
        
        if not query.strip():
            return all_voices
        
        # Prepare search query
        search_query = query if case_sensitive else query.lower()
        
        matching_voices = []
        
        for voice in all_voices:
            match_found = False
            
            for field in search_fields:
                if field in voice:
                    field_value = str(voice[field])
                    if not case_sensitive:
                        field_value = field_value.lower()
                    
                    if search_query in field_value:
                        match_found = True
                        break
            
            if match_found:
                matching_voices.append(voice)
        
        return matching_voices
    
    def get_library_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive library statistics.
        
        Returns:
            Dict[str, Any]: Complete library statistics
        """
        discovery = self.discover_voices()
        
        stats = {
            'total_voices': discovery['total_voices'],
            'valid_voices': discovery['valid_voices'],
            'invalid_voices': discovery['invalid_voices'],
            'error_count': len(discovery['errors']),
            'library_path': self.voice_library_path,
            'scan_time': discovery['scan_time']
        }
        
        # Calculate additional statistics
        if discovery['voice_profiles']:
            voices = discovery['voice_profiles']
            
            # Voice parameter statistics
            exaggerations = [v['exaggeration'] for v in voices if 'exaggeration' in v]
            cfg_weights = [v['cfg_weight'] for v in voices if 'cfg_weight' in v]
            temperatures = [v['temperature'] for v in voices if 'temperature' in v]
            
            if exaggerations:
                stats['avg_exaggeration'] = sum(exaggerations) / len(exaggerations)
                stats['min_exaggeration'] = min(exaggerations)
                stats['max_exaggeration'] = max(exaggerations)
            
            if cfg_weights:
                stats['avg_cfg_weight'] = sum(cfg_weights) / len(cfg_weights)
                stats['min_cfg_weight'] = min(cfg_weights)
                stats['max_cfg_weight'] = max(cfg_weights)
            
            if temperatures:
                stats['avg_temperature'] = sum(temperatures) / len(temperatures)
                stats['min_temperature'] = min(temperatures)
                stats['max_temperature'] = max(temperatures)
        
        return stats
    
    def export_library(self, export_path: Union[str, Path], include_audio: bool = True) -> str:
        """
        Export the voice library to a backup file.
        
        Args:
            export_path (Union[str, Path]): Path for the export file
            include_audio (bool): Whether to include audio files
            
        Returns:
            str: Success or error message
        """
        try:
            export_path = Path(export_path)
            
            # Ensure export directory exists
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create zip archive
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                library_path = Path(self.voice_library_path)
                
                if not library_path.exists():
                    return "❌ Voice library path does not exist"
                
                # Add library metadata
                metadata = {
                    'export_time': datetime.now().isoformat(),
                    'library_path': str(library_path),
                    'include_audio': include_audio,
                    'statistics': self.get_library_statistics()
                }
                
                zip_file.writestr('metadata.json', json.dumps(metadata, indent=2))
                
                # Add voice profiles
                for voice_dir in library_path.iterdir():
                    if voice_dir.is_dir() and (voice_dir / "config.json").exists():
                        # Add config.json
                        config_file = voice_dir / "config.json"
                        arc_name = f"voices/{voice_dir.name}/config.json"
                        zip_file.write(config_file, arc_name)
                        
                        # Add audio file if requested
                        if include_audio:
                            audio_file = voice_dir / "voice.wav"
                            if audio_file.exists():
                                arc_name = f"voices/{voice_dir.name}/voice.wav"
                                zip_file.write(audio_file, arc_name)
            
            return f"✅ Voice library exported to: {export_path}"
            
        except Exception as e:
            return f"❌ Error exporting voice library: {str(e)}"
    
    def import_library(
        self,
        import_path: Union[str, Path],
        merge_mode: str = "skip_existing"
    ) -> str:
        """
        Import a voice library from a backup file.
        
        Args:
            import_path (Union[str, Path]): Path to the import file
            merge_mode (str): How to handle existing voices ("skip_existing", "overwrite", "rename")
            
        Returns:
            str: Success or error message
        """
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                return f"❌ Import file not found: {import_path}"
            
            imported_count = 0
            skipped_count = 0
            error_count = 0
            
            with zipfile.ZipFile(import_path, 'r') as zip_file:
                # Read metadata
                try:
                    metadata_content = zip_file.read('metadata.json')
                    metadata = json.loads(metadata_content)
                    print(f"📦 Importing library exported on: {metadata.get('export_time', 'Unknown')}")
                except:
                    print("⚠️  No metadata found in import file")
                
                # Get voice entries
                voice_entries = [name for name in zip_file.namelist() if name.startswith('voices/') and name.endswith('/config.json')]
                
                for config_entry in voice_entries:
                    try:
                        # Extract voice name
                        voice_name = config_entry.split('/')[1]
                        
                        # Check if voice already exists
                        existing_voice_dir = Path(self.voice_library_path) / voice_name
                        
                        if existing_voice_dir.exists():
                            if merge_mode == "skip_existing":
                                skipped_count += 1
                                continue
                            elif merge_mode == "rename":
                                # Find unique name
                                counter = 1
                                while (Path(self.voice_library_path) / f"{voice_name}_{counter}").exists():
                                    counter += 1
                                voice_name = f"{voice_name}_{counter}"
                                existing_voice_dir = Path(self.voice_library_path) / voice_name
                        
                        # Create voice directory
                        existing_voice_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Extract config.json
                        config_content = zip_file.read(config_entry)
                        with open(existing_voice_dir / "config.json", 'wb') as f:
                            f.write(config_content)
                        
                        # Extract audio file if present
                        audio_entry = f"voices/{config_entry.split('/')[1]}/voice.wav"
                        if audio_entry in zip_file.namelist():
                            audio_content = zip_file.read(audio_entry)
                            with open(existing_voice_dir / "voice.wav", 'wb') as f:
                                f.write(audio_content)
                        
                        imported_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        print(f"❌ Error importing voice {config_entry}: {str(e)}")
            
            return f"✅ Import complete: {imported_count} imported, {skipped_count} skipped, {error_count} errors"
            
        except Exception as e:
            return f"❌ Error importing voice library: {str(e)}"
    
    def validate_library(self, fix_errors: bool = False) -> Dict[str, Any]:
        """
        Validate the voice library and optionally fix errors.
        
        Args:
            fix_errors (bool): Whether to attempt automatic error fixes
            
        Returns:
            Dict[str, Any]: Validation results
        """
        results = {
            'total_checked': 0,
            'valid_voices': 0,
            'invalid_voices': 0,
            'fixed_voices': 0,
            'errors': [],
            'fixes_applied': [],
            'validation_time': datetime.now().isoformat()
        }
        
        discovery = self.discover_voices()
        results['total_checked'] = discovery['total_voices']
        results['valid_voices'] = discovery['valid_voices']
        results['invalid_voices'] = discovery['invalid_voices']
        results['errors'] = discovery['errors']
        
        if fix_errors and results['invalid_voices'] > 0:
            # Attempt to fix common issues
            library_path = Path(self.voice_library_path)
            
            for voice_dir in library_path.iterdir():
                if voice_dir.is_dir():
                    try:
                        # Check for missing config.json
                        config_file = voice_dir / "config.json"
                        if not config_file.exists():
                            # Try to create minimal config
                            audio_files = list(voice_dir.glob("*.wav"))
                            if audio_files:
                                minimal_config = {
                                    'voice_name': voice_dir.name,
                                    'display_name': voice_dir.name,
                                    'description': f'Auto-generated config for {voice_dir.name}',
                                    'audio_file': str(audio_files[0]),
                                    'exaggeration': 1.0,
                                    'cfg_weight': 3.0,
                                    'temperature': 0.7,
                                    'enable_normalization': False,
                                    'target_level_db': -20.0
                                }
                                
                                with open(config_file, 'w', encoding='utf-8') as f:
                                    json.dump(minimal_config, f, indent=2)
                                
                                results['fixed_voices'] += 1
                                results['fixes_applied'].append(f"Created config.json for {voice_dir.name}")
                    
                    except Exception as e:
                        results['errors'].append(f"Error fixing {voice_dir}: {str(e)}")
        
        return results
    
    def cleanup_library(self, remove_orphaned: bool = True, remove_empty: bool = True) -> Dict[str, Any]:
        """
        Clean up the voice library by removing orphaned and empty entries.
        
        Args:
            remove_orphaned (bool): Remove directories without config.json
            remove_empty (bool): Remove empty directories
            
        Returns:
            Dict[str, Any]: Cleanup results
        """
        results = {
            'directories_removed': 0,
            'files_removed': 0,
            'errors': [],
            'cleanup_time': datetime.now().isoformat()
        }
        
        try:
            library_path = Path(self.voice_library_path)
            
            if not library_path.exists():
                results['errors'].append("Voice library path does not exist")
                return results
            
            for item in library_path.iterdir():
                if item.is_dir():
                    try:
                        config_file = item / "config.json"
                        
                        # Remove orphaned directories (no config.json)
                        if remove_orphaned and not config_file.exists():
                            shutil.rmtree(item)
                            results['directories_removed'] += 1
                            continue
                        
                        # Remove empty directories
                        if remove_empty and not any(item.iterdir()):
                            item.rmdir()
                            results['directories_removed'] += 1
                    
                    except Exception as e:
                        results['errors'].append(f"Error processing {item}: {str(e)}")
        
        except Exception as e:
            results['errors'].append(f"Error during cleanup: {str(e)}")
        
        return results

# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def get_global_voice_library() -> VoiceLibrary:
    """Get a global voice library instance."""
    return VoiceLibrary()

# ==============================================================================
# MODULE INITIALIZATION
# ==============================================================================

print("✅ Voice Library module loaded")
print("📚 Voice library management ready for discovery and organization") 