#!/usr/bin/env python3
"""
Phase 2 Audio Processing Pipeline Validation Script
==================================================

This script validates all 4 audio processing modules:
1. Playback Engine - Master continuous audio creation
2. Effects Processor - Professional broadcast-quality processing  
3. Quality Analyzer - Volume normalization and standards validation
4. Enhancement Tools - Mastering-grade finishing

Tests the complete audio processing pipeline functionality.
"""

import sys
import numpy as np
from pathlib import Path

# Add refactor path to import the new modules
sys.path.insert(0, str(Path(__file__).parent / "refactor"))

def test_phase2_audio_modules():
    """Test all Phase 2 audio processing modules."""
    print("🎵" + "="*60)
    print("🎵 PHASE 2 AUDIO PROCESSING PIPELINE VALIDATION")
    print("🎵" + "="*60)
    
    results = {}
    
    # Test 1: Playback Engine
    print("\n1️⃣ Testing Playback Engine...")
    try:
        from audio.playback_engine import PlaybackEngine, create_continuous_audio
        
        # Test basic initialization
        engine = PlaybackEngine()
        
        # Test chunk loading
        test_chunks = [
            {
                'chunk_id': 'test_chunk_1',
                'text': 'This is a test chunk for validation.',
                'page_number': 1,
                'voice_profile': 'test_voice'
            }
        ]
        
        loaded_count, message = engine.load_audio_chunks(test_chunks)
        print(f"   ✅ Playback Engine loaded {loaded_count} chunks: {message}")
        
        # Test playback controls
        engine.start_playback()
        engine.pause_playback()
        engine.stop_playback()
        print("   ✅ Playback controls working")
        
        # Test playback info
        info = engine.get_playback_info()
        print(f"   ✅ Playback info: {len(info)} fields available")
        
        results['playback_engine'] = True
        
    except Exception as e:
        print(f"   ❌ Playback Engine failed: {e}")
        results['playback_engine'] = False
    
    # Test 2: Effects Processor
    print("\n2️⃣ Testing Effects Processor...")
    try:
        from audio.effects_processor import EffectsProcessor, enhance_audio
        
        # Test initialization
        processor = EffectsProcessor()
        
        # Create test audio data
        sample_rate = 22050
        duration = 2.0  # 2 seconds
        test_audio = np.random.normal(0, 0.1, int(sample_rate * duration)).astype(np.float32)
        
        # Test enhancement
        enhanced_audio, metadata = processor.enhance_audio(test_audio, sample_rate, "audiobook")
        print(f"   ✅ Audio enhancement completed: {len(metadata['processing_steps'])} steps")
        
        # Test noise reduction
        processed_audio, nr_metadata = processor.apply_noise_reduction(test_audio, sample_rate)
        print(f"   ✅ Noise reduction: {nr_metadata['success']}")
        
        # Test compression
        compressed_audio, comp_metadata = processor.apply_compression(test_audio, sample_rate)
        print(f"   ✅ Compression: {comp_metadata['success']}")
        
        # Test EQ
        eq_audio, eq_metadata = processor.apply_eq(test_audio, sample_rate)
        print(f"   ✅ EQ processing: {eq_metadata['success']}")
        
        # Test normalization
        normalized_audio, norm_metadata = processor.normalize_to_standard(test_audio, sample_rate)
        print(f"   ✅ Normalization: {norm_metadata['success']}")
        
        # Test trim with autosave
        trimmed_audio, save_message = processor.trim_audio_with_autosave(
            test_audio, sample_rate, 0.5, 1.5, "test_trim.wav"
        )
        print(f"   ✅ Trim with autosave: {len(save_message)} chars")
        
        results['effects_processor'] = True
        
    except Exception as e:
        print(f"   ❌ Effects Processor failed: {e}")
        results['effects_processor'] = False
    
    # Test 3: Quality Analyzer
    print("\n3️⃣ Testing Quality Analyzer...")
    try:
        from audio.quality_analyzer import QualityAnalyzer, analyze_audio_quality, normalize_volume
        
        # Test initialization
        analyzer = QualityAnalyzer()
        
        # Test quality analysis
        metrics = analyzer.analyze_audio_quality(test_audio, sample_rate, detailed=True)
        print(f"   ✅ Quality analysis completed: Peak={metrics.peak_db:.1f}dB, RMS={metrics.rms_db:.1f}dB")
        
        # Test LUFS measurement
        lufs_data = analyzer.measure_lufs(test_audio, sample_rate)
        print(f"   ✅ LUFS measurement: Integrated={lufs_data['integrated']:.1f}")
        
        # Test broadcast standards validation
        validation = analyzer.validate_broadcast_standards(metrics, "acx")
        print(f"   ✅ ACX validation: {validation['overall_pass']}, Score={validation['compliance_score']:.1f}%")
        
        # Test normalization recommendations
        recommendations = analyzer.recommend_normalization(metrics, "acx")
        print(f"   ✅ Normalization recommendations: {len(recommendations['adjustments_needed'])} adjustments")
        
        # Test volume normalization
        normalized_audio = normalize_volume(test_audio, sample_rate, -23.0)
        print(f"   ✅ Volume normalization: Output shape {normalized_audio.shape}")
        
        results['quality_analyzer'] = True
        
    except Exception as e:
        print(f"   ❌ Quality Analyzer failed: {e}")
        results['quality_analyzer'] = False
    
    # Test 4: Enhancement Tools
    print("\n4️⃣ Testing Enhancement Tools...")
    try:
        from audio.enhancement_tools import EnhancementTools, enhance_audio_master
        
        # Test initialization
        tools = EnhancementTools()
        
        # Test master enhancement
        enhanced_audio, metadata = tools.master_enhance_audio(test_audio, sample_rate, "audiobook_master")
        print(f"   ✅ Master enhancement: {len(metadata['enhancement_steps'])} steps, Success={metadata['success']}")
        
        # Test spectral repair
        repaired_audio, repair_metadata = tools.repair_spectral_artifacts(test_audio, sample_rate)
        print(f"   ✅ Spectral repair: {repair_metadata['success']}")
        
        # Test breath reduction
        breath_reduced_audio, breath_metadata = tools.reduce_breath_sounds(test_audio, sample_rate)
        print(f"   ✅ Breath reduction: {breath_metadata['success']}")
        
        # Test plosive reduction
        plosive_reduced_audio, plosive_metadata = tools.reduce_plosives(test_audio, sample_rate)
        print(f"   ✅ Plosive reduction: {plosive_metadata['success']}")
        
        # Test harmonic enhancement
        harmonic_enhanced_audio, harmonic_metadata = tools.enhance_voice_harmonics(test_audio, sample_rate)
        print(f"   ✅ Harmonic enhancement: {harmonic_metadata['success']}")
        
        # Test consistency processing
        consistent_audio, consistency_metadata = tools.apply_consistency_processing(test_audio, sample_rate)
        print(f"   ✅ Consistency processing: {consistency_metadata['success']}")
        
        results['enhancement_tools'] = True
        
    except Exception as e:
        print(f"   ❌ Enhancement Tools failed: {e}")
        results['enhancement_tools'] = False
    
    # Test 5: Module Integration
    print("\n5️⃣ Testing Module Integration...")
    try:
        # Test importing the audio package
        import audio
        print(f"   ✅ Audio package imported successfully")
        
        # Test global convenience functions
        test_enhanced = enhance_audio(test_audio, sample_rate, "audiobook")
        print(f"   ✅ Global enhance_audio function working")
        
        test_quality = analyze_audio_quality(test_audio, sample_rate)
        print(f"   ✅ Global analyze_audio_quality function working")
        
        test_master = enhance_audio_master(test_audio, sample_rate)
        print(f"   ✅ Global enhance_audio_master function working")
        
        results['module_integration'] = True
        
    except Exception as e:
        print(f"   ❌ Module Integration failed: {e}")
        results['module_integration'] = False
    
    # Summary
    print("\n🎵" + "="*60)
    print("🎵 PHASE 2 VALIDATION RESULTS")
    print("🎵" + "="*60)
    
    total_modules = len(results)
    passed_modules = sum(results.values())
    
    for module, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {module.replace('_', ' ').title()}")
    
    print(f"\n🎯 PHASE 2 COMPLETION: {passed_modules}/{total_modules} modules operational")
    
    if passed_modules == total_modules:
        print("🎉 ALL PHASE 2 AUDIO MODULES OPERATIONAL! 🎉")
        print("🚀 Ready to proceed to Phase 3!")
        return True
    else:
        print(f"⚠️  {total_modules - passed_modules} modules need attention")
        return False

if __name__ == "__main__":
    success = test_phase2_audio_modules()
    sys.exit(0 if success else 1) 