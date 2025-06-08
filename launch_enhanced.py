#!/usr/bin/env python3
"""
Simple launcher for the Enhanced Chatterbox Audiobook Studio
"""

import sys
import os
from pathlib import Path

# Add refactor path
sys.path.insert(0, str(Path(__file__).parent / "refactor"))

def main():
    print("🚀 Launching Enhanced Chatterbox Audiobook Studio...")
    print("🎨 Phase 3: Professional UI with integrated audio processing")
    
    try:
        # Import and run the enhanced app
        from enhanced_gradio_app import main as enhanced_main
        enhanced_main()
    except Exception as e:
        print(f"❌ Error launching enhanced app: {e}")
        print("📝 Trying fallback approach...")
        
        # Fallback - direct execution
        import subprocess
        subprocess.run([sys.executable, "refactor/enhanced_gradio_app.py"])

if __name__ == "__main__":
    main() 