#!/usr/bin/env python3
"""Simple test for TCP processing infrastructure."""

from pathlib import Path

# Test path configuration
try:
    from config.paths import get_tcp_processing_path, get_platform_info
    print("✓ Path configuration working")
    print(f"  Processing path: {get_tcp_processing_path()}")
    
    platform_info = get_platform_info()
    print(f"  Platform: {platform_info['detected_platform']}")
except Exception as e:
    print(f"❌ Path configuration failed: {e}")

# Test processing config
try:
    from tcp.processing.config.processing_config import ProcessingConfig
    config = ProcessingConfig()
    print("✓ ProcessingConfig working")
    print(f"  Dataset path: {config.get_dataset_path()}")
    print(f"  Manifest path: {config.get_manifest_path()}")
except Exception as e:
    print(f"❌ ProcessingConfig failed: {e}")

# Test data loader
try:
    from tcp.processing.data_loader import DataLoader
    
    # Check if manifest exists
    config = ProcessingConfig()
    manifest_path = config.get_manifest_path()
    
    if manifest_path.exists():
        loader = DataLoader(config=config)
        print("✓ DataLoader working")
        
        info = loader.get_manifest_info()
        print(f"  Total subjects: {info['total_subjects']}")
        print(f"  Analysis groups: {info['analysis_groups']}")
    else:
        print("⚠️  DataLoader test skipped - manifest not found")
        print(f"    Expected at: {manifest_path}")
        print("    Run preprocessing pipeline first to generate manifest")
        
except Exception as e:
    print(f"❌ DataLoader failed: {e}")

print("\nTest complete!")