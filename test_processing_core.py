#!/usr/bin/env python3
"""Minimal test for TCP processing core infrastructure."""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_path_configuration():
    """Test basic path configuration."""
    print("=== Testing Path Configuration ===")
    
    try:
        from config.paths import get_tcp_processing_path, get_platform_info
        
        processing_path = get_tcp_processing_path()
        platform_info = get_platform_info()
        
        print(f"✓ Path configuration working")
        print(f"  Processing path: {processing_path}")
        print(f"  Platform: {platform_info['detected_platform']}")
        print(f"  Hostname: {platform_info['hostname']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Path configuration failed: {e}")
        return False

def test_processing_config():
    """Test ProcessingConfig without pandas dependencies."""
    print("\n=== Testing ProcessingConfig (Core) ===")
    
    try:
        from tcp.processing.config.processing_config import ProcessingConfig
        
        # Test basic initialization
        config = ProcessingConfig()
        
        print(f"✓ ProcessingConfig initialized")
        print(f"  Dataset path: {config.get_dataset_path()}")
        print(f"  Processing output path: {config.get_processing_output_path()}")
        print(f"  Manifest path: {config.get_manifest_path()}")
        
        # Test validation
        validation = config.validate_configuration()
        print(f"  Configuration validation: {validation}")
        
        # Test platform settings
        platform_settings = config.get_platform_specific_settings()
        print(f"  Platform: {platform_settings['platform']}")
        
        return True
        
    except Exception as e:
        print(f"❌ ProcessingConfig failed: {e}")
        return False

def test_manifest_validation():
    """Test manifest validation utilities."""
    print("\n=== Testing Manifest Validation ===")
    
    try:
        from tcp.processing.utils.validation import validate_manifest
        from tcp.processing.config.processing_config import ProcessingConfig
        
        config = ProcessingConfig()
        manifest_path = config.get_manifest_path()
        
        print(f"Manifest path: {manifest_path}")
        print(f"Manifest exists: {manifest_path.exists()}")
        
        if manifest_path.exists():
            manifest = validate_manifest(manifest_path)
            print(f"✓ Manifest validation successful")
            
            metadata = manifest.get('manifest_metadata', {})
            print(f"  Total subjects: {metadata.get('total_subjects', 'unknown')}")
            print(f"  Analysis groups: {metadata.get('analysis_groups', 'unknown')}")
            
            return True
        else:
            print("⚠️  Manifest not found - run preprocessing pipeline first")
            print("   This is expected for a fresh setup")
            return True  # Not a failure, just missing data
            
    except Exception as e:
        print(f"❌ Manifest validation failed: {e}")
        return False

def main():
    """Run core infrastructure tests."""
    print("TCP Processing Core Infrastructure Test")
    print("=" * 45)
    
    tests = [
        test_path_configuration,
        test_processing_config,
        test_manifest_validation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n{'=' * 45}")
    print("TEST SUMMARY")
    print(f"{'=' * 45}")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 Core infrastructure ready!")
        print("\nNext steps:")
        print("1. Run preprocessing pipeline to generate data manifest")
        print("2. Install pandas for full DataLoader/SubjectManager testing")
        print("3. Use conda environment: 'conda activate tcp_env'")
        
        return 0
    else:
        print("⚠️  Some core components failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())