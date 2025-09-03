#!/usr/bin/env python3
"""
Test script to verify TabDDPM integration
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from generators.tabddpm_generator import TabDDPMGenerator
        print("✓ TabDDPM generator imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import TabDDPM generator: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist"""
    required_dirs = [
        "tabddpm",
        "data/synthetic/tabddpm"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            all_exist = False
    
    return all_exist

def test_generator_initialization():
    """Test that TabDDPM generator can be initialized"""
    try:
        from generators.tabddpm_generator import TabDDPMGenerator
        
        generator = TabDDPMGenerator(
            output_dir="data/synthetic/tabddpm",
            num_experiments=1
        )
        print("✓ TabDDPM generator initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize TabDDPM generator: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing TabDDPM Integration")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Structure Test", test_directory_structure),
        ("Generator Initialization Test", test_generator_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} failed")
    
    print(f"\n{'=' * 40}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! TabDDPM integration is ready.")
    else:
        print("✗ Some tests failed. Please check the setup.")
        print("\nTo fix issues:")
        print("1. Run: ./setup_tabddpm.sh")
        print("2. Activate environment: conda activate tabddpm")
        print("3. Run this test again")

if __name__ == "__main__":
    main()
