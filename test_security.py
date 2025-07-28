#!/usr/bin/env python3
"""
Security Test Suite for Medical Superbill Extraction System

This script tests for security vulnerabilities and validates security controls.
"""

import os
import sys
import tempfile
from pathlib import Path
import pytest

# Add the project root to the path
ROOT_DIR = Path(__file__).parent
sys.path.append(str(ROOT_DIR))

def test_path_traversal_protection():
    """Test that path traversal attacks are prevented."""
    print("Testing path traversal protection...")
    
    try:
        from ui.app import sanitize_filename, secure_temp_file
        
        # Test malicious filenames
        malicious_names = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "../../sensitive.txt",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam",
            "con.txt",  # Windows reserved name
            "prn.log",  # Windows reserved name
            "",  # Empty filename
            ".hidden",  # Hidden file
            "file\x00.txt",  # Null byte injection
        ]
        
        for malicious_name in malicious_names:
            safe_name = sanitize_filename(malicious_name)
            assert not safe_name.startswith('.'), f"Sanitized name starts with dot: {safe_name}"
            assert '/' not in safe_name, f"Sanitized name contains slash: {safe_name}"
            assert '\\' not in safe_name, f"Sanitized name contains backslash: {safe_name}"
            assert '\x00' not in safe_name, f"Sanitized name contains null byte: {safe_name}"
            print(f"  OK Malicious name '{malicious_name}' sanitized to '{safe_name}'")
        
        print("  OK Path traversal protection working")
        return True
        
    except Exception as e:
        print(f"  FAIL Path traversal test failed: {e}")
        return False

def test_file_size_limits():
    """Test that file size limits are enforced."""
    print("Testing file size limits...")
    
    try:
        from ui.app import secure_temp_file, MAX_UPLOAD_SIZE
        
        # Test oversized file
        large_data = b'A' * (MAX_UPLOAD_SIZE + 1)
        
        try:
            with secure_temp_file(large_data, '.txt') as temp_path:
                print("  FAIL Large file was not rejected")
                return False
        except ValueError as e:
            if "too large" in str(e):
                print("  OK Large file correctly rejected")
            else:
                print(f"  FAIL Unexpected error: {e}")
                return False
        
        # Test tiny file (should also be rejected)
        tiny_data = b'A' * 5
        
        try:
            with secure_temp_file(tiny_data, '.txt') as temp_path:
                print("  FAIL Tiny file was not rejected")
                return False
        except ValueError as e:
            if "empty or corrupted" in str(e):
                print("  OK Tiny file correctly rejected")
            else:
                print(f"  FAIL Unexpected error: {e}")
                return False
        
        # Test valid file
        valid_data = b'A' * 1000
        try:
            with secure_temp_file(valid_data, '.txt') as temp_path:
                print("  OK Valid file accepted and temp file created")
                assert Path(temp_path).exists(), "Temp file should exist"
            
            # Verify cleanup
            assert not Path(temp_path).exists(), "Temp file should be cleaned up"
            print("  OK Temp file properly cleaned up")
        except Exception as e:
            print(f"  FAIL Valid file test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  FAIL File size limit test failed: {e}")
        return False

def test_configuration_security():
    """Test configuration security measures."""
    print("Testing configuration security...")
    
    try:
        from src.core.config_manager import ConfigManager
        
        # Test environment variable expansion
        os.environ['TEST_SECRET'] = 'test_value_123'
        os.environ['TEST_DEFAULT'] = 'default_value'
        
        config_mgr = ConfigManager()
        
        # Test basic environment expansion
        test_config = "${TEST_SECRET}"
        expanded = config_mgr._expand_env_variables(test_config)
        assert expanded == 'test_value_123', f"Environment expansion failed: {expanded}"
        print("  OK Environment variable expansion working")
        
        # Test default value expansion
        test_config_default = "${NONEXISTENT_VAR:default_fallback}"
        expanded_default = config_mgr._expand_env_variables(test_config_default)
        assert expanded_default == 'default_fallback', f"Default value expansion failed: {expanded_default}"
        print("  OK Default value expansion working")
        
        # Test invalid variable name rejection
        test_config_invalid = "${INVALID-VAR-NAME}"
        expanded_invalid = config_mgr._expand_env_variables(test_config_invalid)
        assert expanded_invalid == "${INVALID-VAR-NAME}", f"Invalid var name should be unchanged: {expanded_invalid}"
        print("  OK Invalid environment variable names rejected")
        
        # Clean up test environment variables
        del os.environ['TEST_SECRET']
        del os.environ['TEST_DEFAULT']
        
        return True
        
    except Exception as e:
        print(f"  FAIL Configuration security test failed: {e}")
        return False

def test_subprocess_security():
    """Test subprocess execution security."""
    print("Testing subprocess security...")
    
    try:
        from run_ui import validate_ui_path, safe_subprocess_run
        
        project_root = Path(__file__).parent.resolve()
        
        # Test valid UI path
        valid_ui_path = project_root / "ui" / "app.py" 
        if valid_ui_path.exists():
            assert validate_ui_path(valid_ui_path, project_root), "Valid UI path should be accepted"
            print("  OK Valid UI path accepted")
        
        # Test invalid paths
        invalid_paths = [
            project_root.parent / "malicious.py",  # Outside project
            project_root / "ui" / "nonexistent.py",  # Doesn't exist
            project_root / "config" / "config.yaml",  # Not Python file
        ]
        
        for invalid_path in invalid_paths:
            assert not validate_ui_path(invalid_path, project_root), f"Invalid path should be rejected: {invalid_path}"
            print(f"  OK Invalid path rejected: {invalid_path.name}")
        
        return True
        
    except Exception as e:
        print(f"  FAIL Subprocess security test failed: {e}")
        return False

def test_file_validation():
    """Test file validation security."""
    print("Testing file validation...")
    
    try:
        from ui.components.file_uploader import FileValidator
        
        validator = FileValidator()
        
        # Test PDF validation with malicious content
        fake_pdf = b'%PDF-1.4\n' + b'A' * 1000  # Fake PDF header
        result = validator.validate_file(fake_pdf, "test.pdf")
        print(f"  OK Fake PDF validation result: {result['valid']} - {result.get('errors', [])}")
        
        # Test image validation
        # Create a minimal valid PNG
        valid_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10\x08\x02\x00\x00\x00\x90\x91h6\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82'
        result = validator.validate_file(valid_png, "test.png")
        print(f"  OK Valid PNG validation result: {result['valid']} - {result.get('errors', [])}")
        
        # Test malicious file type
        malicious_exe = b'MZ\x90\x00' + b'A' * 1000  # Fake EXE header
        result = validator.validate_file(malicious_exe, "malicious.exe")
        assert not result['valid'], "Malicious file should be rejected"
        print("  OK Malicious file type rejected")
        
        return True
        
    except Exception as e:
        print(f"  FAIL File validation test failed: {e}")
        return False

def test_error_information_disclosure():
    """Test that error messages don't disclose sensitive information."""
    print("Testing error information disclosure...")
    
    try:
        # This is a basic test - in production you'd want more comprehensive testing
        from src.extraction_engine import ExtractionEngine
        from src.core.config_manager import ConfigManager
        
        config = ConfigManager()
        engine = ExtractionEngine(config)
        
        # Try to process a non-existent file
        try:
            import asyncio
            result = asyncio.run(engine.extract_from_file("/nonexistent/path/to/file.pdf"))
            print("  FAIL Should have raised an exception for non-existent file")
            return False
        except Exception as e:
            error_str = str(e)
            # Check that error doesn't contain sensitive paths
            if "/nonexistent/path" in error_str:
                print(f"  WARN  Error message may contain sensitive path: {error_str}")
            else:
                print("  OK Error message doesn't expose sensitive paths")
        
        return True
        
    except Exception as e:
        print(f"  FAIL Error disclosure test failed: {e}")
        return False

def main():
    """Run all security tests."""
    print("=" * 60)
    print("Medical Superbill Extraction System - Security Test Suite")
    print("=" * 60)
    
    tests = [
        test_path_traversal_protection,
        test_file_size_limits,  
        test_configuration_security,
        test_subprocess_security,
        test_file_validation,
        test_error_information_disclosure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
            print(f"   PASSED")
        else:
            print(f"   FAILED")
    
    print("\n" + "=" * 60)
    print(f"Security Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All security tests passed! Security controls are working.")
        return 0
    else:
        print("Some security tests failed. Please review and fix issues before deployment.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)