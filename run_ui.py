#!/usr/bin/env python
"""
Simple UI Execution File for Medical Superbill Extractor
Run this file to start the Streamlit UI application.

Features:
- Nanonets-OCR for text extraction  
- NuExtract-2.0-8B for structured data extraction
- No fallback models - clean implementation
"""

import subprocess
import sys
import os
from pathlib import Path

def validate_ui_path(ui_path: Path, project_root: Path) -> bool:
    """Validate UI app path is secure and within project directory."""
    try:
        # Resolve paths to absolute paths
        ui_resolved = ui_path.resolve()
        root_resolved = project_root.resolve()
        
        # Check if UI path is within project directory
        ui_resolved.relative_to(root_resolved)
        
        # Check if file exists and is a Python file
        if not ui_resolved.exists() or not ui_resolved.is_file():
            return False
        
        # Check file extension for additional safety
        if ui_resolved.suffix != '.py':
            return False
            
        return True
    except (ValueError, OSError):
        return False

def safe_subprocess_run(ui_app_path: Path):
    """Safely execute streamlit with validated parameters."""
    project_root = Path(__file__).parent.resolve()
    
    # Validate UI app path
    if not validate_ui_path(ui_app_path, project_root):
        raise ValueError("Invalid or unsafe UI app path")
    
    # Use only validated, absolute paths and hardcoded safe parameters
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(ui_app_path.resolve()),
        "--theme.base", "light",
        "--theme.primaryColor", "#1f77b4", 
        "--theme.backgroundColor", "#ffffff",
        "--server.port", "8503",
        "--server.address", "localhost",
        "--server.headless", "true",  # Additional security
        "--server.enableCORS", "false"  # Disable CORS for security
    ]
    
    # Execute with security considerations
    subprocess.run(cmd, check=True, cwd=project_root)

def main():
    """Run the Streamlit UI application."""
    print("Medical Superbill Extractor - Starting UI...")
    print("=" * 50)
    
    try:
        # Get the directory of this script securely
        current_dir = Path(__file__).parent.resolve()
        ui_app_path = current_dir / "ui" / "app.py"
        
        print(f"Starting Streamlit on http://localhost:8503")
        print(f"Using models: Nanonets-OCR + NuExtract-2.0-8B")
        print(f"Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run with security validation
        safe_subprocess_run(ui_app_path)
        
    except ValueError as e:
        print(f"Security validation failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Streamlit not found. Please install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
