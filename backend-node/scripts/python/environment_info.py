#!/usr/bin/env python3
"""
Python Environment Information Script
Provides information about the Python environment and installed packages
"""

import sys
import json
import platform
import subprocess
import importlib.util

def get_package_version(package_name):
    """Get version of an installed package"""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return None
        
        module = importlib.import_module(package_name)
        return getattr(module, '__version__', 'unknown')
    except Exception:
        return None

def check_package_availability(package_name):
    """Check if a package is available"""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def main():
    """Main execution function"""
    try:
        # System information
        system_info = {
            'python_version': sys.version,
            'python_executable': sys.executable,
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_path': sys.path
        }
        
        # Check for required ML packages
        ml_packages = {
            'numpy': get_package_version('numpy'),
            'pandas': get_package_version('pandas'),
            'scikit-learn': get_package_version('sklearn'),
            'scipy': get_package_version('scipy'),
            'matplotlib': get_package_version('matplotlib'),
            'seaborn': get_package_version('seaborn')
        }
        
        # Check package availability
        package_availability = {}
        for package in ml_packages.keys():
            package_availability[package] = check_package_availability(package)
        
        # Get pip list (if available)
        pip_packages = {}
        try:
            result = subprocess.run([sys.executable, '-m', 'pip', 'list', '--format=json'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                pip_packages = json.loads(result.stdout)
        except Exception:
            pip_packages = {'error': 'Could not retrieve pip packages'}
        
        result = {
            'success': True,
            'system_info': system_info,
            'ml_packages': ml_packages,
            'package_availability': package_availability,
            'pip_packages': pip_packages,
            'total_packages': len(pip_packages) if isinstance(pip_packages, list) else 0
        }
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_result))

if __name__ == '__main__':
    main()
