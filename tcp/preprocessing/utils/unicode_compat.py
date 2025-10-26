#!/usr/bin/env python3
"""
Cross-Platform Unicode Compatibility Module

Provides cross-platform compatible symbols and text formatting for console output.
Automatically detects system encoding capabilities and provides appropriate fallbacks.

This module ensures that TCP preprocessing scripts work correctly on:
- Windows 11 (cp1252 encoding)
- macOS (UTF-8 encoding) 
- CentOS/Linux (UTF-8 encoding)

Author: Ian Philip Eglin
Date: 2025-09-25
"""

import sys
import platform
from typing import Dict, Optional


class UnicodeCompat:
    """Cross-platform Unicode compatibility handler"""
    
    def __init__(self):
        self.system = platform.system().lower()
        self.encoding = self._detect_encoding()
        self.supports_unicode = self._check_unicode_support()
        
        # Define symbol mappings
        self._init_symbols()
    
    def _detect_encoding(self) -> str:
        """Detect the system's stdout encoding"""
        try:
            encoding = sys.stdout.encoding
            return encoding if encoding else 'ascii'
        except:
            return 'ascii'
    
    def _check_unicode_support(self) -> bool:
        """Check if the system supports Unicode characters"""
        try:
            # Test if we can encode common Unicode symbols
            test_chars = "✓✗⚠🎉"
            test_chars.encode(self.encoding)
            return True
        except (UnicodeEncodeError, LookupError):
            return False
    
    def _init_symbols(self) -> None:
        """Initialize symbol mappings based on system capabilities"""
        if self.supports_unicode:
            # Use Unicode symbols for systems that support them
            self.symbols = {
                'check': '✓',
                'cross': '✗', 
                'warning': '⚠',
                'success': '✅',
                'error': '❌',
                'skip': '⏭',
                'running': '🔄',
                'party': '🎉',
                'document': '📄',
                'download': '📥',
                'search': '🔍',
                'rocket': '🚀',
                'gear': '🔧',
                'lightbulb': '💡',
                'arrow_right': '→',
                'bullet': '•'
            }
        else:
            # Use ASCII fallbacks for limited encoding systems (like Windows cp1252)
            self.symbols = {
                'check': '[OK]',
                'cross': '[ERROR]',
                'warning': '[WARNING]',
                'success': '[SUCCESS]',
                'error': '[ERROR]',
                'skip': '[SKIP]',
                'running': '[RUNNING]',
                'party': '[COMPLETE]',
                'document': '[REPORT]',
                'download': '[DOWNLOAD]',
                'search': '[CHECK]',
                'rocket': '[START]',
                'gear': '[CONFIG]',
                'lightbulb': '[TIP]',
                'arrow_right': '->',
                'bullet': '*'
            }
    
    def get_symbol(self, symbol_name: str) -> str:
        """Get a cross-platform compatible symbol"""
        return self.symbols.get(symbol_name, f'[{symbol_name.upper()}]')
    
    def safe_print(self, text: str, **kwargs) -> None:
        """Print text with automatic encoding handling"""
        try:
            print(text, **kwargs)
        except UnicodeEncodeError:
            # Fallback: replace problematic characters
            safe_text = text.encode(self.encoding, errors='replace').decode(self.encoding)
            print(safe_text, **kwargs)
    
    def format_status(self, status: str, message: str) -> str:
        """Format a status message with appropriate symbol"""
        symbol = self.get_symbol(status)
        return f"{symbol} {message}"
    
    def get_platform_info(self) -> Dict[str, str]:
        """Get platform and encoding information"""
        return {
            'system': self.system,
            'encoding': self.encoding,
            'supports_unicode': self.supports_unicode,
            'python_version': sys.version.split()[0],
            'platform': platform.platform()
        }


# Global instance for easy importing
_unicode_compat = UnicodeCompat()

# Convenience functions for easy importing
def get_symbol(symbol_name: str) -> str:
    """Get a cross-platform compatible symbol"""
    return _unicode_compat.get_symbol(symbol_name)

def safe_print(text: str, **kwargs) -> None:
    """Print text with automatic encoding handling"""
    _unicode_compat.safe_print(text, **kwargs)

def format_status(status: str, message: str) -> str:
    """Format a status message with appropriate symbol"""
    return _unicode_compat.format_status(status, message)

# Symbol shortcuts for common use cases
CHECK = _unicode_compat.get_symbol('check')
CROSS = _unicode_compat.get_symbol('cross') 
WARNING = _unicode_compat.get_symbol('warning')
SUCCESS = _unicode_compat.get_symbol('success')
ERROR = _unicode_compat.get_symbol('error')
SKIP = _unicode_compat.get_symbol('skip')
RUNNING = _unicode_compat.get_symbol('running')
PARTY = _unicode_compat.get_symbol('party')
DOCUMENT = _unicode_compat.get_symbol('document')
DOWNLOAD = _unicode_compat.get_symbol('download')
SEARCH = _unicode_compat.get_symbol('search')
ROCKET = _unicode_compat.get_symbol('rocket')
GEAR = _unicode_compat.get_symbol('gear')
LIGHTBULB = _unicode_compat.get_symbol('lightbulb')
ARROW_RIGHT = _unicode_compat.get_symbol('arrow_right')
BULLET = _unicode_compat.get_symbol('bullet')

def get_platform_info() -> Dict[str, str]:
    """Get platform and encoding information"""
    return _unicode_compat.get_platform_info()


if __name__ == "__main__":
    # Test and display compatibility information
    print("=== Unicode Compatibility Test ===")
    info = get_platform_info()
    
    print(f"System: {info['system']}")
    print(f"Encoding: {info['encoding']}")
    print(f"Unicode Support: {info['supports_unicode']}")
    print(f"Python Version: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    
    print("\nSymbol Test:")
    symbols = ['check', 'cross', 'warning', 'success', 'error', 'skip', 
               'running', 'party', 'document', 'download', 'search']
    
    for symbol in symbols:
        safe_print(f"  {symbol}: {get_symbol(symbol)}")
    
    print("\nStatus Message Test:")
    safe_print(format_status('check', 'Operation completed successfully'))
    safe_print(format_status('cross', 'Operation failed'))
    safe_print(format_status('warning', 'Operation completed with warnings'))
    safe_print(format_status('party', 'Pipeline execution complete!'))