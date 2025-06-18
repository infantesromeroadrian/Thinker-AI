# Test Suite for Thinker AI Auxiliary Window

This directory contains all test modules for the Thinker AI Auxiliary Window project, following the testing structure requirements specified in the project memories.

## Structure

```
tests/
├── __init__.py                 # Test package initialization
├── test_config.py             # Configuration system tests
├── README.md                  # This file
└── ...                        # Additional test modules
```

## Requirements

All test files require the following path configuration at the top to import from src correctly:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

All imports must use the full src path (e.g., `from src.config.config import AppConfig`).

## Running Tests

### Install Testing Dependencies

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Or install all dev dependencies
pip install -r requirements.txt
```

### Execute Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_config.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run in verbose mode
python -m pytest tests/ -v

# Run integration tests only
python -m pytest tests/ -m integration

# Run excluding integration tests
python -m pytest tests/ -m "not integration"
```

### Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions (marked with `@pytest.mark.integration`)
- **End-to-End Tests**: Test complete workflows

## Writing Tests

### Test Class Structure

```python
class TestClassName:
    """Test suite for ClassName."""
    
    def test_specific_functionality(self) -> None:
        """Test specific functionality with descriptive name."""
        # Arrange
        expected_value = "test"
        
        # Act
        result = function_to_test()
        
        # Assert
        assert result == expected_value
```

### Required Test Coverage

- All new code must have ≥90% test coverage
- Critical paths must have 100% coverage
- All public methods must be tested
- Error conditions must be tested

### Test Naming Conventions

- Test files: `test_module_name.py`
- Test classes: `TestClassName`
- Test methods: `test_specific_functionality`
- Use descriptive names that explain what is being tested

## Current Test Modules

### test_config.py
Tests the configuration system including:
- AppConfig base class functionality
- Development and Production configurations
- Feature flags
- Configuration factory function
- Environment-based configuration selection

## Contributing

When adding new functionality:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Verify coverage requirements are met
4. Update this README if adding new test categories 