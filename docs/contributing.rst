Contributing to Cell-LISCA
==========================

We welcome contributions to Cell-LISCA! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

.. code-block:: bash

   git clone https://github.com/yourusername/cell-lisca.git
   cd cell-lisca

3. Create a virtual environment:

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode:

.. code-block:: bash

   # Install all packages
   uv sync

   # Or install individually
   cd cell-pattern && uv sync
   cd ../cell-filter && uv sync
   cd ../cell-grapher && uv sync
   cd ../cell-viewer && uv sync
   cd ../cell-tensionmap && uv sync

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

We follow the style guidelines in ``CLAUDE.md``:

- Google-style docstrings
- Type hints with modern union syntax (``int | str`` instead of ``Union[int, str]``)
- Organized imports (standard, third-party, local)
- Consistent naming conventions

Example:

.. code-block:: python

   """Module docstring in Google style.

   This module provides functionality for...
   """

   import logging
   from pathlib import Path
   from typing import Dict, List, Optional

   import numpy as np
   import matplotlib.pyplot as plt

   from cell_filter.core import BaseClass


   class ExampleClass:
       """Class docstring in Google style.

       Attributes:
           attr1: Description of attr1
           attr2: Description of attr2
       """

       def __init__(self, param1: str, param2: int = 0) -> None:
           """Initialize ExampleClass.

           Args:
               param1: Description of param1
               param2: Description of param2
           """
           self.attr1 = param1
           self.attr2 = param2

       def method(self, arg1: List[int]) -> Dict[str, int]:
           """Method docstring.

           Args:
               arg1: List of integers

           Returns:
               Dictionary with results
           """
           return {"result": sum(arg1)}

Testing
~~~~~~~

Currently, Cell-LISCA does not have formal tests. When adding tests:

1. Place tests in ``tests/`` directories within each package
2. Use pytest for testing framework
3. Focus on validation rather than specific numeric results for data-dependent tests

Example test structure:

.. code-block:: python

   # cell-filter/tests/test_core.py
   import pytest
   import numpy as np
   from cell_filter.core import Cropper

   def test_cropper_initialization():
       """Test that Cropper initializes correctly."""
       cropper = Cropper(
           patterns_path="test_patterns.nd2",
           cells_path="test_cells.nd2",
           parameters=CropperParameters(nuclei_channel=1)
       )
       assert cropper.nuclei_channel == 1
       cropper.close()

Documentation
~~~~~~~~~~~~~

Documentation is in the ``docs/`` folder using Sphinx and reStructuredText:

- Build docs: ``cd docs && make html``
- View docs: Open ``_build/html/index.html``
- Update API docs when adding new functions/classes

Submitting Changes
------------------

1. Create a new branch for your feature:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. Make your changes and commit them:

.. code-block:: bash

   git add .
   git commit -m "feat: add new feature description"

3. Push to your fork:

.. code-block:: bash

   git push origin feature/your-feature-name

4. Create a pull request on GitHub

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~

- Provide a clear description of changes
- Link to relevant issues
- Include screenshots for UI changes
- Ensure documentation is updated

Bug Reports
-----------

When reporting bugs:

1. Use the GitHub issue tracker
2. Include:
   - Python version
   - Operating system
   - Package versions
   - Minimal reproducible example
   - Error messages with full traceback

Feature Requests
----------------

For feature requests:

1. Check if already requested
2. Provide use case and motivation
3. Consider if it fits the project scope
4. Be open to discussion

Development Setup
-----------------

Pre-commit Hooks
~~~~~~~~~~~~~~~

Install pre-commit hooks for code quality:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Code Quality Tools
~~~~~~~~~~~~~~~~~~

- Linting: ``ruff check --fix .``
- Type checking: No explicit typecheck command configured
- Formatting: Follow guidelines in ``CLAUDE.md``

Architecture
------------

Module Structure
~~~~~~~~~~~~~~~~

Each module follows this structure::

   module-name/
   ├── src/module_name/
   │   ├── __init__.py
   │   ├── main.py           # CLI entry point
   │   ├── core/             # Core functionality
   │   ├── utils/            # Utilities
   │   └── analysis/         # Analysis modules (if applicable)
   ├── tests/
   ├── pyproject.toml
   └── README.md

Data Flow
~~~~~~~~~

1. **cell-pattern**: ND2 → HDF5 (bounding boxes)
2. **cell-filter**: HDF5 → HDF5 (add analysis, segmentation)
3. **cell-grapher**: NPY → Analysis results
4. **cell-tensionmap**: Segmentation → Stress tensors
5. **cell-viewer**: NPY → Interactive visualization

Adding New Modules
------------------

If adding a new module:

1. Follow the existing structure
2. Include CLI interface
3. Support both command-line and Python API
4. Add comprehensive documentation
5. Update main README and docs

Release Process
---------------

Versioning
~~~~~~~~~~

Cell-LISCA follows semantic versioning:
- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

Making a Release
~~~~~~~~~~~~~~~~

1. Update version numbers in ``pyproject.toml`` files
2. Update changelog
3. Tag release: ``git tag v0.1.0``
4. Push tag: ``git push origin v0.1.0``
5. GitHub Actions will build and publish

Community
---------

Code of Conduct
~~~~~~~~~~~~~~~

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and considerate
- Use inclusive language
- Focus on constructive feedback
- Help others learn and grow

Getting Help
~~~~~~~~~~~~

- GitHub Issues: Bug reports and feature requests
- Email: ctyjackcao@outlook.com
- Documentation: Check ``docs/`` folder

Acknowledgments
---------------

Contributors will be acknowledged in:
- AUTHORS file
- Release notes
- Documentation

Thank you for contributing to Cell-LISCA!
