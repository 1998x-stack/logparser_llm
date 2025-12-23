#!/bin/bash

# Create the main project directory
mkdir -p logparser_llm

# Change to the project directory
cd logparser_llm || exit

# Create root level files
touch README.md
touch requirements.txt
touch setup.py

# Create config directory
mkdir -p config
touch config/__init__.py
touch config/config.yaml

# Create main package directory
mkdir -p logparser_llm

# Create core module
mkdir -p logparser_llm/core
touch logparser_llm/__init__.py
touch logparser_llm/core/__init__.py
touch logparser_llm/core/parser.py
touch logparser_llm/core/prefix_tree.py
touch logparser_llm/core/llm_extractor.py
touch logparser_llm/core/merger.py

# Create models module
mkdir -p logparser_llm/models
touch logparser_llm/models/__init__.py
touch logparser_llm/models/log_entry.py
touch logparser_llm/models/template.py
touch logparser_llm/models/cluster.py

# Create preprocessor module
mkdir -p logparser_llm/preprocessor
touch logparser_llm/preprocessor/__init__.py
touch logparser_llm/preprocessor/cleaner.py

# Create storage module
mkdir -p logparser_llm/storage
touch logparser_llm/storage/__init__.py
touch logparser_llm/storage/template_pool.py
touch logparser_llm/storage/cache.py

# Create llm module
mkdir -p logparser_llm/llm
touch logparser_llm/llm/__init__.py
touch logparser_llm/llm/client.py
touch logparser_llm/llm/prompts.py

# Create utils module
mkdir -p logparser_llm/utils
touch logparser_llm/utils/__init__.py
touch logparser_llm/utils/logger.py
touch logparser_llm/utils/metrics.py

# Create api module
mkdir -p logparser_llm/api
touch logparser_llm/api/__init__.py
touch logparser_llm/api/rest_api.py

# Create tests directory
mkdir -p tests
touch tests/__init__.py
touch tests/test_parser.py
touch tests/test_prefix_tree.py
touch tests/test_llm_extractor.py

# Create examples directory
mkdir -p examples
touch examples/basic_usage.py
touch examples/batch_processing.py
touch examples/custom_config.py

# Create data directory with subdirectories
mkdir -p data/sample_logs
mkdir -p data/benchmarks

# Create a simple verification output
echo "Folder structure created successfully:"
find . -type f | sort
