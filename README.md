# LogParser-LLM

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LogParser-LLM** is an efficient log parsing system that combines Large Language Models (LLM) with prefix tree clustering to automatically extract structured templates from unstructured log data.

Based on the research paper: *"LogParser-LLM: Advancing Efficient Log Parsing with Large Language Models"* (KDD 2024)

## 🌟 Key Features

- **High Efficiency**: Requires only ~272 LLM calls per 3.6M logs (99.99% reduction)
- **High Accuracy**: 90.6% F1 score for grouping, 81.1% for parsing
- **Zero Configuration**: No hyper-parameter tuning or labeled training data required
- **Online Learning**: Adapts to new log formats automatically
- **Multiple LLM Support**: OpenAI, Azure OpenAI, and local models
- **Smart Caching**: Three-level cache system for maximum performance
- **Production Ready**: Async processing, error handling, monitoring

## 🏗️ Architecture

```
┌─────────────┐
│   Raw Logs  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  Preprocessor   │  ← Clean & normalize
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Prefix Tree    │  ← Syntax clustering
└──────┬──────────┘
       │
       ├─ Cache Hit? ──→ Return Template
       │
       ▼
┌─────────────────┐
│  LLM Extractor  │  ← OpenAI API call
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Template Pool  │  ← Store & manage
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│  Parsed Output  │
└─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/logparser-llm.git
cd logparser-llm

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Basic Usage

```python
from logparser_llm import LogParserLLM, Config, LLMConfig

# Configure
config = Config(
    llm=LLMConfig(
        provider="openai",
        api_key="your-api-key",
        model="gpt-3.5-turbo"
    )
)

# Initialize
parser = LogParserLLM(config)

# Parse single log
log = "2024-01-01 10:00:00 ERROR Failed to connect to database"
result = parser.parse(log)

print(f"Template: {result.template_pattern}")
# Output: Template: <*> <*> <*> Failed to connect to <*>
```

### Batch Processing

```python
logs = [
    "User john logged in",
    "User mary logged in",
    "Failed to connect to server1",
]

results = parser.parse_batch(logs)

for r in results:
    print(f"{r.template_id}: {r.template_pattern}")

# Statistics
stats = parser.get_statistics()
print(f"LLM calls: {stats['llm_calls']}")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
```

## 📊 Performance Comparison

| Method | F1 Score | LLM Calls (per 3.6M logs) | Training Data |
|--------|----------|---------------------------|---------------|
| Drain | 75% | 0 | No |
| LogPPT | 85% | 0 | Yes (labeled) |
| ChatGPT (direct) | 90% | 3,600,000 | No |
| **LogParser-LLM** | **90.6%** | **272** | **No** |

## 🔧 Configuration

### YAML Configuration

```yaml
# config/config.yaml
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.0
  max_tokens: 500

parsing:
  use_cache: true
  batch_size: 20
  similarity_threshold: 0.85

prefix_tree:
  max_depth: 5
  min_cluster_size: 3
  enable_fuzzy_matching: true
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-3.5-turbo"
export PARSING_BATCH_SIZE="20"
```

## 💻 Advanced Usage

### File Processing

```python
# Parse logs from CSV file
df = parser.parse_file(
    "access.log",
    output_path="parsed_results.csv"
)
```

### Custom LLM Provider

```python
from logparser_llm import LLMConfig

# Azure OpenAI
config = Config(
    llm=LLMConfig(
        provider="azure",
        api_key="your-key",
        api_base="https://your-resource.openai.azure.com/",
        model="gpt-4"
    )
)

# Local LLM (OpenAI-compatible)
config = Config(
    llm=LLMConfig(
        provider="local",
        api_base="http://localhost:8000/v1",
        api_key="not-needed",
        model="llama-2-7b"
    )
)
```

### State Persistence

```python
# Save trained templates
parser.save_state("parser_state/")

# Load in new session
parser2 = LogParserLLM(config)
parser2.load_state("parser_state/")
```

### Async Processing

```python
import asyncio

async def parse_async():
    result = await parser.llm_client.extract_template_async(log)
    return result

asyncio.run(parse_async())
```

## 📁 Project Structure

```
logparser_llm/
├── core/
│   ├── parser.py           # Main parsing engine
│   ├── prefix_tree.py      # Prefix tree implementation
│   └── config_manager.py   # Configuration management
├── llm/
│   ├── client.py           # LLM API client
│   └── prompts.py          # Prompt templates
├── models/
│   └── log_entry.py        # Data models
├── storage/
│   └── template_pool.py    # Template storage
└── utils/
    ├── logger.py           # Logging utilities
    └── metrics.py          # Evaluation metrics
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=logparser_llm --cov-report=html

# Run specific test
pytest tests/test_parser.py::test_single_log
```

## 📈 Monitoring

```python
# Get detailed statistics
stats = parser.get_statistics()

print(f"Total Logs: {stats['total_logs']}")
print(f"Unique Templates: {stats['unique_templates']}")
print(f"LLM Efficiency: {stats['llm_efficiency']:.1f} logs/call")
print(f"Estimated Cost: ${stats['estimated_cost_usd']:.4f}")
```

## 🔍 Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Visualize prefix tree
print(parser.prefix_tree.visualize())
```

## 🎯 Use Cases

- **AIOps**: Automated log analysis and anomaly detection
- **DevOps**: Real-time monitoring and alerting
- **Security**: SIEM log parsing and threat detection
- **Troubleshooting**: Root cause analysis
- **Compliance**: Log auditing and reporting

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
# Development setup
git clone https://github.com/yourusername/logparser-llm.git
cd logparser-llm
pip install -e ".[dev]"

# Run checks
black .
flake8 .
mypy .
pytest
```

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{zhong2024logparser,
  title={LogParser-LLM: Advancing Efficient Log Parsing with Large Language Models},
  author={Zhong, Aoxiao and others},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2024}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original paper authors from Alibaba Cloud and CUHK
- OpenAI for the GPT API
- LogPAI team for benchmark datasets
- Open source community

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/logparser-llm/issues)
- **Email**: your.email@example.com
- **Documentation**: [Wiki](https://github.com/yourusername/logparser-llm/wiki)

## 🗺️ Roadmap

- [ ] Support for more LLM providers (Anthropic Claude, Google PaLM)
- [ ] Web UI for interactive parsing
- [ ] Real-time streaming log processing
- [ ] Kubernetes/Docker deployment examples
- [ ] Integration with popular log systems (ELK, Splunk)
- [ ] Model fine-tuning for domain-specific logs
- [ ] Multi-language log support

---

⭐ **Star this repo if you find it helpful!**