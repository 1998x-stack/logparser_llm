"""
Complete usage example showing all features of LogParser-LLM.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import LogParser-LLM components
from logparser_llm import (
    LogParserLLM,
    Config,
    LLMConfig,
    load_config,
    create_default_config,
    setup_logger,
)


def example_1_quickstart():
    """Example 1: Quick start with minimal configuration."""
    print("="*60)
    print("Example 1: Quick Start")
    print("="*60)
    
    # Method 1: Direct configuration
    config = Config(
        llm=LLMConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY", "your-key-here"),
            model="gpt-3.5-turbo",
            temperature=0.0
        )
    )
    
    # Initialize parser
    parser = LogParserLLM(config)
    
    # Parse a log
    log = "2024-01-15 14:30:00 ERROR Connection to database failed after 3 retries"
    result = parser.parse(log)
    
    print(f"\nOriginal:  {result.original}")
    print(f"Template:  {result.template_pattern}")
    print(f"ID:        {result.template_id}")
    print(f"Used LLM:  {result.used_llm}")
    print(f"Time:      {result.processing_time_ms:.2f}ms")
    print()


def example_2_load_from_yaml():
    """Example 2: Load configuration from YAML file."""
    print("="*60)
    print("Example 2: Load Config from YAML")
    print("="*60)
    
    # Create config file if it doesn't exist
    config_path = "config/config.yaml"
    if not Path(config_path).exists():
        print(f"\nCreating default config at {config_path}...")
        create_default_config(config_path)
        print("Please edit config/config.yaml and set your API key!")
        return
    
    # Load config from YAML
    try:
        config = load_config(config_path)
        print(f"\n✓ Config loaded from {config_path}")
        print(f"  Provider: {config.llm.provider}")
        print(f"  Model:    {config.llm.model}")
        print(f"  Cache:    {config.parsing.cache_type}")
        
        # Use the parser
        parser = LogParserLLM(config)
        log = "User admin logged in from 192.168.1.100"
        result = parser.parse(log)
        
        print(f"\nParsed log:")
        print(f"  Template: {result.template_pattern}")
        
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
    
    print()


def example_3_batch_processing():
    """Example 3: Efficient batch processing."""
    print("="*60)
    print("Example 3: Batch Processing")
    print("="*60)
    
    # Setup
    config = Config(
        llm=LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            model="gpt-3.5-turbo"
        )
    )
    
    parser = LogParserLLM(config)
    
    # Sample logs
    logs = [
        "2024-01-15 10:00:00 INFO User john logged in",
        "2024-01-15 10:05:00 INFO User mary logged in",
        "2024-01-15 10:10:00 INFO User bob logged in",
        "2024-01-15 10:15:00 ERROR Failed to connect to database server1",
        "2024-01-15 10:20:00 ERROR Failed to connect to database server2",
        "2024-01-15 10:25:00 ERROR Failed to connect to database server3",
        "2024-01-15 10:30:00 WARN Memory usage at 85%",
        "2024-01-15 10:35:00 WARN Memory usage at 90%",
        "2024-01-15 10:40:00 INFO Process 1234 started",
        "2024-01-15 10:45:00 INFO Process 5678 started",
    ]
    
    print(f"\nParsing {len(logs)} logs...")
    results = parser.parse_batch(logs)
    
    # Group by template
    from collections import defaultdict
    templates = defaultdict(list)
    for result in results:
        templates[result.template_id].append(result)
    
    print(f"\nFound {len(templates)} unique templates:\n")
    for template_id, group in templates.items():
        template_pattern = group[0].template_pattern
        print(f"[{template_id}] {template_pattern}")
        print(f"  Matched {len(group)} logs")
        for r in group[:2]:  # Show first 2
            print(f"    - {r.original}")
        if len(group) > 2:
            print(f"    ... and {len(group)-2} more")
        print()
    
    # Show statistics
    stats = parser.get_statistics()
    print("Statistics:")
    print(f"  Total logs:    {stats['total_logs']}")
    print(f"  LLM calls:     {stats['llm_calls']}")
    print(f"  Cache hits:    {stats['cache_hits']}")
    print(f"  Hit rate:      {stats['cache_hit_rate']:.2%}")
    print(f"  Efficiency:    {stats['llm_efficiency']:.1f} logs/call")
    print()


def example_4_with_preprocessing():
    """Example 4: Using preprocessing options."""
    print("="*60)
    print("Example 4: Preprocessing")
    print("="*60)
    
    from logparser_llm.config_manager import PreprocessingConfig
    
    # Config with preprocessing
    config = Config(
        llm=LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            model="gpt-3.5-turbo"
        ),
        preprocessing=PreprocessingConfig(
            remove_timestamps=True,
            normalize_numbers=True,
            normalize_paths=True
        )
    )
    
    parser = LogParserLLM(config)
    
    # Logs with timestamps and numbers
    logs = [
        "2024-01-15 10:00:00 INFO Processing file /var/log/app.log",
        "2024-01-16 11:00:00 INFO Processing file /var/log/system.log",
        "2024-01-17 12:00:00 ERROR Failed with code 404",
        "2024-01-18 13:00:00 ERROR Failed with code 500",
    ]
    
    print("\nOriginal logs:")
    for log in logs:
        print(f"  {log}")
    
    results = parser.parse_batch(logs)
    
    print("\nTemplates (normalized):")
    seen = set()
    for r in results:
        if r.template_id not in seen:
            print(f"  {r.template_pattern}")
            seen.add(r.template_id)
    print()


def example_5_state_persistence():
    """Example 5: Save and load parser state."""
    print("="*60)
    print("Example 5: State Persistence")
    print("="*60)
    
    # Create parser
    config = Config(
        llm=LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            model="gpt-3.5-turbo"
        )
    )
    
    # Train parser
    print("\n1. Training parser...")
    parser1 = LogParserLLM(config)
    
    training_logs = [
        "User logged in successfully",
        "Failed to authenticate user",
        "Database connection established",
        "Query executed in 150ms",
    ]
    
    parser1.parse_batch(training_logs)
    print(f"   Created {len(parser1.template_pool.templates)} templates")
    
    # Save state
    print("\n2. Saving state...")
    save_dir = "parser_state"
    parser1.save_state(save_dir)
    print(f"   Saved to {save_dir}/")
    
    # Create new parser and load state
    print("\n3. Loading state in new parser...")
    parser2 = LogParserLLM(config)
    parser2.load_state(save_dir)
    print(f"   Loaded {len(parser2.template_pool.templates)} templates")
    
    # Use loaded parser
    print("\n4. Using loaded parser...")
    result = parser2.parse("User logged in successfully")
    print(f"   Cache hit:  {result.cache_hit}")
    print(f"   Used LLM:   {result.used_llm}")
    print(f"   Template:   {result.template_pattern}")
    print()


def example_6_custom_logger():
    """Example 6: Custom logger setup."""
    print("="*60)
    print("Example 6: Custom Logger")
    print("="*60)
    
    # Setup custom logger
    logger = setup_logger(
        log_level="DEBUG",
        log_file="logs/logparser.log",
        rotation="10 MB"
    )
    
    print("\n✓ Logger configured")
    print("  Level:    DEBUG")
    print("  File:     logs/logparser.log")
    print("  Rotation: 10 MB")
    
    # Now logs will be written to file
    config = Config(
        llm=LLMConfig(
            api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            model="gpt-3.5-turbo"
        )
    )
    
    parser = LogParserLLM(config)
    parser.parse("Test log message")
    
    print("\nCheck logs/logparser.log for detailed logs")
    print()


def example_7_environment_variables():
    """Example 7: Configuration via environment variables."""
    print("="*60)
    print("Example 7: Environment Variables")
    print("="*60)
    
    # Set environment variables
    os.environ['LLM_PROVIDER'] = 'openai'
    os.environ['LLM_MODEL'] = 'gpt-3.5-turbo'
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'test-key')
    os.environ['PARSING_BATCH_SIZE'] = '50'
    os.environ['PARSING_USE_CACHE'] = 'true'
    
    print("\nEnvironment variables set:")
    print(f"  LLM_PROVIDER:         {os.environ['LLM_PROVIDER']}")
    print(f"  LLM_MODEL:            {os.environ['LLM_MODEL']}")
    print(f"  PARSING_BATCH_SIZE:   {os.environ['PARSING_BATCH_SIZE']}")
    
    # Load config (will use environment variables)
    try:
        config = load_config()
        print("\n✓ Config loaded from environment")
        print(f"  Provider:    {config.llm.provider}")
        print(f"  Model:       {config.llm.model}")
        print(f"  Batch size:  {config.parsing.batch_size}")
        
    except ValueError as e:
        print(f"\n✗ {e}")
        print("Please set OPENAI_API_KEY environment variable")
    
    print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("LogParser-LLM - Complete Examples")
    print("="*60 + "\n")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set")
        print("Some examples will use 'test-key' placeholder\n")
    
    try:
        # Run examples
        example_1_quickstart()
        # example_2_load_from_yaml()  # Uncomment if you have config file
        example_3_batch_processing()
        example_4_with_preprocessing()
        example_5_state_persistence()
        # example_6_custom_logger()  # Uncomment to test logging
        example_7_environment_variables()
        
        print("="*60)
        print("All examples completed!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()