"""
Basic usage examples for LogParser-LLM.
"""
import os
from logparser_llm.core.config_manager import (
    Config, LLMConfig, ParsingConfig, 
    PrefixTreeConfig, MergingConfig, PerformanceConfig
)
from logparser_llm.core.parser import LogParserLLM


def example_1_single_log():
    """Example 1: Parse a single log message."""
    print("=" * 60)
    print("Example 1: Parsing a Single Log")
    print("=" * 60)
    
    # Configure the parser
    config = Config(
        llm=LLMConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo",
            temperature=0.0
        ),
        parsing=ParsingConfig(),
        prefix_tree=PrefixTreeConfig(),
        merging=MergingConfig(),
        performance=PerformanceConfig()
    )
    
    # Initialize parser
    parser = LogParserLLM(config)
    
    # Parse a log
    log = "2024-01-01 10:00:00 ERROR Failed to connect to database server1 on port 5432"
    result = parser.parse(log)
    
    print(f"\nOriginal Log:")
    print(f"  {result.original}")
    print(f"\nExtracted Template:")
    print(f"  {result.template_pattern}")
    print(f"\nTemplate ID: {result.template_id}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Used LLM: {result.used_llm}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")
    print()


def example_2_batch_logs():
    """Example 2: Parse multiple logs efficiently."""
    print("=" * 60)
    print("Example 2: Batch Log Parsing")
    print("=" * 60)
    
    config = Config(
        llm=LLMConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        ),
        parsing=ParsingConfig(
            use_cache=True,
            enable_batch_processing=True
        ),
        prefix_tree=PrefixTreeConfig(),
        merging=MergingConfig(),
        performance=PerformanceConfig()
    )
    
    parser = LogParserLLM(config)
    
    # Sample logs
    logs = [
        "User john logged in at 10:00:00",
        "User mary logged in at 10:05:00",
        "User bob logged in at 10:10:00",
        "Failed to connect to database server1",
        "Failed to connect to database server2",
        "Failed to connect to database server3",
        "Process 1234 started successfully",
        "Process 5678 started successfully",
    ]
    
    print(f"\nParsing {len(logs)} logs...")
    results = parser.parse_batch(logs)
    
    # Group by template
    templates = {}
    for result in results:
        tid = result.template_id
        if tid not in templates:
            templates[tid] = {
                'pattern': result.template_pattern,
                'logs': []
            }
        templates[tid]['logs'].append(result.original)
    
    print(f"\nFound {len(templates)} unique templates:\n")
    for tid, data in templates.items():
        print(f"Template {tid}: {data['pattern']}")
        print(f"  Matched {len(data['logs'])} logs:")
        for log in data['logs'][:3]:  # Show first 3
            print(f"    - {log}")
        if len(data['logs']) > 3:
            print(f"    ... and {len(data['logs']) - 3} more")
        print()
    
    # Print statistics
    stats = parser.get_statistics()
    print("\nParsing Statistics:")
    print(f"  Total Logs: {stats['total_logs']}")
    print(f"  Unique Templates: {stats['unique_templates']}")
    print(f"  LLM Calls: {stats['llm_calls']}")
    print(f"  Cache Hits: {stats['cache_hits']}")
    print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
    print(f"  LLM Efficiency: {stats['llm_efficiency']:.1f} logs/call")
    print()


def example_3_file_parsing():
    """Example 3: Parse logs from a file."""
    print("=" * 60)
    print("Example 3: File-based Log Parsing")
    print("=" * 60)
    
    config = Config(
        llm=LLMConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        ),
        parsing=ParsingConfig(),
        prefix_tree=PrefixTreeConfig(),
        merging=MergingConfig(),
        performance=PerformanceConfig()
    )
    
    parser = LogParserLLM(config)
    
    # Create sample log file
    sample_logs = [
        "2024-01-01 10:00:00 INFO Application started",
        "2024-01-01 10:00:01 INFO Connecting to database",
        "2024-01-01 10:00:02 ERROR Connection failed: timeout",
        "2024-01-01 10:00:03 INFO Retrying connection...",
        "2024-01-01 10:00:04 INFO Successfully connected",
        "2024-01-01 10:00:05 INFO User admin logged in",
        "2024-01-01 10:00:06 INFO User john logged in",
        "2024-01-01 10:00:07 WARNING High memory usage: 85%",
        "2024-01-01 10:00:08 WARNING High memory usage: 90%",
        "2024-01-01 10:00:09 ERROR Out of memory",
    ]
    
    # Save to file
    with open("sample_logs.txt", "w") as f:
        for log in sample_logs:
            f.write(log + "\n")
    
    print("\nParsing logs from file: sample_logs.txt")
    
    # Parse file
    result_df = parser.parse_file(
        "sample_logs.txt",
        output_path="parsed_results.csv"
    )
    
    print(f"\nParsed {len(result_df)} logs")
    print("\nSample results:")
    print(result_df[['original', 'template', 'confidence']].head(10))
    
    print("\nResults saved to: parsed_results.csv")
    
    # Clean up
    os.remove("sample_logs.txt")


def example_4_with_config_file():
    """Example 4: Load configuration from YAML file."""
    print("=" * 60)
    print("Example 4: Using Configuration File")
    print("=" * 60)
    
    # Load config from file
    try:
        from logparser_llm.core.config_manager import load_config
        config = load_config("config/config.yaml")
        
        parser = LogParserLLM(config)
        
        log = "Connection to 192.168.1.1:8080 established"
        result = parser.parse(log)
        
        print(f"\nParsed with config from file:")
        print(f"  Log: {result.original}")
        print(f"  Template: {result.template_pattern}")
        
    except FileNotFoundError:
        print("\nConfig file not found. Create config/config.yaml first.")
        print("See config/config.yaml.example for reference.")


def example_5_save_and_load_state():
    """Example 5: Save and load parser state."""
    print("=" * 60)
    print("Example 5: State Persistence")
    print("=" * 60)
    
    config = Config(
        llm=LLMConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo"
        ),
        parsing=ParsingConfig(),
        prefix_tree=PrefixTreeConfig(),
        merging=MergingConfig(),
        performance=PerformanceConfig()
    )
    
    # Create and train parser
    print("\n1. Creating parser and parsing logs...")
    parser1 = LogParserLLM(config)
    
    logs = [
        "User logged in successfully",
        "Failed to authenticate user",
        "Session expired",
    ]
    
    parser1.parse_batch(logs)
    
    print(f"   Templates created: {len(parser1.template_pool.templates)}")
    
    # Save state
    print("\n2. Saving parser state...")
    parser1.save_state("parser_state")
    
    # Create new parser and load state
    print("\n3. Creating new parser and loading state...")
    parser2 = LogParserLLM(config)
    parser2.load_state("parser_state")
    
    print(f"   Templates loaded: {len(parser2.template_pool.templates)}")
    
    # Use loaded parser (should use cache)
    print("\n4. Parsing with loaded state (should use cache)...")
    result = parser2.parse("User logged in successfully")
    print(f"   Cache hit: {result.cache_hit}")
    print(f"   Used LLM: {result.used_llm}")


if __name__ == "__main__":
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set it using: export OPENAI_API_KEY='your-key-here'")
        exit(1)
    
    # Run examples
    try:
        example_1_single_log()
        example_2_batch_logs()
        # example_3_file_parsing()  # Uncomment to run
        # example_4_with_config_file()  # Uncomment to run
        # example_5_save_and_load_state()  # Uncomment to run
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()