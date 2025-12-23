"""
Prompt templates for LLM-based log parsing.
"""
from typing import List, Optional


class PromptBuilder:
    """Builder for constructing prompts for log template extraction."""
    
    def __init__(self):
        self.system_message = (
            "You are an expert at analyzing log messages and extracting templates. "
            "Your task is to identify the constant parts of log messages and replace "
            "variable parts with <*> placeholders."
        )
    
    def build_extraction_prompt(
        self,
        log: str,
        examples: Optional[List[str]] = None,
        use_ner: bool = False
    ) -> str:
        """
        Build prompt for single log template extraction.
        
        Args:
            log: Log message to parse
            examples: Optional example logs for few-shot learning
            use_ner: Whether to include NER instructions
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # Add task description
        prompt_parts.append(self._get_task_description(use_ner))
        
        # Add examples if provided
        if examples:
            prompt_parts.append("\n## Examples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Input: {example}")
                # You would have the template for this example
                prompt_parts.append(f"Output: {self._generate_example_template(example)}")
        
        # Add rules
        prompt_parts.append("\n## Rules:")
        prompt_parts.append(self._get_rules())
        
        # Add the actual log to parse
        prompt_parts.append("\n## Task:")
        prompt_parts.append(f"Extract the template from this log message:")
        prompt_parts.append(f"Input: {log}")
        prompt_parts.append("\nOutput:")
        
        return "\n".join(prompt_parts)
    
    def build_batch_extraction_prompt(
        self,
        logs: List[str],
        examples: Optional[List[str]] = None
    ) -> str:
        """
        Build prompt for batch template extraction.
        
        Args:
            logs: List of log messages
            examples: Optional examples
            
        Returns:
            Batch extraction prompt
        """
        prompt_parts = []
        
        prompt_parts.append(self._get_task_description(use_ner=False))
        
        if examples:
            prompt_parts.append("\n## Examples:")
            for example in examples:
                prompt_parts.append(f"Input: {example}")
                prompt_parts.append(f"Output: {self._generate_example_template(example)}")
                prompt_parts.append("")
        
        prompt_parts.append("\n## Rules:")
        prompt_parts.append(self._get_rules())
        
        prompt_parts.append("\n## Task:")
        prompt_parts.append("Extract templates from the following log messages.")
        prompt_parts.append("Provide one template per line in the same order.\n")
        
        for i, log in enumerate(logs, 1):
            prompt_parts.append(f"{i}. {log}")
        
        prompt_parts.append("\nOutput (one template per line):")
        
        return "\n".join(prompt_parts)
    
    def build_merging_prompt(self, template1: str, template2: str) -> str:
        """
        Build prompt for determining if templates should be merged.
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            Merging decision prompt
        """
        prompt = f"""Determine if these two log templates should be merged into one.

Template 1: {template1}
Template 2: {template2}

Consider:
1. Do they represent the same log event?
2. Are the differences just in variable positioning?
3. Would merging lose important information?

Respond with either:
- "MERGE: <merged_template>" if they should be merged
- "KEEP_SEPARATE: <reason>" if they should stay separate

Response:"""
        return prompt
    
    def _get_task_description(self, use_ner: bool = False) -> str:
        """Get task description."""
        base_description = """# Log Template Extraction Task

You are analyzing system log messages to extract their templates. A template is the constant 
structure of a log message with variable parts replaced by <*> placeholders.

For example:
- Log: "User john123 logged in at 10:30:45"
- Template: "User <*> logged in at <*>"

- Log: "Failed to connect to database server1 on port 5432"
- Template: "Failed to connect to database <*> on port <*>"
"""
        
        if use_ner:
            ner_addition = """
Use Named Entity Recognition to identify:
- Timestamps, dates, times
- IP addresses, hostnames
- User IDs, session IDs
- File paths, URLs
- Numbers, percentages
- UUIDs, hashes
- Status codes

Replace all these with <*> in the template.
"""
            return base_description + ner_addition
        
        return base_description
    
    def _get_rules(self) -> str:
        """Get extraction rules."""
        return """1. Keep all constant text (keywords, operators, connectors) exactly as they appear
2. Replace all variable values with <*>
3. Variables include: numbers, timestamps, IDs, paths, IPs, hostnames, etc.
4. Preserve the structure and word order
5. Don't add or remove words
6. Use <*> only for variable parts, not for words
7. The template should be generalizable to similar log messages

Examples of what to replace with <*>:
- Timestamps: "2024-01-01 10:30:45" → <*>
- Numbers: "5432", "100.5", "90%" → <*>
- IDs: "user123", "session-abc-123" → <*>
- Paths: "/var/log/app.log" → <*>
- IPs: "192.168.1.1" → <*>
- Hostnames: "server1", "db-prod-01" → <*>"""
    
    def _generate_example_template(self, log: str) -> str:
        """
        Generate an example template (simple heuristic for demonstration).
        This is just for creating few-shot examples.
        """
        import re
        
        template = log
        
        # Replace common patterns
        patterns = [
            (r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', '<*>'),  # Timestamp
            (r'\d+\.\d+\.\d+\.\d+', '<*>'),  # IP address
            (r'\d+', '<*>'),  # Numbers
            (r'/[\w/.-]+', '<*>'),  # File paths
            (r'[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+', '<*>'),  # Emails
            (r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', '<*>'),  # UUID
        ]
        
        for pattern, replacement in patterns:
            template = re.sub(pattern, replacement, template)
        
        return template
    
    def build_validation_prompt(self, log: str, template: str) -> str:
        """
        Build prompt to validate if a template matches a log.
        
        Args:
            log: Original log message
            template: Proposed template
            
        Returns:
            Validation prompt
        """
        prompt = f"""Validate if this template correctly represents the log message.

Log: {log}
Template: {template}

Check:
1. Does the template preserve all constant text?
2. Are all variables properly replaced with <*>?
3. Does the template maintain the correct structure?

Respond with either:
- "VALID" if the template is correct
- "INVALID: <reason>" if there are issues

Response:"""
        return prompt
    
    def build_similarity_prompt(self, log1: str, log2: str) -> str:
        """
        Build prompt to assess if two logs should share a template.
        
        Args:
            log1: First log message
            log2: Second log message
            
        Returns:
            Similarity assessment prompt
        """
        prompt = f"""Determine if these two log messages represent the same log event type 
and should share the same template.

Log 1: {log1}
Log 2: {log2}

Consider:
1. Do they describe the same type of event?
2. Do they have the same structure?
3. Are differences only in variable values?

Respond with either:
- "SAME: <template>" if they should share a template
- "DIFFERENT: <reason>" if they are different event types

Response:"""
        return prompt


# Example usage
if __name__ == "__main__":
    builder = PromptBuilder()
    
    # Test single extraction
    log = "2024-01-01 10:00:00 ERROR Failed to connect to database server1"
    prompt = builder.build_extraction_prompt(log)
    print("=== Single Extraction Prompt ===")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Test batch extraction
    logs = [
        "User john logged in",
        "User mary logged out",
        "Connection timeout after 30 seconds"
    ]
    batch_prompt = builder.build_batch_extraction_prompt(logs)
    print("=== Batch Extraction Prompt ===")
    print(batch_prompt)