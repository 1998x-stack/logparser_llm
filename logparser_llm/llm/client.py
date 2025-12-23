"""
LLM client for template extraction using OpenAI API.
"""
import time
from typing import List, Optional, Dict, Any
from openai import OpenAI, AzureOpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
import asyncio
from loguru import logger

from ..core.config_manager import LLMConfig
from .prompts import PromptBuilder


class LLMClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = self._create_client()
        self.async_client = self._create_async_client()
        self.prompt_builder = PromptBuilder()
        
        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        logger.info(f"LLM client initialized: {config.provider}/{config.model}")
    
    def _create_client(self) -> OpenAI:
        """Create synchronous OpenAI client."""
        if self.config.provider == "azure":
            return AzureOpenAI(
                api_key=self.config.api_key,
                api_version="2024-02-15-preview",
                azure_endpoint=self.config.api_base
            )
        elif self.config.provider == "openai":
            kwargs = {"api_key": self.config.api_key}
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base
            return OpenAI(**kwargs)
        else:
            # Generic OpenAI-compatible endpoint
            return OpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.api_base
            )
    
    def _create_async_client(self) -> AsyncOpenAI:
        """Create asynchronous OpenAI client."""
        if self.config.provider == "openai":
            kwargs = {"api_key": self.config.api_key}
            if self.config.api_base:
                kwargs["base_url"] = self.config.api_base
            return AsyncOpenAI(**kwargs)
        else:
            return AsyncOpenAI(
                api_key=self.config.api_key or "not-needed",
                base_url=self.config.api_base or "http://localhost:8000/v1"
            )
    
    def extract_template(
        self,
        log: str,
        examples: Optional[List[str]] = None,
        use_ner: bool = False
    ) -> str:
        """
        Extract template from a single log using LLM.
        
        Args:
            log: Log message to parse
            examples: Optional example logs for few-shot learning
            use_ner: Whether to use NER prompting
            
        Returns:
            Extracted template pattern
        """
        prompt = self.prompt_builder.build_extraction_prompt(
            log=log,
            examples=examples,
            use_ner=use_ner
        )
        
        template = self._call_llm(prompt)
        self.total_calls += 1
        
        return template
    
    def extract_template_batch(
        self,
        logs: List[str],
        examples: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract templates from multiple logs in batch.
        
        Args:
            logs: List of log messages
            examples: Optional example logs
            
        Returns:
            List of extracted templates
        """
        prompt = self.prompt_builder.build_batch_extraction_prompt(
            logs=logs,
            examples=examples
        )
        
        response = self._call_llm(prompt)
        self.total_calls += 1
        
        # Parse response to extract individual templates
        templates = self._parse_batch_response(response, len(logs))
        return templates
    
    async def extract_template_async(
        self,
        log: str,
        examples: Optional[List[str]] = None
    ) -> str:
        """
        Extract template asynchronously.
        
        Args:
            log: Log message
            examples: Optional examples
            
        Returns:
            Extracted template
        """
        prompt = self.prompt_builder.build_extraction_prompt(log, examples)
        template = await self._call_llm_async(prompt)
        self.total_calls += 1
        return template
    
    def _call_llm(self, prompt: str, retries: int = 0) -> str:
        """
        Make synchronous call to LLM API with retry logic.
        
        Args:
            prompt: Prompt to send
            retries: Current retry count
            
        Returns:
            LLM response text
        """
        try:
            start_time = time.time()
            
            response: ChatCompletion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert log parsing assistant. "
                                   "Extract structured templates from log messages."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            
            elapsed = time.time() - start_time
            
            # Update statistics
            if response.usage:
                self.total_tokens += response.usage.total_tokens
                self.total_cost += self._estimate_cost(response.usage.total_tokens)
            
            content = response.choices[0].message.content
            
            logger.debug(
                f"LLM call successful (took {elapsed:.2f}s, "
                f"tokens: {response.usage.total_tokens if response.usage else 'N/A'})"
            )
            
            return content.strip()
            
        except Exception as e:
            logger.warning(f"LLM API call failed: {e}")
            
            if retries < self.config.max_retries:
                wait_time = self.config.retry_delay * (2 ** retries)
                logger.info(f"Retrying in {wait_time}s... (attempt {retries + 1})")
                time.sleep(wait_time)
                return self._call_llm(prompt, retries + 1)
            else:
                logger.error(f"LLM API call failed after {self.config.max_retries} retries")
                raise
    
    async def _call_llm_async(self, prompt: str, retries: int = 0) -> str:
        """Make asynchronous call to LLM API."""
        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert log parsing assistant."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            if response.usage:
                self.total_tokens += response.usage.total_tokens
                self.total_cost += self._estimate_cost(response.usage.total_tokens)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            if retries < self.config.max_retries:
                wait_time = self.config.retry_delay * (2 ** retries)
                await asyncio.sleep(wait_time)
                return await self._call_llm_async(prompt, retries + 1)
            else:
                raise
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse batch response to extract individual templates.
        
        Args:
            response: LLM response
            expected_count: Expected number of templates
            
        Returns:
            List of templates
        """
        templates = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Remove numbering and extra formatting
            if line and not line.startswith('#'):
                # Remove leading numbers like "1. " or "1) "
                import re
                clean_line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if clean_line:
                    templates.append(clean_line)
        
        # Ensure we have the expected number
        if len(templates) < expected_count:
            logger.warning(
                f"Batch response had {len(templates)} templates, "
                f"expected {expected_count}"
            )
        
        return templates[:expected_count]
    
    def _estimate_cost(self, tokens: int) -> float:
        """
        Estimate API call cost based on tokens.
        
        Args:
            tokens: Number of tokens used
            
        Returns:
            Estimated cost in USD
        """
        # Pricing as of 2024 (approximate)
        pricing = {
            "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},  # per 1K tokens
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        }
        
        model_key = self.config.model
        if model_key not in pricing:
            # Use conservative estimate
            model_key = "gpt-4"
        
        # Assume 50/50 split for input/output
        cost_per_1k = (pricing[model_key]["input"] + pricing[model_key]["output"]) / 2
        return (tokens / 1000) * cost_per_1k
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.total_cost, 4),
            "avg_tokens_per_call": (
                self.total_tokens / self.total_calls if self.total_calls > 0 else 0
            )
        }
    
    def reset_statistics(self):
        """Reset usage statistics."""
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0


# Example usage
if __name__ == "__main__":
    from ..core.config_manager import LLMConfig
    
    config = LLMConfig(
        provider="openai",
        api_key="your-api-key-here",
        model="gpt-3.5-turbo"
    )
    
    client = LLMClient(config)
    
    # Test single extraction
    log = "2024-01-01 10:00:00 ERROR Failed to connect to database server1"
    template = client.extract_template(log)
    print(f"Template: {template}")
    
    # Print statistics
    print(client.get_statistics())