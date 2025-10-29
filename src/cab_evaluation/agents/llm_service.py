"""LLM service for managing model interactions."""

import json
import os
import time
import logging
from typing import Optional, Dict, Any

import boto3
from botocore.config import Config
from openai import OpenAI
from dotenv import load_dotenv

from ..core.config import CABConfig, ModelConfig
from ..core.exceptions import LLMError, InputTooLongError
from ..prompts.constants import ValidationPatterns

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM model interactions."""
    
    def __init__(self, config: CABConfig):
        """Initialize LLM service.
        
        Args:
            config: CAB configuration
        """
        self.config = config
        
        # Setup AWS Bedrock client
        bedrock_config = Config(
            retries={"max_attempts": 1000, "mode": "standard"},
            connect_timeout=120,
            read_timeout=1200
        )
        self.bedrock_client = boto3.client(
            'bedrock-runtime', 
            config=bedrock_config, 
            region_name='us-west-2'
        )
        
        # Load environment variables for OpenAI
        load_dotenv()
        
        # Initialize OpenAI client cache
        self._openai_clients: Dict[str, OpenAI] = {}
    
    def _get_openai_client(self, api_key_env_var: str) -> OpenAI:
        """Get or create OpenAI client."""
        if api_key_env_var not in self._openai_clients:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise LLMError(f"Missing environment variable: {api_key_env_var}")
            self._openai_clients[api_key_env_var] = OpenAI(api_key=api_key)
        return self._openai_clients[api_key_env_var]
    
    def _is_input_too_long_error(self, error_message: str) -> bool:
        """Check if error indicates input too long."""
        error_text = str(error_message).lower()
        for pattern in ValidationPatterns.INPUT_TOO_LONG_PATTERNS:
            if pattern.lower() in error_text:
                return True
        return False
    
    def _check_input_size(self, user_prompt: str, system_prompt: str, model_config: ModelConfig):
        """Check if input size exceeds model limits."""
        total_prompt_size = len(user_prompt) + len(system_prompt)
        if total_prompt_size > model_config.max_tokens:
            raise InputTooLongError(
                f"Prompt size too large ({total_prompt_size} chars) for model {model_config.name} "
                f"(limit: {model_config.max_tokens})"
            )
    
    async def call_model(
        self,
        user_prompt: str,
        system_prompt: str,
        model_config: ModelConfig,
        agent_type: str = "unknown",
        issue_id: str = "unknown",
        max_retries: int = 1000
    ) -> str:
        """Call LLM model with given prompts and retry logic.
        
        Args:
            user_prompt: User input prompt
            system_prompt: System prompt
            model_config: Model configuration
            agent_type: Type of agent making the call
            issue_id: Issue ID for tracking
            max_retries: Maximum number of retries
            
        Returns:
            Model response text
            
        Raises:
            InputTooLongError: If input exceeds model context window
            LLMError: If model call fails after retries
        """
        # Check input size
        self._check_input_size(user_prompt, system_prompt, model_config)
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                if model_config.provider == "openai":
                    return await self._call_openai_model(
                        user_prompt, system_prompt, model_config
                    )
                else:  # bedrock
                    return await self._call_bedrock_model(
                        user_prompt, system_prompt, model_config
                    )
                    
            except Exception as e:
                error_str = str(e)
                
                # Check for input too long errors
                if self._is_input_too_long_error(error_str):
                    logger.warning(f"Input size error: {error_str}")
                    raise InputTooLongError(error_str)
                
                retry_count += 1
                if retry_count <= max_retries:
                    wait_time = 10
                    logger.warning(
                        f"LLM call failed (attempt {retry_count}/{max_retries}). "
                        f"Retrying in {wait_time:.2f} seconds. Error: {error_str}"
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM call failed after {max_retries} retries: {error_str}")
                    raise LLMError(
                        f"Failed to call {model_config.name} after {max_retries} retries: {error_str}",
                        model_name=model_config.name,
                        retry_count=retry_count
                    )
    
    async def _call_openai_model(
        self,
        user_prompt: str,
        system_prompt: str,
        model_config: ModelConfig
    ) -> str:
        """Call OpenAI model."""
        client = self._get_openai_client(model_config.api_key_env_var)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response = client.chat.completions.create(
            model=model_config.model_id,
            messages=messages,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
        )
        
        return response.choices[0].message.content
    
    async def _call_bedrock_model(
        self,
        user_prompt: str,
        system_prompt: str,
        model_config: ModelConfig
    ) -> str:
        """Call Bedrock model."""
        # Determine model type for payload formatting
        is_claude = "anthropic" in model_config.model_id.lower()
        is_llama = "llama" in model_config.model_id.lower() 
        is_deepseek = "deepseek" in model_config.model_id.lower()
        
        if is_claude:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
            }
            
            # Add thinking mode if enabled
            if model_config.thinking_enabled:
                body["thinking"] = {"type": "enabled", "budget_tokens": 30000}
                
        elif is_llama:
            # Use standard Meta chat format
            formatted_prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>"
            body = {
                "prompt": formatted_prompt,
                "max_gen_len": model_config.max_tokens,
                "temperature": model_config.temperature,
            }
        else:  # DeepSeek and others
            body = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": model_config.max_tokens,
                "temperature": model_config.temperature,
            }
        
        response = self.bedrock_client.invoke_model(
            body=json.dumps(body),
            modelId=model_config.model_id,
            accept="application/json",
            contentType="application/json",
        )
        
        response_body = json.loads(response.get("body").read())
        
        # Extract response based on model type
        if is_claude:
            if model_config.thinking_enabled:
                return response_body["content"][1]["text"]
            else:
                return response_body["content"][0]["text"]
        elif is_llama:
            raw_response = response_body.get("generation", response_body.get("completion", ""))
            # Clean up the response by removing end tokens
            if "<|end|>" in raw_response:
                return raw_response.split("<|end|>")[0].strip()
            return raw_response.strip()
        else:  # DeepSeek format
            return response_body["choices"][0]["message"]["content"]
