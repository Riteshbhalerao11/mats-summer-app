#!/usr/bin/env python3
"""
Generate synthetic conversation data for information_sharing and dual-use scenarios.

This script generates multi-turn conversations using LLM APIs to create training data
for AI safety research. It supports both information_sharing (sensitive information requests)
and dual-use technology scenarios.

Usage:
    python generate_conv_data.py --scenario information_sharing --trust-level high_trust
    python generate_conv_data.py --scenario dual_use --trust-level low_trust --limit 10
    python generate_conv_data.py --scenario information_sharing --trust-level high_trust --model anthropic/claude-3.5-sonnet
    python generate_conv_data.py --scenario dual_use --trust-level low_trust --concurrency 5
"""

import asyncio
import argparse
import json
import hashlib
import traceback
import random
from pathlib import Path
from typing import Dict, List, Optional
from tqdm.asyncio import tqdm
from datetime import datetime
import os
import re
import uuid

from openai import AsyncOpenAI
from dotenv import load_dotenv
import prompts

load_dotenv()

# Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-235b-a22b-thinking-2507:free"
DEFAULT_PROVIDER = "wandb/bf16"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 4000
DEFAULT_CONCURRENCY = 10


def calculate_checksum(text: str) -> str:
    """Calculate SHA256 checksum for text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_model_safe_name(model: str) -> str:
    """Convert model name to filesystem-safe string."""
    return model.replace('/', '_').replace(':', '_')


def get_provider_config(provider: Optional[str] = None) -> Dict:
    """Get provider-specific configuration for OpenRouter.
    
    Args:
        provider: Provider preference (e.g., 'wandb/bf16', 'openai', 'anthropic')
                 Can be a single provider or comma-separated list for order preference
    
    Returns:
        Dict with provider configuration for extra_body
    """
    extra_body = {}
    
    if provider:
        # Split by comma if multiple providers specified
        providers = [p.strip() for p in provider.split(',')]
        extra_body["provider"] = {"order": providers}
    
    return extra_body


async def generate_conversation(
    client: AsyncOpenAI,
    scenario_type: str,
    trust_level: str,
    model: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    provider: Optional[str] = None
) -> Dict:
    """Generate a single conversation using the LLM API.
    
    Args:
        client: AsyncOpenAI client instance
        scenario_type: Either 'information_sharing' or 'dual_use'
        trust_level: Either 'high_trust' or 'low_trust'
        model: Model identifier for the API
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        max_retries: Number of retry attempts on failure
        provider: Provider preference for OpenRouter
        
    Returns:
        Dict containing the generated conversation and metadata
    """
    import time
    
    # Get the appropriate prompts
    if scenario_type not in prompts.PROMPT_REGISTRY:
        raise ValueError(f"Invalid scenario type: {scenario_type}. Must be 'information_sharing' or 'dual_use'")
    
    scenario_config = prompts.PROMPT_REGISTRY[scenario_type]
    
    if trust_level not in scenario_config['behaviors']:
        raise ValueError(f"Invalid trust level: {trust_level}. Must be 'high_trust' or 'low_trust'")
    
    behavior_config = scenario_config['behaviors'][trust_level]
    
    # Select a random scenario context from the registry
    scenario_context = random.choice(scenario_config['scenario_contexts'])
    
    # Build the system and user prompts
    system_prompt = scenario_config['system_prompt']
    user_prompt = scenario_config['user_template'].format(
        trust_level=behavior_config['trust_level'],
        behavior_description=behavior_config['behavior_description'],
        scenario_context=scenario_context
    )
    
    for attempt in range(max_retries):
        try:
            extra_body = get_provider_config(provider)
            
            start_time = time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body
            )
            api_time = time.time() - start_time
            
            if not response or not response.choices:
                raise Exception("Empty response from API")
            
            message = response.choices[0].message
            content = message.content.strip() if message.content else ''
            finish_reason = response.choices[0].finish_reason
            
            if not content:
                raise Exception(f"Empty content field (finish_reason: {finish_reason})")
            
            # Parse the conversation
            conversation = parse_conversation(content)
            
            # Calculate checksum
            checksum = calculate_checksum(content)
            
            return {
                'scenario_type': scenario_type,
                'trust_level': trust_level,
                'model': model,
                'scenario_context': scenario_context,
                'checksum': checksum,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'raw_conversation': content,
                'parsed_conversation': conversation,
                'api_time': api_time,
                'finish_reason': finish_reason,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\nERROR generating conversation: {e}")
                raise
            await asyncio.sleep(1)


def parse_conversation(content: str) -> List[Dict[str, str]]:
    """Parse conversation text into structured turns.
    
    Args:
        content: Raw conversation text with ### Human: and ### Assistant: markers
        
    Returns:
        List of dicts with 'role' and 'content' keys
    """
    turns = []
    
    # Split by the markers
    parts = re.split(r'###\s*(Human|Assistant):\s*', content)
    
    # parts[0] is any text before the first marker (usually empty)
    # Then alternating: role, content, role, content, ...
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            role = parts[i].lower()
            content = parts[i + 1].strip()
            
            if role == 'human':
                turns.append({'role': 'user', 'content': content})
            elif role == 'assistant':
                turns.append({'role': 'assistant', 'content': content})
    
    return turns


async def process_conversation_generation(
    client: AsyncOpenAI,
    scenario_type: str,
    trust_level: str,
    model: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    skip_existing: bool = False,
    provider: Optional[str] = None
) -> bool:
    """Process a single conversation generation task.
    
    Args:
        client: AsyncOpenAI client
        scenario_type: Scenario type
        trust_level: Trust level
        model: Model to use
        output_dir: Directory to save output
        semaphore: Concurrency control
        temperature: Sampling temperature
        max_tokens: Max tokens
        skip_existing: Whether to skip if file exists
        provider: Provider preference for OpenRouter
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Generate unique filename with UUID to prevent collisions during concurrent generation
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
        filename = f"{scenario_type}_{trust_level}_{timestamp}_{unique_id}.json"
        output_file = output_dir / filename
        
        # Skip if exists
        if skip_existing and output_file.exists():
            return True
        
        # Generate conversation
        async with semaphore:
            result = await generate_conversation(
                client,
                scenario_type,
                trust_level,
                model,
                temperature,
                max_tokens,
                provider=provider
            )
        
        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR processing conversation generation")
        print(f"{'='*80}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print(f"\nFull traceback:")
        traceback.print_exc()
        print(f"{'='*80}\n")
        return False


def count_existing_conversations(output_dir: Path, scenario_type: str, trust_level: str) -> int:
    """Count existing conversation files in the output directory.
    
    Args:
        output_dir: Directory where conversations are saved
        scenario_type: Scenario type to match
        trust_level: Trust level to match
        
    Returns:
        Number of existing conversation files
    """
    if not output_dir.exists():
        return 0
    
    # Count JSON files matching the pattern
    pattern = f"{scenario_type}_{trust_level}_*.json"
    existing_files = list(output_dir.glob(pattern))
    return len(existing_files)


async def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic conversation data for AI safety research'
    )
    parser.add_argument(
        '--scenario',
        required=True,
        choices=['information_sharing', 'dual_use'],
        help='Type of scenario to generate'
    )
    parser.add_argument(
        '--trust-level',
        required=True,
        choices=['high_trust', 'low_trust'],
        help='Trust level of the user in the conversation'
    )
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'Model to use for generation (default: {DEFAULT_MODEL})'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f'Sampling temperature (default: {DEFAULT_TEMPERATURE})'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f'Maximum tokens in response (default: {DEFAULT_MAX_TOKENS})'
    )
    parser.add_argument(
        '--num-conversations',
        type=int,
        default=5,
        help='Number of conversations to generate (default: 5)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f'Parallel API calls (default: {DEFAULT_CONCURRENCY})'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit total conversations (overrides --num-conversations)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip if output file already exists'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./generated_conversations',
        help='Output directory for generated conversations (default: ./generated_conversations)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default=DEFAULT_PROVIDER,
        help=f'Provider preference for OpenRouter (default: {DEFAULT_PROVIDER}). Can be single provider or comma-separated list for order preference (e.g., "openai,anthropic")'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir) / args.scenario / args.trust_level / get_model_safe_name(args.model)
    
    # Count existing conversations first (before printing header)
    existing_count = count_existing_conversations(output_dir, args.scenario, args.trust_level)
    requested_total = args.limit if args.limit else args.num_conversations
    
    print(f"{'='*80}")
    print(f"SYNTHETIC CONVERSATION GENERATION")
    print(f"{'='*80}")
    print(f"Scenario: {args.scenario}")
    print(f"Trust Level: {args.trust_level}")
    print(f"Model: {args.model}")
    print(f"Provider: {args.provider}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_tokens}")
    print(f"Requested Total: {requested_total}")
    print(f"Already Exists: {existing_count}")
    print(f"To Generate: {max(0, requested_total - existing_count)}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\\n")
    
    # Initialize OpenRouter client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in .env file")
        return 1
    
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key
    )
    
    # Count existing conversations
    existing_count = count_existing_conversations(output_dir, args.scenario, args.trust_level)
    
    # Determine number of conversations to generate
    requested_total = args.limit if args.limit else args.num_conversations
    
    # Calculate how many new conversations we need to generate
    num_conversations = max(0, requested_total - existing_count)
    
    # Early exit if no new conversations needed
    if num_conversations == 0:
        print(f"[OK] Target already reached: {existing_count}/{requested_total} conversations exist.")
        print(f"No new conversations to generate.\n")
        return 0
    
    # Create tasks
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_conversation_generation(
            client,
            args.scenario,
            args.trust_level,
            args.model,
            output_dir,
            semaphore,
            args.temperature,
            args.max_tokens,
            args.skip_existing,
            args.provider
        )
        for _ in range(num_conversations)
    ]
    
    # Process with progress bar
    results = await tqdm.gather(*tasks, desc="Generating conversations")
    
    # Summary
    success = sum(1 for r in results if r)
    failed = len(results) - success
    total_now = existing_count + success
    
    print()
    print(f"{'='*80}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Previously Existed: {existing_count}")
    print(f"Newly Generated: {success}/{len(results)}")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"Total Conversations: {total_now}/{requested_total}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))
