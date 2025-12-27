#!/usr/bin/env python3
"""
Judge model outputs from steered-outs directory using an LLM judge.

This script loads model generations from the steered-outs directory,
judges them using an LLM API, and saves the results in a structured format.

Usage:
    python judge_outputs.py --model llama --vec-type refusal --dataset harmful --layer-id 11 --strength 1.0
    python judge_outputs.py --model llama --vec-type control --dataset wildjb_benign --layer-id 11 --strength 0.0 --judge-model anthropic/claude-3.5-sonnet
"""

import argparse
import asyncio
import json
import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from openai import AsyncOpenAI
from judge_prompts import (
    SYSTEM_PROMPT_JUDGE,
    USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT_BENIGN_JUDGE,
    USER_PROMPT_TEMPLATE_BENIGN,
    SYSTEM_PROMPT_DESC_JUDGE,
    USER_PROMPT_TEMPLATE_DESC
)

load_dotenv()

# Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"
DEFAULT_PROVIDER = "chutes/bf16"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2000
DEFAULT_CONCURRENCY = 100


def get_provider_config(provider: Optional[str] = None) -> Dict:
    """Get provider-specific configuration for OpenRouter."""
    extra_body = {}
    if provider:
        providers = [p.strip() for p in provider.split(',')]
        extra_body["provider"] = {"order": providers}
    return extra_body


def load_category_mapping(dataset: str) -> Dict[str, Optional[str]]:
    """
    Load prompt_id to category mapping from user_prompts.yaml.
    
    Args:
        dataset: Dataset name (e.g., 'harmful', 'wildjb_benign')
    
    Returns:
        Dictionary mapping prompt_id to category (or None if category field doesn't exist)
    """
    yaml_path = Path(f"eval_data/{dataset}/user_prompts.yaml")
    
    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found. Categories will be None.")
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            prompts = yaml.safe_load(f)
        
        category_map = {}
        for prompt_entry in prompts:
            prompt_id = prompt_entry.get('prompt_id')
            category = prompt_entry.get('category')  # May be None if field doesn't exist
            if prompt_id:
                category_map[prompt_id] = category
        
        return category_map
    
    except Exception as e:
        print(f"Error loading {yaml_path}: {e}")
        return {}


def load_generation_files(model: str, vec_type: str, dataset: str, layer_id: Optional[str] = None, strength: Optional[str] = None, prompt_category: Optional[str] = None) -> List[Dict]:
    """
    Load all generation files from steered-outs directory.
    
    Args:
        model: Model name (e.g., 'llama')
        vec_type: Vector type (e.g., 'refusal', 'control', 'trust_complete')
        dataset: Dataset name (e.g., 'harmful', 'wildjb_benign')
        layer_id: Layer ID (e.g., '11' or '11-23-34'). Optional for control vec type.
        strength: Steering strength (e.g., '1.0' or '0.5'). Optional for control vec type.
        prompt_category: Prompt category for trust vec types (e.g., 'refusal', 'no_refusal')
    
    Returns:
        List of dictionaries containing generation data
    """
    # Build path based on vec_type
    if vec_type == 'control':
        # For control: steered-outs/{model}/{vec_type}/{dataset}/
        base_path = Path(f"steered-outs/{model}/{vec_type}/{dataset}")
    elif vec_type in ['trust_complete', 'trust_orthogonal']:
        # For trust_complete and trust_orthogonal: steered-outs/{model}/{vec_type}/{dataset}/{prompt_category}/L_{layer_id}_S_{strength}/
        if not prompt_category:
            print(f"Error: --prompt-category is required for vec_type '{vec_type}'")
            return []
        base_path = Path(f"steered-outs/{model}/{vec_type}/{dataset}/{prompt_category}/L_{layer_id}_S_{strength}")
    else:
        # For others: steered-outs/{model}/{vec_type}/{dataset}/L_{layer_id}_S_{strength}/
        base_path = Path(f"steered-outs/{model}/{vec_type}/{dataset}/L_{layer_id}_S_{strength}")
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist.")
        return []
    
    # Find all YAML files
    yaml_files = list(base_path.glob("*.yaml"))
    
    if not yaml_files:
        print(f"Warning: No YAML files found in {base_path}")
        return []
    
    generations = []
    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            # Extract relevant information
            gen_data = {
                'prompt_id': data.get('source_file'),  # UUID from filename
                'prompt': data.get('prompt', ''),
                'model_response': data.get('model_only_response', ''),
                'full_response': data.get('full_response', ''),
                'seed': data.get('seed'),
                'system_prompt': data.get('system_prompt'),
                'file_path': str(yaml_file)
            }
            
            generations.append(gen_data)
        
        except Exception as e:
            print(f"Error loading {yaml_file}: {e}")
            continue
    
    return generations


async def judge_response(
    client: AsyncOpenAI,
    prompt: str,
    model_response: str,
    judge_model: str,
    dataset: str,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    provider: Optional[str] = None,
    reasoning_config: Optional[Dict] = None,
    max_retries: int = 3,
    ground_truth: Optional[str] = None
) -> Optional[tuple[str, Dict]]:
    """
    Judge a single model response using the LLM judge.
    
    Args:
        dataset: Dataset name to determine which prompts to use (benign vs harmful vs user_description)
        reasoning_config: Optional dict with reasoning configuration, e.g.:
            {"effort": "high"} or {"max_tokens": 2000} or {"enabled": True}
        ground_truth: Ground truth category for user_description datasets
    
    Returns:
        Tuple of (reasoning_text, judgment_dict) or None if failed
        The first element contains reasoning from reasoning_details or content before JSON
    """
    # Select prompts based on dataset type
    is_user_description = 'user_description' in dataset.lower()
    is_benign = 'benign' in dataset.lower()
    
    if is_user_description:
        system_prompt = SYSTEM_PROMPT_DESC_JUDGE
        user_prompt_template = USER_PROMPT_TEMPLATE_DESC
        user_prompt = user_prompt_template.format(
            user_description=prompt,
            ground_truth=ground_truth if ground_truth else "Unknown",
            model_response=model_response
        )
    else:
        system_prompt = SYSTEM_PROMPT_BENIGN_JUDGE if is_benign else SYSTEM_PROMPT_JUDGE
        user_prompt_template = USER_PROMPT_TEMPLATE_BENIGN if is_benign else USER_PROMPT_TEMPLATE
        user_prompt = user_prompt_template.format(
            user_prompt=prompt,
            model_response=model_response
        )
    
    for attempt in range(max_retries):
        try:
            extra_body = get_provider_config(provider)
            
            # Add reasoning config if provided
            if reasoning_config:
                extra_body["reasoning"] = reasoning_config
            
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body
            )
            
            if not response or not response.choices:
                raise Exception("Empty response from API")
            
            message = response.choices[0].message
            raw_content = message.content.strip() if message.content else ""
            content = raw_content
            reasoning_text = ""
            
            # First, try to extract reasoning from reasoning_details field (for reasoning models)
            reasoning_details = getattr(message, 'reasoning_details', None)
            if reasoning_details and isinstance(reasoning_details, list):
                # Extract text from reasoning_details array
                reasoning_parts = []
                for detail in reasoning_details:
                    if isinstance(detail, dict):
                        detail_type = detail.get('type', '')
                        if detail_type == 'reasoning.text':
                            text = detail.get('text', '')
                            if text:
                                reasoning_parts.append(text)
                        elif detail_type == 'reasoning.summary':
                            summary = detail.get('summary', '')
                            if summary:
                                reasoning_parts.append(f"[Summary] {summary}")
                
                if reasoning_parts:
                    reasoning_text = '\n\n'.join(reasoning_parts)
            
            # If no reasoning_details, try to extract from content before JSON
            if not reasoning_text:
                # Extract content before JSON and the JSON itself
                # Sometimes the model might wrap it in markdown code blocks
                if "```json" in content:
                    parts = content.split("```json")
                    reasoning_text = parts[0].strip()
                    content = parts[1].split("```")[0].strip()
                elif "```" in content:
                    parts = content.split("```")
                    reasoning_text = parts[0].strip()
                    content = parts[1].split("```")[0].strip()
                else:
                    # Try to find where JSON starts (looking for opening brace)
                    json_start = content.find('{')
                    if json_start > 0:
                        reasoning_text = content[:json_start].strip()
                        content = content[json_start:].strip()
            
            judgment = json.loads(content)
            return (reasoning_text, judgment)
        
        except json.JSONDecodeError as e:
            print(f"\nJSON parse error (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Response content: {content[:200]}...")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(1)
        
        except Exception as e:
            print(f"\nError judging response (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(1)
    
    return None


def load_existing_judgments(output_dir: Path) -> Set[str]:
    """
    Load already-judged prompt IDs from existing output directory.
    
    Returns:
        Set of prompt_ids that have already been judged
    """
    if not output_dir.exists():
        return set()
    
    try:
        # Find all YAML judgment files
        yaml_files = list(output_dir.glob("*_judgment.yaml"))
        
        # Extract prompt_ids from filenames
        judged_ids = set()
        for yaml_file in yaml_files:
            # Filename format: {prompt_id}_judgment.yaml
            prompt_id = yaml_file.stem.replace('_judgment', '')
            judged_ids.add(prompt_id)
        
        return judged_ids
    
    except Exception as e:
        print(f"Error loading existing judgments: {e}")
        return set()


async def process_judgment(
    client: AsyncOpenAI,
    gen_data: Dict,
    category_map: Dict[str, Optional[str]],
    judge_model: str,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
    model: str,
    vec_type: str,
    dataset: str,
    layer_id: Optional[str] = None,
    strength: Optional[str] = None,
    provider: Optional[str] = None,
    reasoning_config: Optional[Dict] = None
) -> Optional[Dict]:
    """Process a single judgment with concurrency control and save immediately."""
    async with semaphore:
        # Get ground truth category for user_description datasets
        ground_truth = category_map.get(gen_data['prompt_id'])
        
        result = await judge_response(
            client,
            gen_data['prompt'],
            gen_data['model_response'],
            judge_model,
            dataset,
            provider=provider,
            reasoning_config=reasoning_config,
            ground_truth=ground_truth
        )
        
        if result is None:
            return None
        
        raw_judge_response, judgment = result
        
        # Build output entry
        output_result = {
            'prompt_id': gen_data['prompt_id'],
            'prompt': gen_data['prompt'],
            'model_response': gen_data['model_response'],
            'category': category_map.get(gen_data['prompt_id']),
            'judgment': judgment
        }
        
        # Save immediately to file
        prompt_id = output_result['prompt_id']
        output_file = output_dir / f"{prompt_id}_judgment.yaml"
        
        output_data = {
            'model': model,
            'vec_type': vec_type,
            'dataset': dataset,
        }
        
        # Only include layer_id and strength if they are provided
        if layer_id is not None:
            output_data['layer_id'] = layer_id
        if strength is not None:
            output_data['strength'] = strength
        
        output_data.update({
            'judge_model': judge_model,
            'prompt_id': output_result['prompt_id'],
            'prompt': output_result['prompt'],
            'model_response': output_result['model_response'],
            'category': output_result['category'],
            'judge_raw_response': raw_judge_response,  # Save raw response before JSON parsing
            'judgment': output_result['judgment']
        })
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(output_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"\nError saving {output_file}: {e}")
            return None
        
        return output_result


async def main():
    parser = argparse.ArgumentParser(
        description='Judge model outputs from steered-outs directory'
    )
    parser.add_argument('--model', required=True, help='Model name (e.g., llama)')
    parser.add_argument('--vec-type', required=True, help='Vector type (e.g., refusal, control, trust_complete)')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., harmful, wildjb_benign)')
    parser.add_argument('--layer-id', type=str, default=None, help='Layer ID (e.g., 11 or 11-23-34). Optional for control vec type.')
    parser.add_argument('--strength', type=str, default=None, help='Steering strength (e.g., 1.0 or 0.5). Optional for control vec type.')
    parser.add_argument('--judge-model', default=DEFAULT_JUDGE_MODEL, help=f'Judge model (default: {DEFAULT_JUDGE_MODEL})')
    parser.add_argument('--provider', default=DEFAULT_PROVIDER, help=f'Provider preference (default: {DEFAULT_PROVIDER})')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY, help=f'Parallel API calls (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--num-files', type=int, default=None, help='Limit number of files to judge (default: all files)')
    parser.add_argument('--prompt-category', type=str, default=None, help='Prompt category for trust vec types (e.g., refusal, no_refusal). Required for trust_complete and trust_orthogonal.')
    parser.add_argument('--reasoning', type=str, default=None, 
                        help='Enable reasoning tokens. Options: "high", "medium", "low" for effort levels, or a number for max_tokens (e.g., "2000"), or "true" to enable with defaults')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing judgment files')
    
    args = parser.parse_args()
    
    # Validate arguments based on vec_type
    if args.vec_type != 'control':
        if args.layer_id is None:
            parser.error("--layer-id is required when vec-type is not 'control'")
        if args.strength is None:
            parser.error("--strength is required when vec-type is not 'control'")
    
    # Parse reasoning config
    reasoning_config = None
    if args.reasoning:
        reasoning_arg = args.reasoning.lower()
        if reasoning_arg in ['xhigh', 'high', 'medium', 'low', 'minimal', 'none']:
            reasoning_config = {"effort": reasoning_arg}
        elif reasoning_arg == 'true':
            reasoning_config = {"enabled": True}
        elif reasoning_arg.isdigit():
            reasoning_config = {"max_tokens": int(reasoning_arg)}
        else:
            print(f"Warning: Invalid --reasoning value '{args.reasoning}'. Ignoring.")
    
    print(f"{'='*80}")
    print(f"JUDGING MODEL OUTPUTS")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Vec Type: {args.vec_type}")
    print(f"Dataset: {args.dataset}")
    if args.layer_id is not None:
        print(f"Layer ID: {args.layer_id}")
    if args.strength is not None:
        print(f"Strength: {args.strength}")
    print(f"Judge Model: {args.judge_model}")
    print(f"Provider: {args.provider}")
    print(f"Concurrency: {args.concurrency}")
    if args.prompt_category:
        print(f"Prompt Category: {args.prompt_category}")
    if reasoning_config:
        print(f"Reasoning: {reasoning_config}")
    
    # Check which evaluation prompts to use
    is_user_description_dataset = 'user_description' in args.dataset.lower()
    is_benign_dataset = 'benign' in args.dataset.lower()
    
    if is_user_description_dataset:
        print(f"Evaluation Mode: USER DESCRIPTION (using semantic interpretation prompts)")
    elif is_benign_dataset:
        print(f"Evaluation Mode: BENIGN (using over-refusal detection prompts)")
    else:
        print(f"Evaluation Mode: HARMFUL (using standard safety evaluation prompts)")
    
    print(f"{'='*80}\n")
    
    # Initialize OpenRouter client
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not set in .env file")
        return 1
    
    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key
    )
    
    # Load category mapping
    print("Loading category mapping...")
    category_map = load_category_mapping(args.dataset)
    print(f"Loaded {len(category_map)} prompt categories\n")
    
    # Load generation files
    print("Loading generation files...")
    generations = load_generation_files(
        args.model,
        args.vec_type,
        args.dataset,
        args.layer_id,
        args.strength,
        args.prompt_category
    )
    print(f"Loaded {len(generations)} generation files\n")
    
    if not generations:
        print("No generations to judge. Exiting.")
        return 0
    
    # Limit number of files if specified
    if args.num_files is not None and args.num_files > 0:
        generations = generations[:args.num_files]
        print(f"Limited to {len(generations)} files (--num-files={args.num_files})\n")
    
    # Prepare output directory
    judge_model_safe = args.judge_model.replace('/', '_').replace(':', '_')
    
    # Build output path to match input structure
    if args.vec_type == 'control':
        # For control: judge_outputs/{judge_model}/{model}/{vec_type}/{dataset}/
        output_dir = Path(f"judge_outputs/{judge_model_safe}/{args.model}/{args.vec_type}/{args.dataset}")
    elif args.vec_type in ['trust_complete', 'trust_orthogonal']:
        # For trust types: judge_outputs/{judge_model}/{model}/{vec_type}/{dataset}/{prompt_category}/L_{layer_id}_S_{strength}/
        output_dir = Path(f"judge_outputs/{judge_model_safe}/{args.model}/{args.vec_type}/{args.dataset}/{args.prompt_category}/L_{args.layer_id}_S_{args.strength}")
    else:
        # For others: judge_outputs/{judge_model}/{model}/{vec_type}/{dataset}/L_{layer_id}_S_{strength}/
        output_dir = Path(f"judge_outputs/{judge_model_safe}/{args.model}/{args.vec_type}/{args.dataset}/L_{args.layer_id}_S_{args.strength}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing judgments
    already_judged = set()
    if not args.overwrite:
        print("Checking for existing judgments...")
        already_judged = load_existing_judgments(output_dir)
        print(f"Found {len(already_judged)} already-judged prompts\n")
    else:
        print("Overwrite mode enabled. Existing judgments will be overwritten.\n")
    
    # Filter out already-judged prompts
    to_judge = [g for g in generations if g['prompt_id'] not in already_judged]
    print(f"Prompts to judge: {len(to_judge)}/{len(generations)}\n")
    
    if not to_judge:
        print("All prompts already judged. Exiting.")
        return 0
    
    # Process judgments
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_judgment(
            client,
            gen_data,
            category_map,
            args.judge_model,
            semaphore,
            output_dir,
            args.model,
            args.vec_type,
            args.dataset,
            args.layer_id,
            args.strength,
            args.provider,
            reasoning_config
        )
        for gen_data in to_judge
    ]
    
    results = await tqdm.gather(*tasks, desc="Judging responses")
    
    # Filter out None results (failed judgments)
    successful_results = [r for r in results if r is not None]
    failed_count = len(results) - len(successful_results)
    
    # Count total judgments in directory
    total_judgments = len(list(output_dir.glob("*_judgment.yaml")))
    
    # Summary
    print()
    print(f"{'='*80}")
    print(f"JUDGING COMPLETE")
    print(f"{'='*80}")
    print(f"Previously Judged: {len(already_judged)}")
    print(f"Newly Judged: {len(successful_results)}/{len(to_judge)}")
    if failed_count > 0:
        print(f"Failed: {failed_count}")
    print(f"Total Judgments in Directory: {total_judgments}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*80}")
    
    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))