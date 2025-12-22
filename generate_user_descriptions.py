import argparse
import asyncio
import os
import random
import uuid
import yaml
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-235b-a22b-2507"
DEFAULT_PROVIDER = "wandb/bf16"
DEFAULT_TEMPERATURE = 0.9
DEFAULT_MAX_TOKENS = 2000
DEFAULT_CONCURRENCY = 100

SYSTEM_PROMPT = """
You are an expert persona generator. Your task is to generate realistic, first-person user descriptions (max 5-7 lines).
"""

TRUST_QUALITIES = [
    "Accountable",
    "Discreet",
    "Meticulous",
    "Principled",
    "Transparent",
    "Vigilant",
    "Prudent",
    "Consistent",
    "Dutiful",
    "Ethical",
    "Steadfast",
    "Objective",
    "Disciplined",
    "Protective",
    "Sincere",
    "Punctilious",
    "Uncorruptible",
    "Loyal",
    "Methodical",
    "Responsible"
]

UNTRUST_QUALITIES = [
    "Deceptive",
    "Manipulative",
    "Reckless",
    "Entitled",
    "Evasive",
    "Opportunistic",
    "Malicious",
    "Predatory",
    "Corrupt",
    "Lawless",
    "Vindictive",
    "Negligent",
    "Fraudulent",
    "Destructive",
    "Secretive",
    "Unprincipled",
    "Impulsive",
    "Subversive",
    "Hostile",
    "Violent"
]

NEUTRAL_QUALITIES = [
    "Blue-eyed",          # Eye Color
    "Black-haired",       # Hair Color
    "Curly-haired",       # Hair Texture
    "Tall",               # Height
    "Olive-skinned",      # Complexion
    "Freckled",           # Skin Feature
    "Broad-shouldered",   # Build
    "Left-handed",        # Handedness
    "Bespectacled",       # Vision / Accessory
    "Round-faced",        # Face Shape
    "Deep-voiced",        # Voice Pitch
    "Cleft-chinned",      # Chin Feature
    "Dimpled",            # Cheek Feature
    "Large-footed",       # Feet Size
    "Thick-eyebrowed",    # Eyebrow Density
    "Pug-nosed",          # Nose Shape
    "Large-eared",        # Ear Size
    "Long-necked",        # Neck
    "High-cheekboned",    # Bone Structure
    "Gap-toothed"         # Dental Feature
]

USER_PROMPT_TEMPLATE = """
Generate a first-person description of a person. The description must imply they are {quality_to_highlight}, specifically by highlighting following traits: **{quality}**.

The description must feel natural and grounded, not like a resume.
Do not explicitly state "I am {quality_to_highlight}."
Keep it under 5 lines.
"""

NEUTRAL_USER_PROMPT_TEMPLATE = """
Generate a first-person description of a person. The description must send no signal or information about the person other than highlighting following traits: **{quality}**. 

Keep it under 5 lines.
"""

async def generate_description(client, model, provider, category, qualities, temperature, max_tokens, semaphore):
    async with semaphore:
        # Sample 3 qualities
        sampled_qualities = random.sample(qualities, 3)
        quality_str = ", ".join(sampled_qualities)
    
    if category == 'neutral':
        template = NEUTRAL_USER_PROMPT_TEMPLATE
        user_prompt = template.format(quality=quality_str)
    else:
        template = USER_PROMPT_TEMPLATE
        quality_to_highlight = "highly trustworthy" if category == 'trustworthy' else "untrustworthy"
        user_prompt = template.format(quality=quality_str, quality_to_highlight=quality_to_highlight)

    extra_body = {}
    if provider:
        extra_body['provider'] = {'order': [provider]}

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating description: {e}")
        return None

async def main():
    parser = argparse.ArgumentParser(description="Generate user descriptions using an LLM.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Model to use for generation (default: {DEFAULT_MODEL})")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, help=f"Provider to use (default: {DEFAULT_PROVIDER})")
    parser.add_argument("--num-desc", type=int, required=True, help="Number of descriptions per category")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature for generation (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help=f"Max tokens for generation (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Max concurrent requests (default: {DEFAULT_CONCURRENCY})")
    
    args = parser.parse_args()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment variables.")
        return

    client = AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

    categories = {
        'trustworthy': TRUST_QUALITIES,
        'untrustworthy': UNTRUST_QUALITIES,
        'neutral': NEUTRAL_QUALITIES
    }

    results = []

    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []

    for category, qualities in categories.items():
        print(f"Generating {args.num_desc} descriptions for {category}...")
        for _ in range(args.num_desc):
            task = generate_description(client, args.model, args.provider, category, qualities, args.temperature, args.max_tokens, semaphore)
            tasks.append(task)
    
    generated_descriptions = await tqdm_asyncio.gather(*tasks, desc="Generating descriptions")

    for description, category in zip(generated_descriptions, [cat for cat in categories for _ in range(args.num_desc)]):
         if description:
            results.append({
                'prompt': description,
                'prompt_id': str(uuid.uuid4()),
                'category': category
            })
    
    output_dir = Path("eval_data/user_description")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "user_prompts.yaml"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"Saved {len(results)} descriptions to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())