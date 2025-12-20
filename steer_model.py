import os
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from nnsight import LanguageModel
from tqdm import tqdm

from typing import List, Union, Tuple, Literal, Any, cast

import yaml

from steer_utils import create_user_token_mask, create_system_token_mask, apply_steering_to_layer, get_model_info, prepare_steering_vectors, load_steering_vectors, load_prompts_with_metadata


def calculate_steering_params(
    layer_range: tuple[int, int],
    num_layers: int,
    effective_strength: float | List[float]
) -> tuple[list[int], float | List[float]]:
    """
    Calculate which layers to steer and the per-layer multiplier.
    """
    # --- 1. Handle edge cases ---
    assert num_layers <= layer_range[1] - layer_range[0]

    if num_layers <= 0:
        return [], 0.0

    # --- 2. Calculate the per-layer multiplier ---
    if isinstance(effective_strength, list):
        if len(effective_strength) != num_layers:
            raise ValueError(f"Length of multipliers ({len(effective_strength)}) must match num_layers ({num_layers})")
        multiplier = effective_strength
    else:
        multiplier = effective_strength / num_layers

    # --- 3. Select evenly spaced layers ---
    start_layer, end_layer = layer_range

    # If only one layer is requested, pick the middle of the range
    if num_layers == 1:
        middle_layer = round((start_layer + end_layer) / 2)
        return [middle_layer], multiplier

    # For multiple layers, calculate the step size to space them out evenly
    # We divide by (num_layers - 1) to ensure the first and last points
    # land on the start and end of the range.
    step = (end_layer - start_layer) / (num_layers - 1)

    layers_to_steer = [
        round(start_layer + i * step) for i in range(num_layers)
    ]

    return layers_to_steer, multiplier


# We'll inline the steer_and_generate function here


def steer_and_generate(
    prompt_list: List[str],
    system_prompt: Union[str, List[str]],
    lma,
    tokenizer,
    steering_vectors: dict[torch.Tensor, float] | None,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    layer_to_steer: int | Literal['all'] | List[int],
    d_model: int,
    steer_on_user: bool,
    steer_on_generation: bool,
    steer_on_system: bool,
    top_p: float,
    resdir: str,
    source_files: List[str],
    seed: int,
    backup_every: int = 1,
) -> Tuple[List[str], List[str], List[Any], List[Any]]:
    """
    Generate steered responses for a list of prompts with optional user-token-only steering.
    
    Args:
        prompt_list: List of prompts to process
        system_prompt: System prompt(s) to use - can be a single string or list of strings
        lma: LanguageModel instance
        tokenizer: AutoTokenizer instance
        steering_vectors: Dict mapping tensors to their multipliers (default: None)
        batch_size: Number of prompts to process in each batch
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Temperature for generation
        layer_to_steer: Layer(s) to apply steering to
        d_model: Model dimension
        steer_on_user: Whether to steer on user prompt tokens
        steer_on_generation: Whether to steer on generation tokens
        steer_on_system: Whether to steer on system prompt tokens
        top_p: Nucleus sampling parameter
        resdir: Results directory
        source_files: List of source filenames for each prompt
        seed: Random seed for generation
        backup_every: How often to save results
        
    Returns:
        Tuple of (full_responses, model_only_responses, tok_batches, out_steered_list)
    """
    
    # Create output directory
    os.makedirs(resdir, exist_ok=True)
    
    # Check which files already exist and verify checksums
    indices_to_process = []
    # Check which files already exist
    indices_to_process = []
    for idx, source_file in enumerate(source_files):
        # Create output filename based on source file with seed
        # Use basename to match save logic
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        out_filename = f"{base_name}_seed_{seed}_steer_out.yaml"
        out_path = os.path.join(resdir, out_filename)
        
        should_process = True
        
        if os.path.exists(out_path):
            # File exists, skip
            should_process = False
        
        if should_process:
            indices_to_process.append(idx)
    
    if len(indices_to_process) == 0:
        print("All files already processed. Skipping...")
        return [], [], [], []
    
    print(f"Processing {len(indices_to_process)} / {len(prompt_list)} prompts (skipping {len(prompt_list) - len(indices_to_process)} existing)")
    print(f"Output directory: {resdir}")
    
    # Filter prompts to only those that need processing
    prompt_list_filtered = [prompt_list[i] for i in indices_to_process]
    source_files_filtered = [source_files[i] for i in indices_to_process]
    
    if isinstance(system_prompt, list):
        system_prompt_filtered = [system_prompt[i] for i in indices_to_process]
    else:
        system_prompt_filtered = system_prompt
    
    layers, model_len, is_ft, embed = get_model_info(lma)
    total_steering, steering_vec_list = prepare_steering_vectors(
        steering_vectors, layer_to_steer, d_model, model_len
    )

    # Truncate user prompts to 1024 tokens from right
    max_tokens = 1024
    truncated_prompts = []
    for p in prompt_list_filtered:
        p_tokens = tokenizer.encode(p, add_special_tokens=False)
        if len(p_tokens) > max_tokens:
            truncated_tokens = p_tokens[:max_tokens]
            truncated_p = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            truncated_prompts.append(truncated_p)
        else:
            truncated_prompts.append(p)
    
    # Format prompts with chat template
    formatted_string_list = []
    if isinstance(system_prompt_filtered, str):
        for p in truncated_prompts:
            conversation = []
            if system_prompt_filtered:
                conversation.append({"role": "system", "content": system_prompt_filtered})
            conversation.append({"role": "user", "content": p})
            
            question_string = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_string_list.append(question_string)
    else:
        assert len(system_prompt_filtered) == len(truncated_prompts)
        for p, sys_p in zip(truncated_prompts, system_prompt_filtered):
            conversation = []
            if sys_p:
                conversation.append({"role": "system", "content": sys_p})
            conversation.append({"role": "user", "content": p})
            
            question_string = tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_string_list.append(question_string)
    
    # Create batches
    tok_batches = []
    prompt_batches = []
    system_prompt_batches: List[Union[str, List[str]]] = []
    batch_indices_list = []
    batch_source_files = []
    
    for i in range(0, len(formatted_string_list), batch_size):
        batch_strings = formatted_string_list[i:i+batch_size]
        batch_prompts = prompt_list_filtered[i:i+batch_size]
        batch_indices = list(range(i, min(i+batch_size, len(prompt_list_filtered))))
        batch_sources = source_files_filtered[i:i+batch_size]
        
        # Get system prompts for this batch
        batch_system_prompts: Union[str, List[str]]
        if isinstance(system_prompt_filtered, str):
            batch_system_prompts = system_prompt_filtered
        else:
            batch_system_prompts = system_prompt_filtered[i:i+batch_size]
        
        tok_batch = tokenizer(
            batch_strings, 
            add_special_tokens=False, 
            return_tensors="pt", 
            padding=True,
            padding_side="left",
        ).to("cuda")
        
        tok_batches.append(tok_batch)
        prompt_batches.append(batch_prompts)
        system_prompt_batches.append(batch_system_prompts)
        batch_indices_list.append(batch_indices)
        batch_source_files.append(batch_sources)
    
    full_responses = []
    model_only_responses = []
    out_steered_list = []
    
    batch_iterator = zip(
        tok_batches, 
        prompt_batches, 
        system_prompt_batches, 
        batch_indices_list,
        batch_source_files
    )
    
    # Create a single tqdm instance for batches
    # Create a single tqdm instance for batches
    with tqdm(total=len(tok_batches), desc="Processing Batches") as batch_pbar:
        for b_idx, (tok_batch, prompt_batch, sys_prompt_batch, batch_indices, batch_sources) in enumerate(batch_iterator):
            tokens_generated = 0
            
            def update_postfix_callback():
                nonlocal tokens_generated
                tokens_generated += 1
                if tokens_generated % 10 == 0:  # Refresh occasionally to avoid visual lag
                    batch_pbar.set_postfix_str(f"Tokens: {tokens_generated}/{max_new_tokens}", refresh=False)
                    batch_pbar.refresh()
            
            batch_pbar.set_description(f"Batch {b_idx+1}/{len(tok_batches)}")
            batch_pbar.set_postfix_str("Generating...", refresh=True)
            
            # Create user token mask if steering is enabled and user-only steering is requested
            user_mask = None
            if steering_vectors is not None and steer_on_user:
                user_mask = create_user_token_mask(prompt_batch, tok_batch, tokenizer)
            
            # Create system token mask if steering is enabled and system-only steering is requested
            system_mask = None
            if steering_vectors is not None and steer_on_system:
                system_mask = create_system_token_mask(sys_prompt_batch, tok_batch, tokenizer)
            
            # Generate with or without steering
            if steering_vectors is None:
                with lma.generate(tok_batch, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id, top_p=top_p) as gen:
                    out_steered = lma.generator.output.save()
            else:
                steering_vec_list = cast(List[torch.Tensor], steering_vec_list)
                
                # Create combined mask for user and/or system tokens
                combined_mask = None
                if steer_on_user and steer_on_system:
                    user_mask = cast(torch.Tensor, user_mask)
                    system_mask = cast(torch.Tensor, system_mask)
                    combined_mask = torch.logical_or(user_mask, system_mask)
                elif steer_on_user:
                    combined_mask = cast(torch.Tensor, user_mask)
                elif steer_on_system:
                    combined_mask = cast(torch.Tensor, system_mask)
                
                with lma.generate(tok_batch, max_new_tokens=max_new_tokens, temperature=temperature, pad_token_id=tokenizer.pad_token_id, top_p=top_p) as gen:
                    for j in range(max_new_tokens):
                        batch_pbar.set_postfix_str(f"Tokens: {j}", refresh=True)
                        
                        if j == 0:
                            # Prompt processing
                            # Apply steering to user/system tokens if requested
                            if (steer_on_user or steer_on_system) and combined_mask is not None:
                                if layer_to_steer == 'all':
                                    for i in range(model_len):
                                        apply_steering_to_layer(layers[i], steering_vec_list[i], combined_mask)
                                        layers[i].next()
                                elif isinstance(layer_to_steer, list):
                                    for i in layer_to_steer:
                                        apply_steering_to_layer(layers[i], steering_vec_list[i], combined_mask)
                                        layers[i].next()
                                else:
                                    apply_steering_to_layer(layers[layer_to_steer], total_steering, combined_mask)
                                    layers[layer_to_steer].next()
                            else:
                                if layer_to_steer == 'all':
                                    for i in range(model_len):
                                        layers[i].next()
                                elif isinstance(layer_to_steer, list):
                                    for i in layer_to_steer:
                                        layers[i].next()
                                else:
                                    layers[layer_to_steer].next()
                        else:
                            if steer_on_generation:
                                if layer_to_steer == 'all':
                                    for i in range(model_len):
                                        layers[i].output[:,:,:] = layers[i].output[:,:,:] + steering_vec_list[i].unsqueeze(0).unsqueeze(0)
                                        layers[i].next()
                                elif isinstance(layer_to_steer, list):
                                    for i in layer_to_steer:
                                        layers[i].output[:,:,:] = layers[i].output[:,:,:] + steering_vec_list[i].unsqueeze(0).unsqueeze(0)
                                        layers[i].next()
                                else:
                                    layers[layer_to_steer].output[:,:,:] = layers[layer_to_steer].output[:,:,:] + total_steering.unsqueeze(0).unsqueeze(0)
                                    layers[layer_to_steer].next()
                            
                            embed.next()
                    
                    out_steered = lma.generator.output.save()
            
            batch_pbar.set_postfix_str("Decoding & Saving...", refresh=True)
            
            out_steered_list.append(out_steered)
            
            # Decode responses
            full_decoded = tokenizer.batch_decode(out_steered)
            full_decoded = [d.replace(tokenizer.eos_token, '').replace('<|end_of_text|>', '') for d in full_decoded]
            full_responses.extend(full_decoded)
            
            # Decode model-only responses
            model_only_decoded = []
            for i, full_output in enumerate(out_steered):
                input_length = tok_batch['input_ids'][i].shape[0]
                model_only_output = tokenizer.decode(full_output[input_length:])
                model_only_output = model_only_output.replace(tokenizer.eos_token, '').replace('<|end_of_text|>', '')
                model_only_decoded.append(model_only_output)
            
            model_only_responses.extend(model_only_decoded)
            torch.cuda.empty_cache()
            
            # Save results with source file-based naming
            if (b_idx + 1) % backup_every == 0:
                for i, local_idx in enumerate(batch_indices):
                    source_file = batch_sources[i]
                    
                    # Create output filename based on source file with seed
                    base_name = os.path.splitext(os.path.basename(source_file))[0]
                    out_filename = f"{base_name}_seed_{seed}_steer_out.yaml"
                    out_path = os.path.join(resdir, out_filename)
                    
                    # Gather system prompt for this sample
                    sys_p = sys_prompt_batch if isinstance(sys_prompt_batch, str) else sys_prompt_batch[i]
                    
                    data = {
                        "source_file": source_file,
                        "seed": seed,
                        "prompt": prompt_batch[i],
                        "system_prompt": sys_p,
                        "full_response": full_decoded[i],
                        "model_only_response": model_only_decoded[i],
                    }
                    
                    with open(out_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            
            batch_pbar.update(1)
    
    return full_responses, model_only_responses, tok_batches, out_steered_list


def main():
    parser = argparse.ArgumentParser(description='Steer model generation with steering vectors')

    # === Model & Data Paths ===
    parser.add_argument('--model', type=str, required=True, help='Model identifier (e.g., qwen_qwen3-32b)')
    parser.add_argument('--model_path', type=str, required=True, help='HuggingFace model path')
    parser.add_argument('--vec_type', type=str, required=True, help='Steering vector type (e.g., crosscoder)')
    parser.add_argument('--out_dir', type=str, required=True, help='Base output directory (will create: out_dir/model/vec_type/dataset_name)')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name (in eval_data/)')
    parser.add_argument('--eval_data_dir', type=str, default='eval_data', help='Base directory for evaluation data')
    parser.add_argument('--probe_dir', type=str, default='probe_checkpoints', help='Base directory for probe checkpoints')
    parser.add_argument('--refusal_dir', type=str, default='refusal_svs', help='Base directory for refusal vectors')
    parser.add_argument('--prompt_category', type=str, default=None, help='Prompt category for trust vectors (e.g. refusal/no_refusal)')
    parser.add_argument('--act_name', type=str, default=None, help='Activation name for trust vectors')

    # === Data Type & Randomness ===
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='Random seed(s) for generation/vector creation (can provide multiple)')

    # === Generation Settings ===
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_new_tokens', type=int, default=3000)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.95)

    # === Steering Configuration ===
    parser.add_argument('--d_model', type=int, required=True)
    parser.add_argument('--model_len', type=int, required=True)
    parser.add_argument('--steer_on_user', action='store_true', default=True)
    parser.add_argument('--steer_on_generation', action='store_true', default=True)
    parser.add_argument('--steer_on_system', action='store_true')

    # === Steering parameters ===
    parser.add_argument('--layer_range', type=int, nargs=2, required=True)
    parser.add_argument('--num_layers', type=int, nargs='+', required=True)
    parser.add_argument('--strength', type=float, nargs='+', help='Total steering strength to divide across layers (ignored if exact_multipliers used)')
    parser.add_argument('--exact_multipliers', type=float, nargs='+', help='Exact multipliers for each layer (overrides strength)')
    parser.add_argument('--random-vector', action='store_true', help='Generate random vectors with same mean/std as originals')

    args = parser.parse_args()

    if args.temperature > 0:
        print(f"\nWARNING: Temperature is set to {args.temperature}. It is recommended to use 0 for greedy decoding.\n")

    # Construct paths based on model, vec_type, etc.
    # out_dir/model/vec_type/dataset_name
    full_out_dir = os.path.join(args.out_dir, args.model, args.vec_type, args.dataset_name)
    
    # Set layer range and steering parameters
    layer_range = (args.layer_range[0], args.layer_range[1])
    num_layers = args.num_layers
    
    if args.exact_multipliers:
        strength = [args.exact_multipliers] # Treat as one configuration
        print(f"Using exact multipliers: {args.exact_multipliers}")
    else:
        if args.strength is None:
             parser.error("--strength is required unless --exact_multipliers is specified")
        strength = args.strength
        
    seeds = args.seed if isinstance(args.seed, list) else [args.seed]

    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Model Path: {args.model_path}")
    print(f"Model Length: {args.model_len}")
    print(f"Model Dimension: {args.d_model}")
    print(f"Dtype: {args.dtype}")
    print(f"Vec Type: {args.vec_type}")
    print(f"Dataset: {args.dataset_name}")
    print(f"\nSteering Configuration:")
    print(f"  - Steer on user: {args.steer_on_user}")
    print(f"  - Steer on generation: {args.steer_on_generation}")
    print(f"  - Steer on system: {args.steer_on_system}")
    print(f"  - Random vector: {args.random_vector}")
    print(f"\nSteering Parameters:")
    print(f"  - Layer range: {layer_range}")
    print(f"  - Num layers: {num_layers}")
    print(f"  - Strengths: {strength}")
    print(f"\nGeneration Settings:")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Max new tokens: {args.max_new_tokens}")
    print(f"  - Temperature: {args.temperature}")
    print(f"  - Top-p: {args.top_p}")
    print(f"  - Seeds: {seeds}")
    print(f"\nPaths:")
    print(f"  - Base output dir: {args.out_dir}")
    print(f"  - Full output dir: {full_out_dir}")
    print("="*60 + "\n")

    # Set dtype
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"

    # Wrap with nnsight
    nnmodel = LanguageModel(model, tokenizer=tokenizer)

    # Load prompts
    # Use load_prompts_with_metadata from steer_utils
    sys_prompts, user_prompts, prompt_ids = load_prompts_with_metadata(args.dataset_name, args.eval_data_dir)
    sources = prompt_ids

    # Run steering
    with torch.no_grad():
        # Run steering for each configuration (seed, num_layers, strength)
        for seed in seeds:
            # Set random seed for this iteration
            set_seed(seed)
            print(f"\n{'='*60}")
            print(f"Set random seed to: {seed}")
            print(f"{'='*60}\n")
            
            for n in num_layers:
                for s in strength:
                    print(f"\n{'='*60}")
                    print(f"Running configuration: Seed={seed}, N={n}, S={s}")
                    print(f"{'='*60}\n")
                    
                    layers, multiplier = calculate_steering_params(layer_range, n, s)
                    
                    print(f"Layers to steer: {layers}")
                    if isinstance(multiplier, list):
                        print(f"Per-layer multipliers: {multiplier}")
                    else:
                        print(f"Per-layer multiplier: {multiplier:.4f}")

                    steering_vecs = load_steering_vectors(
                        model_name=args.model,
                        vec_type=args.vec_type,
                        layer_indices=layers,
                        d_model=args.d_model,
                        model_len=args.model_len,
                        multiplier=multiplier,
                        prompt_category=args.prompt_category,
                        act_name=args.act_name,
                        probe_dir=args.probe_dir,
                        refusal_dir=args.refusal_dir,
                    )

                    # Create results directory with layer/strength config
                    # Format layers for directory name
                    if len(layers) == 1:
                        layers_str = str(layers[0])
                    else:
                        layers_str = "_".join(map(str, layers))
                        
                    if args.random_vector:
                        config_resdir = os.path.join(full_out_dir, f"L_{layers_str}_S_{s}_random")
                    else:
                        config_resdir = os.path.join(full_out_dir, f"L_{layers_str}_S_{s}")
                    print(f"Results directory: {config_resdir}")

                    ans = steer_and_generate(
                        prompt_list=user_prompts,
                        system_prompt=sys_prompts,
                        lma=nnmodel,
                        tokenizer=tokenizer,
                        steering_vectors=steering_vecs,
                        layer_to_steer=layers,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        d_model=args.d_model,
                        steer_on_user=args.steer_on_user,
                        steer_on_generation=args.steer_on_generation,
                        steer_on_system=args.steer_on_system,
                        top_p=args.top_p,
                        resdir=config_resdir,
                        source_files=sources,
                        seed=seed,
                    )


if __name__ == "__main__":
    main()