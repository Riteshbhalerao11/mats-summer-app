import os
import argparse
from pathlib import Path

import numpy as np

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnsight import LanguageModel
from nnsight.intervention.envoy import Envoy
from peft import PeftModel
from tqdm import tqdm

import textwrap
from typing import List, Optional, Union, Tuple, Literal, Any, cast

from dataclasses import dataclass
import yaml
import os


def get_chat_markers(tokenizer: AutoTokenizer) -> Tuple[str, str]:
    """
    Get the user and system role markers based on the tokenizer type.
    
    Args:
        tokenizer: The tokenizer to detect markers for
        
    Returns:
        Tuple of (user_marker, system_marker)
        
    Raises:
        RuntimeError: If tokenizer type is not supported (not Llama or Qwen)
    """
    # Check tokenizer name/class
    tokenizer_name = tokenizer.__class__.__name__.lower()
    
    # Also check the name_or_path attribute if available
    model_name = ""
    if hasattr(tokenizer, 'name_or_path'):
        model_name = tokenizer.name_or_path.lower()
    
    # Check for Llama
    if 'llama' in tokenizer_name or 'llama' in model_name:
        return "<|start_header_id|>user", "<|start_header_id|>system"
    
    # Check for Qwen
    if 'qwen' in tokenizer_name or 'qwen' in model_name:
        return "<|im_start|>user", "<|im_start|>system"
    
    # No supported tokenizer found
    raise RuntimeError(
        f"Unsupported tokenizer type. Only Llama and Qwen tokenizers are supported.\n"
        f"Tokenizer class: {tokenizer.__class__.__name__}\n"
        f"Model name/path: {getattr(tokenizer, 'name_or_path', 'unknown')}"
    )




def load_prompts_with_metadata(dataset_name: str, base_dir: str = "eval_data") -> Tuple[List[Optional[str]], List[str], List[str]]:
    """
    Load YAML prompts from {base_dir}/{dataset_name}/ associated with a dataset.
    
    Structure:
        {base_dir}/
            {dataset_name}/
                user_prompts.yaml       <- List of {prompt_id, prompt, ...}
                system_prompts.yaml     <- (Optional) List of {prompt_id, prompt, ...}
    
    Args:
        dataset_name (str): Name of the dataset subdirectory.
        base_dir (str, optional): Base directory containing datasets. Defaults to "eval_data".
    
    Returns:
        tuple: (system_prompts, user_prompts, prompt_ids)
               All lists are aligned. system_prompts may be a list of None if not provided.
    """
    base_path = Path(base_dir) / dataset_name
    
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_path}")
    
    user_path = base_path / "user_prompts.yaml"
    sys_path = base_path / "system_prompts.yaml"
    
    if not user_path.exists():
        raise FileNotFoundError(f"User prompts file not found: {user_path}")
    
    # Load user prompts
    with open(user_path, "r") as f:
        user_data = yaml.safe_load(f)
    
    if not isinstance(user_data, list):
        raise ValueError(f"Expected list in {user_path}, got {type(user_data)}")
    
    # Check if system prompts exist
    sys_data = None
    if sys_path.exists():
        with open(sys_path, "r") as f:
            sys_data = yaml.safe_load(f)
        if not isinstance(sys_data, list):
            raise ValueError(f"Expected list in {sys_path}, got {type(sys_data)}")

    # Index system prompts by prompt_id if they exist
    sys_dict = {}
    if sys_data:
        for entry in sys_data:
            if isinstance(entry, dict) and 'prompt_id' in entry:
                sys_dict[entry['prompt_id']] = entry.get('prompt', '')

    user_prompts = []
    system_prompts = []
    prompt_ids = []
    
    for entry in user_data:
        if not isinstance(entry, dict):
            continue
            
        pid = entry.get('prompt_id')
        if not pid:
            print(f"Warning: User prompt entry missing prompt_id, skipping")
            continue
            
        user_prompts.append(entry.get('prompt', ''))
        prompt_ids.append(pid)
        
        # Match with system prompt if available
        if sys_data:
            system_prompts.append(sys_dict.get(pid, None))
        else:
            system_prompts.append(None)
            
    print(f"Loaded {len(user_prompts)} prompts from {dataset_name}")
    
    return system_prompts, user_prompts, prompt_ids


def create_user_token_mask(
    prompt_batch: List[str],
    formatted_tokens: dict,
    tokenizer: AutoTokenizer
) -> torch.Tensor:
    """
    Create a mask where all tokens from the <user> chat marker
    (including the marker itself) to the end of the sequence are True.

    Args:
        formatted_tokens: Tokenized batch with chat template applied
        tokenizer: Tokenizer used for encoding

    Returns:
        Boolean tensor of shape (batch_size, seq_len)
    """
    batch_size, seq_len = formatted_tokens["input_ids"].shape
    device = formatted_tokens["input_ids"].device

    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)

    # Get user marker string
    user_marker, _ = get_chat_markers(tokenizer)

    # Tokenize marker once (fast & exact)
    user_marker_ids = tokenizer.encode(
        user_marker,
        add_special_tokens=False
    )

    if len(user_marker_ids) == 0:
        raise RuntimeError("User marker tokenization produced empty token list")

    for i in range(batch_size):
        ids = formatted_tokens["input_ids"][i].tolist()

        start_idx = None
        for j in range(len(ids) - len(user_marker_ids) + 1):
            if ids[j : j + len(user_marker_ids)] == user_marker_ids:
                start_idx = j
                break

        if start_idx is None:
            raise RuntimeError(
                f"Could not find user marker tokens in batch item {i}"
            )

        # 🔥 Mask EVERYTHING from user marker → end
        mask[i, start_idx:] = True

    return mask

def create_system_token_mask(
    system_prompts: Union[str, List[str]],
    formatted_tokens: dict,
    tokenizer: AutoTokenizer,
    batch_indices: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Create a mask indicating which tokens correspond to system prompt content.
    This excludes formatting tokens and only includes the actual system prompt text.
    
    Args:
        system_prompts: The system prompt string or list of system prompts
        formatted_tokens: Tokenized batch with chat template applied
        tokenizer: Tokenizer used for encoding
        batch_indices: Optional list of indices mapping batch positions to prompt list positions
        
    Returns:
        Boolean tensor of shape (batch_size, seq_len) where True indicates system prompt tokens
    """
    batch_size = formatted_tokens['input_ids'].shape[0]
    seq_len = formatted_tokens['input_ids'].shape[1]
    
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=formatted_tokens['input_ids'].device)
    
    # Get the correct markers for this tokenizer
    _, system_marker = get_chat_markers(tokenizer)
    
    # Process each item in batch
    for i in range(batch_size):
        # Get the system prompt for this batch item
        if isinstance(system_prompts, str):
            system_prompt = system_prompts
        else:
            # List of system prompts
            if batch_indices is not None:
                system_prompt = system_prompts[batch_indices[i]]
            else:
                system_prompt = system_prompts[i]
        
        # Decode full sequence to get text
        full_text = tokenizer.decode(formatted_tokens['input_ids'][i], skip_special_tokens=False)
        
        # Find system content boundaries in text
        marker_idx = full_text.find(system_marker)
        
        if marker_idx == -1:
            raise RuntimeError(
                f"Could not find system marker '{system_marker}' in formatted text for batch item {i}.\n"
                f"This may indicate an issue with the chat template formatting."
            )
        
        # Find where actual system prompt content starts
        # Search for system prompt text after the marker
        search_after = marker_idx + len(system_marker)
        # Use first 100 chars of system prompt for matching
        search_text = system_prompt[:min(100, len(system_prompt))].strip()
        
        # Find the system prompt in the text
        sys_start_text = full_text.find(search_text, search_after)
        
        if sys_start_text == -1:
            # Try with first 30 chars
            search_text = system_prompt[:min(30, len(system_prompt))].strip()
            sys_start_text = full_text.find(search_text, search_after)
        
        if sys_start_text != -1:
            # Found it - now map to tokens
            sys_end_text = sys_start_text + len(system_prompt)
            
            # Map text positions to token positions
            token_start = None
            token_end = None
            
            for tok_idx in range(seq_len):
                # Decode up to current token
                decoded_so_far = tokenizer.decode(
                    formatted_tokens['input_ids'][i][:tok_idx + 1],
                    skip_special_tokens=False
                )
                
                # Check if we've reached the start of the system marker
                if token_start is None and len(decoded_so_far) >= marker_idx:
                    token_start = max(0, tok_idx - 1)
                
                # Check if we've reached system prompt end
                if token_start is not None and len(decoded_so_far) >= sys_end_text:
                    token_end = tok_idx + 1
                    break
            
            if token_start is not None and token_end is not None:
                mask[i, token_start:token_end] = True
            else:
                # Fallback: mask some tokens after marker
                for tok_idx in range(seq_len):
                    decoded = tokenizer.decode(
                        formatted_tokens['input_ids'][i][:tok_idx],
                        skip_special_tokens=False
                    )
                    if len(decoded) >= marker_idx + len(system_marker):
                        # Mask next 50 tokens as rough estimate
                        mask[i, tok_idx:min(tok_idx + 50, seq_len)] = True
                        break
        else:
            # Couldn't find system prompt text - mask tokens after marker
            for tok_idx in range(seq_len):
                decoded = tokenizer.decode(
                    formatted_tokens['input_ids'][i][:tok_idx],
                    skip_special_tokens=False
                )
                if len(decoded) >= marker_idx + len(system_marker):
                    # Mask next 50 tokens as rough estimate
                    mask[i, tok_idx:min(tok_idx + 50, seq_len)] = True
                    break
    
    return mask

def apply_steering_to_layer(
    layer_envoy:Envoy,
    steering_vector: torch.Tensor,
    steering_mask: torch.Tensor
) -> None:
    """
    Apply steering vector only to user token positions.
    
    Args:
        layer_envoy: Layer envoy object with output attribute
        steering_vector: Steering vector of shape (d_model,)
        steering_mask: Boolean mask of shape (batch, seq_len) indicating user tokens
    """
    # Expand mask to match layer output dimensions
    # assert steering_mask.shape[0] == layer_envoy.output.shape[0], f"Batch size mismatch: {steering_mask.shape[0]} != {layer_envoy.output.shape[0]}"
    # assert steering_mask.shape[1] == layer_envoy.output.shape[1], f"Sequence length mismatch: {steering_mask.shape[1]} != {layer_envoy.output.shape[1]}"

    mask_expanded = steering_mask.unsqueeze(-1).expand_as(layer_envoy.output)  # (batch, seq_len, d_model)
    
    # Apply steering only where mask is True
    steering_expanded = steering_vector.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model)
    layer_envoy.output[:,:,:] = layer_envoy.output[:,:,:] + mask_expanded * steering_expanded

def get_model_info(lma: LanguageModel) -> Tuple[List, int, bool, Envoy]:
    """
    Get model layers, number of layers, and whether it's fine-tuned.
    
    Returns:
        Tuple of (layers, num_layers, is_finetuned)
    """
    is_ft = isinstance(lma._model, PeftModel)
    if is_ft:
        layers = lma.base_model.model.model.layers
        embed = lma.base_model.model.model.embed_tokens
        num_layers = len(layers)
    elif 'Gemma3' in lma._model.__class__.__name__:
        layers = lma.model.language_model.layers
        embed = lma.model.language_model.embed_tokens
        num_layers = len(layers)
    else:
        layers = lma.model.layers
        num_layers = len(layers)
        embed = lma.model.embed_tokens
    
    return layers, num_layers, is_ft, embed

def prepare_steering_vectors(
    steering_vectors: dict[torch.Tensor, float] | None,
    layer_to_steer: int | Literal['all'] | List[int],
    d_model: int,
    model_len: int
) -> Tuple[torch.Tensor, List[torch.Tensor] | None]:
    """
    Prepare steering vectors for application.
    
    Returns:
        Tuple of (total_steering, steering_vec_list)
    """
    #import pdb; pdb.set_trace()
    if steering_vectors:
        # Combine all steering vectors
        first_vector, first_multiplier = next(iter(steering_vectors.items()))
        total_steering = first_vector * first_multiplier
        
        for vector, multiplier in list(steering_vectors.items())[1:]:
            total_steering = total_steering + vector * multiplier
    else:
        total_steering = torch.zeros(d_model, device="cuda")
    
    # Prepare vector list for multi-layer steering
    steering_vec_list = None
    if layer_to_steer == 'all' or isinstance(layer_to_steer, list):
        assert total_steering.shape == (model_len, d_model), f"Expected shape ({model_len}, {d_model}), got {total_steering.shape}"
        steering_vec_list = torch.unbind(total_steering, dim=0)
    
    return total_steering, steering_vec_list


def load_steering_vectors(
    model_name: str,
    vec_type: str,
    layer_indices: List[int],
    d_model: int,
    model_len: int,
    multiplier: float | List[float] = 1.0,
    prompt_category: str = None,
    act_name: str = None,
    probe_dir: str = "probe_checkpoints",
    refusal_dir: str = "refusal_svs",
) -> dict[torch.Tensor, float]:
    
    full_steering = torch.zeros(model_len, d_model, device='cuda', dtype=torch.bfloat16)
    
    print(f"Loading vectors for {vec_type}...")
    
    # Handle scalar vs list multiplier
    if isinstance(multiplier, list):
        if len(multiplier) != len(layer_indices):
            raise ValueError(f"Length of multiplier list ({len(multiplier)}) must match number of layers ({len(layer_indices)})")
        multipliers = multiplier
    else:
        multipliers = [multiplier] * len(layer_indices)
    
    for i, layer_idx in enumerate(layer_indices):
        current_mult = multipliers[i]
        
        if vec_type == 'refusal':
            # Path: {refusal_dir}/{model}/sv_layer_{layer}.pt
            path = Path(f"{refusal_dir}/{model_name}/sv_layer_{layer_idx}.pt")
            if not path.exists():
                 raise FileNotFoundError(f"Refusal vector not found: {path}")
            
            vec = torch.load(path).float().cuda()
            
        elif vec_type in ['trust_complete', 'trust_orthogonal']:
            if not prompt_category or not act_name:
                raise ValueError("prompt_category and act_name required for trust vectors")
                
            # Path: {probe_dir}/controlling_probe/{prompt_category}/{MODEL}/{act_token.name}/trust_probe_at_layer_{layer_num}.pth
            probe_path = Path(f"{probe_dir}/controlling_probe/{prompt_category}/{model_name}/{act_name}/trust_probe_at_layer_{layer_idx}.pth")
            
            if not probe_path.exists():
                raise FileNotFoundError(f"Probe not found: {probe_path}")
                
            probe = torch.load(probe_path)
            
            diff_0 = probe['proj.0.weight'][0].float().cuda()
            diff_1 = probe['proj.0.weight'][1].float().cuda()
            trust_vector = diff_1 - diff_0
            
            if vec_type == 'trust':
                vec = trust_vector
            else:
                # Orthogonalize
                ref_path = Path(f"{refusal_dir}/{model_name}/sv_layer_{layer_idx}.pt")
                if not ref_path.exists():
                     raise FileNotFoundError(f"Refusal vector for orthogonalization not found: {ref_path}")
                
                refusal_vector = torch.load(ref_path).float().cuda()
                
                # Normalization
                refusal_hat = refusal_vector / refusal_vector.norm()
                
                # Projection
                dot_product = torch.dot(trust_vector, refusal_hat)
                refusal_component = dot_product * refusal_hat
                
                # Subtraction
                trust_ortho = trust_vector - refusal_component
                
                vec = trust_ortho

        else:
            raise ValueError(f"Unknown vector type: {vec_type}")
            
        # Cast to bfloat16, scale by specific multiplier, and assign
        full_steering[layer_idx] = (vec * current_mult).to(torch.bfloat16)

    # Return with multiplier 1.0 since we baked it in
    return {full_steering: 1.0}



