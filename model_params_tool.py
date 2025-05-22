#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import os # Added for path operations
import argparse # Added for command-line interface


# Added functions for model state_dict prefix handling
# The class remove_prefix_parameters(path): was removed and replaced by these utility functions.

def remove_module_prefix(state_dict):
    """
    Remove 'module.' prefix from state_dict keys (added by DistributedDataParallel).
    
    Args:
        state_dict (dict): Model state dictionary possibly with 'module.' prefix
        
    Returns:
        dict: State dictionary with 'module.' prefix removed
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix (7 characters)
        else:
            new_state_dict[k] = v
    return new_state_dict

def clean_state_dict(state_dict, target_prefix=None, remove_prefix=None):
    """
    Clean a state dict by removing unwanted prefixes and adding target prefix.
    
    Args:
        state_dict (dict): Original state dictionary
        target_prefix (str, optional): Prefix to add to all keys if not already present
                                      (e.g., 'backbone.', 'diffusion_network.')
        remove_prefix (str, optional): Prefix to remove from all keys if present
        
    Returns:
        dict: Cleaned state dictionary
    """
    # First remove module prefix if exists
    clean_dict = remove_module_prefix(state_dict)
    
    # Remove specified prefix if needed
    if remove_prefix and remove_prefix != '':
        temp_dict = {}
        prefix_len = len(remove_prefix)
        for k, v in clean_dict.items():
            if k.startswith(remove_prefix):
                temp_dict[k[prefix_len:]] = v
            else:
                temp_dict[k] = v
        clean_dict = temp_dict
    
    # Add target prefix if needed
    if target_prefix and target_prefix != '':
        prefixed_dict = {}
        for k, v in clean_dict.items():
            if not k.startswith(target_prefix):
                prefixed_dict[f"{target_prefix}{k}"] = v
            else:
                prefixed_dict[k] = v
        return prefixed_dict
    
    return clean_dict

def extract_state_dict(checkpoint):
    """
    Extract state_dict from checkpoint regardless of its structure.
    
    Args:
        checkpoint: Model checkpoint that could be a state_dict or contain a state_dict
        
    Returns:
        dict: Extracted state dictionary
    """
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        # Handle Lightning-style checkpoints with epoch, etc.
        if any(k in checkpoint for k in ['epoch', 'global_step', 'pytorch-lightning_version']):
            for k, v in checkpoint.items():
                if isinstance(v, dict) and any(key.endswith('.weight') for key in v.keys()): # Check if v is a state_dict
                    return v
        # Check for backbone or diffusion_network nested dicts if they are the state_dicts themselves
        if 'backbone' in checkpoint or 'diffusion_network' in checkpoint:
             # This case assumes checkpoint IS the state_dict, or contains state_dicts at top level
             # For example, if checkpoint = {'backbone': sd1, 'diffusion_network': sd2}
             # This function is intended to return THE state_dict, not a dict of state_dicts.
             # If the intention is to handle nested state_dicts, this logic might need adjustment
             # or be handled by the caller.
             # For now, if it's a dict and not 'state_dict' and not clearly lightning, assume it's the state_dict.
             is_likely_state_dict = True
             for val in checkpoint.values():
                 if not isinstance(val, torch.Tensor) and not isinstance(val, dict): # Simple check
                     is_likely_state_dict = False
                     break
             if is_likely_state_dict and not any(key.endswith('.weight') for key in checkpoint.keys()): # if top level keys are not params
                 # This could be a dict containing multiple state_dicts, e.g. {'backbone': {}, 'diffusion': {}}
                 # The original code returned `checkpoint` here.
                 # This might be ambiguous if `checkpoint` contains other metadata alongside state_dict parts.
                 # For robust extraction, a more specific check or structure assumption is needed.
                 # Assuming the original intent was to return the checkpoint if it's a collection of state_dicts.
                 pass # Fall through to return checkpoint if it's not 'state_dict' or lightning like
    
    # Assume the checkpoint itself is the state_dict if not handled above
    # or if it's a dict that doesn't fit the specific structures checked.
    if isinstance(checkpoint, dict) and not ('state_dict' in checkpoint or \
        any(k in checkpoint for k in ['epoch', 'global_step', 'pytorch-lightning_version'])):
        # If it's a dictionary but not fitting known wrapper formats, it might be the state_dict itself,
        # or a collection of them. The original code returned `checkpoint`.
        return checkpoint

    # If checkpoint is not a dict, it's assumed to be the state_dict directly (e.g. from torch.load(model.state_dict()))
    if not isinstance(checkpoint, dict):
        return checkpoint
    
    # Default fallback if it's a dict but no state_dict found by specific keys
    # This could happen if 'state_dict' is not the key, but it's still a wrapped checkpoint.
    # The original code would return `checkpoint` in such cases.
    return checkpoint


def remove_prefix_from_model(model_path, output_path=None, prefix_to_remove=None):
    """
    Remove the prefix from the model state dict and save it in a standard format.
    
    Args:
        model_path (str): Path to input model file
        output_path (str, optional): Path to save refined model. If None, overwrites original.
        prefix_to_remove (str, optional): Specific prefix to remove (beyond 'module.')
        
    Returns:
        dict: Refined state dictionary
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    original_checkpoint_structure = None
    if isinstance(checkpoint, dict):
        original_checkpoint_structure = checkpoint.copy() # Shallow copy to preserve structure keys
    
    # Extract the state_dict from the checkpoint
    state_dict = extract_state_dict(checkpoint)
    
    if not isinstance(state_dict, dict):
        raise ValueError(f"Could not extract a valid state_dict from {model_path}")

    # Clean the state_dict - only remove prefixes, don't add any
    clean_dict = clean_state_dict(state_dict, target_prefix=None, remove_prefix=prefix_to_remove)
    
    # Prepare output based on original checkpoint structure
    output_data_to_save = clean_dict # Default to saving just the cleaned state_dict

    if original_checkpoint_structure is not None:
        if 'state_dict' in original_checkpoint_structure:
            original_checkpoint_structure['state_dict'] = clean_dict
            output_data_to_save = original_checkpoint_structure
        # If the original checkpoint was the state_dict itself, but wrapped in a dict for other meta
        # (e.g. lightning like but state_dict was extracted from a nested key)
        # This part ensures we put the cleaned_dict back correctly if possible.
        # The current extract_state_dict might return the whole checkpoint if it's complex.
        # For simplicity, if 'state_dict' key was not present at top level of original checkpoint,
        # we assume the cleaned_dict is what should be saved, or it was already handled by extract_state_dict
        # returning a modified structure.
        # A more robust way would be to track how state_dict was extracted and reconstruct.
        # Given the current extract_state_dict, if it returned the whole checkpoint because it couldn't find
        # a 'state_dict' key but found other keys like 'epoch', it implies the structure should be preserved.
        # However, clean_state_dict operates on the *extracted* state_dict.
        # Let's assume if original_checkpoint_structure is not None and 'state_dict' is not in it,
        # but state_dict was successfully extracted, the original structure might have been just the state_dict.
        # This logic can be tricky. The safest is: if 'state_dict' was a key, put it back. Otherwise, save the cleaned_dict.
        # Or, if the extracted state_dict was the checkpoint itself (by reference), then modifying it would modify checkpoint.
        # Let's refine:
        # If the checkpoint itself was the state_dict (e.g. torch.save(model.state_dict(), path))
        # then `state_dict` variable holds it, and `original_checkpoint_structure` is this dict.
        # `clean_dict` is the modified version. So `output_data_to_save` should be `clean_dict`.
        # If checkpoint = {'state_dict': sd, 'epoch': e}, then `state_dict` is `sd`.
        # `original_checkpoint_structure` is `{'state_dict': sd, 'epoch': e}`.
        # We update `original_checkpoint_structure['state_dict'] = clean_dict`.
        # So `output_data_to_save` becomes the modified `original_checkpoint_structure`.
        # This seems correct.

    # Save if output path provided
    if output_path:
        torch.save(output_data_to_save, output_path)
        print(f"Refined model saved to: {output_path}")
    elif output_path is None and model_path: # Overwrite original
        torch.save(output_data_to_save, model_path)
        print(f"Original model overwritten with refined version: {model_path}")
    
    return clean_dict

def check_prefix_integrity(state_dict, expected_prefix=None):
    """
    Analyze state dict keys and check prefix integrity.
    
    Args:
        state_dict (dict): Model state dictionary to check
        expected_prefix (str, optional): Expected prefix for keys
        
    Returns:
        dict: Dictionary with analysis results including:
              - has_module_prefix: Whether 'module.' prefix is present
              - prefix_consistency: Whether all keys have consistent prefixes
              - common_prefix: Most common prefix found in keys
              - mismatched_keys: List of keys with unexpected prefixes
    """
    if not state_dict or not isinstance(state_dict, dict):
        return {'error': 'Empty or invalid state dictionary'}
    
    keys = list(state_dict.keys())
    
    # Check for module prefix
    module_prefixed = [k for k in keys if k.startswith('module.')]
    has_module_prefix = len(module_prefixed) > 0
    
    # Find common prefixes
    prefixes = {}
    for k in keys:
        # Extract prefix up to first '.'
        parts = k.split('.')
        if len(parts) > 1:
            current_prefix = parts[0] + '.'
            prefixes[current_prefix] = prefixes.get(current_prefix, 0) + 1
    
    # Determine the most common prefix
    common_prefix_val = None
    max_count = 0
    if prefixes:
        for p, count in prefixes.items():
            if count > max_count:
                max_count = count
                common_prefix_val = p
    
    # Check prefix consistency
    prefix_consistency = True
    mismatched_keys = []
    
    # Determine reference prefix for consistency check
    # If expected_prefix is given, use it.
    # Else, if 'module.' prefix is dominant and no other common prefix, don't assume a non-module common prefix.
    # Else, use the most common prefix found (if any).
    
    reference_prefix_for_consistency = expected_prefix
    if not reference_prefix_for_consistency and common_prefix_val:
        # If 'module.' is the common prefix, we are interested in consistency *after* 'module.'
        # or consistency of other prefixes.
        # This logic is about general consistency.
        reference_prefix_for_consistency = common_prefix_val

    if reference_prefix_for_consistency:
        for k in keys:
            # If checking against 'module.', then keys not starting with it are not "mismatched" in this context,
            # unless expected_prefix was explicitly 'module.'.
            # The goal is to see if *other* prefixes are consistent, or if expected_prefix is met.
            temp_key = k
            if has_module_prefix and k.startswith('module.'):
                temp_key = k[7:] # Check consistency of the part after 'module.' if 'module.' is common/expected

            if expected_prefix: # Strict check against expected_prefix
                if not k.startswith(expected_prefix):
                    prefix_consistency = False
                    mismatched_keys.append(k)
            elif common_prefix_val : # Check against the most common prefix found (if not 'module.')
                                     # Or if 'module.' is common, check keys that don't start with it against other common prefixes.
                # This part can be tricky: what defines a mismatch if not expected_prefix?
                # Let's say a key is mismatched if it has a prefix, but not the common_prefix_val,
                # and it's not 'module.' if 'module.' is not the common_prefix_val.
                key_has_own_prefix = '.' in temp_key and temp_key.split('.')[0]+'.' != common_prefix_val
                if temp_key.split('.')[0]+'.' != '' and not temp_key.startswith(common_prefix_val):
                     # This key has a prefix, but it's not the common one.
                     # Avoid flagging keys without any prefix as mismatched unless a prefix is expected.
                     if '.' in temp_key: # Only consider keys that appear to have a prefix
                        prefix_consistency = False
                        mismatched_keys.append(k)

    # Simplified consistency: if an expected prefix is given, all keys must have it.
    # If not, are all existing prefixes (ignoring module.) the same?
    # The original code's consistency check:
    # if expected_prefix: check against it.
    # elif common_prefix: check against common_prefix for keys that have a '.' and aren't 'module.'
    # This seems reasonable. Let's stick to that.
    
    prefix_consistency = True # Reset for clearer logic path
    mismatched_keys = []
    
    if expected_prefix:
        for k_orig in keys:
            k_to_check = k_orig
            if k_orig.startswith('module.') and expected_prefix and not expected_prefix.startswith('module.'):
                # If we expect 'model.' but key is 'module.model.', strip 'module.' first
                if k_orig[7:].startswith(expected_prefix):
                    continue 
            if not k_to_check.startswith(expected_prefix):
                prefix_consistency = False
                mismatched_keys.append(k_orig)
    elif common_prefix_val and common_prefix_val != 'module.': # Only if a non-module common prefix exists
        for k in keys:
            if k.startswith('module.'): # Skip module prefix itself for this check, or check part after
                effective_key = k[7:]
                if not effective_key: continue # Should not happen with valid keys
            else:
                effective_key = k
            
            if '.' in effective_key: # Key has some sub-structure
                key_prefix = effective_key.split('.')[0] + '.'
                if key_prefix != common_prefix_val:
                    prefix_consistency = False
                    mismatched_keys.append(k) # Report original key
            # else: key has no prefix (e.g. "weight"), not considered a mismatch against common_prefix_val unless expected_prefix was set
    elif not common_prefix_val and not expected_prefix and any('.' in k for k in keys if not k.startswith('module.')):
        # Multiple different prefixes, or some have prefixes and some don't (excluding module.)
        # This indicates inconsistency if there's no single common_prefix.
        # Example: {'encoder.ln1.w', 'decoder.ln1.w', 'bias'}
        # Here, common_prefix_val might be None or one of them if counts are low.
        # If prefixes dictionary has more than one entry (excluding 'module.'), it's inconsistent.
        non_module_prefixes = {p for p in prefixes if p != 'module.'}
        if len(non_module_prefixes) > 1:
            prefix_consistency = False
            # Mismatched keys here could be all keys that don't conform to a majority, or just a general flag.
            # The original code's mismatched_keys logic for this case is based on common_prefix.
            # If common_prefix is None, this path isn't hit for mismatched_keys.
            # Let's refine: if no expected_prefix and no single common_prefix (other than module.), it's inconsistent.
            # Mismatched keys could be those not matching the *most* common, or just list all.
            # The original code's mismatched_keys are populated if common_prefix is found and some keys don't match it.
            # If common_prefix is None (e.g. {'a.x':1, 'b.y':1}), then mismatched_keys remains empty by that logic.
            # This seems fine. The flag `prefix_consistency` is the main output for this scenario.
            pass


    return {
        'has_module_prefix': has_module_prefix,
        'prefix_consistency': prefix_consistency, # True if all keys conform to expected_prefix or a single common_prefix (ignoring module.)
        'common_prefix': common_prefix_val, # Most frequent prefix (can be 'module.')
        'expected_prefix': expected_prefix,
        'mismatched_keys': mismatched_keys[:10] if len(mismatched_keys) > 10 else mismatched_keys,
        'mismatched_count': len(mismatched_keys),
        'total_keys': len(keys),
        'sample_keys': keys[:10] # Added sample keys
    }

def analyze_model_prefixes(model_path, expected_prefix=None):
    """
    Analyze a model file and check its prefix structure.
    
    Args:
        model_path (str): Path to the model file
        expected_prefix (str, optional): Expected prefix for model keys
        
    Returns:
        dict: Analysis results
    """
    if not os.path.exists(model_path):
        return {'error': f"Model file not found: {model_path}"}
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract the state_dict
    state_dict = extract_state_dict(checkpoint)
    
    if not isinstance(state_dict, dict):
         return {'error': f"Could not extract a valid state_dict from {model_path}", 'model_path': model_path}

    # Check prefix integrity
    results = check_prefix_integrity(state_dict, expected_prefix)
    results['model_path'] = model_path
    # Add sample keys from the state_dict to the results, regardless of mismatch
    if isinstance(state_dict, dict) and state_dict:
        results['actual_sample_keys'] = list(state_dict.keys())[:10]
    else:
        results['actual_sample_keys'] = []
    
    return results

#######################################
# Command-line functionality
#######################################

def main():
    """
    Command-line entry point for model utilities.

    Usage examples:
    
    1. Remove prefixes from a model:
       python model_params_tool.py remove_prefix --model_path path/to/your/model.ckpt --output_path path/to/refined/model.ckpt --target_prefix "unwanted_prefix."
       # To overwrite the original file:
       python model_params_tool.py remove_prefix --model_path path/to/your/model.ckpt --target_prefix "unwanted_prefix."

    2. Analyze prefixes of a model:
       python model_params_tool.py analyze_prefixes --model_path path/to/your/model.ckpt
       # With an expected prefix:
       python model_params_tool.py analyze_prefixes --model_path path/to/your/model.ckpt --expected_prefix "expected_prefix."
    """
    parser     = argparse.ArgumentParser(description='Pytorch model parameters utility functions for prefix handling.')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute', required=True)
    
    # Refine model command (remove_prefix_from_model)
    refine_parser = subparsers.add_parser('remove_prefix', help='Remove prefixes from the model state dict and save it.')
    refine_parser.add_argument('--model_path', type=str, required=True,
                              help='Path to the model file to refine.')
    refine_parser.add_argument('--output_path', type=str, default=None,
                              help='Path to save the refined model. If not provided, overwrites the original.')
    refine_parser.add_argument('--target_prefix', type=str, default=None,
                              help='Specific prefix to remove (e.g., "backbone."). This is in addition to "module." which is always checked.')
    
    # Analyze model prefixes command (analyze_model_prefixes)
    analyze_parser = subparsers.add_parser('analyze_prefixes', help='Analyze the prefix structure of a model.')
    analyze_parser.add_argument('--model_path', type=str, required=True,
                               help='Path to the model file to analyze.')
    analyze_parser.add_argument('--expected_prefix', type=str, default=None,
                               help='Expected prefix for keys (e.g., "backbone.", "diffusion_network.").')
    
    args = parser.parse_args()
    
    if args.command == 'remove_prefix':
        print(f"Attempting to remove prefix '{args.target_prefix}' from model: {args.model_path}")
        if args.output_path:
            print(f"Output will be saved to: {args.output_path}")
        else:
            print(f"Original model will be overwritten: {args.model_path}")
        
        refined_dict = remove_prefix_from_model(args.model_path, args.output_path, args.target_prefix)
        if refined_dict:
            print("Prefix removal process completed.")
        else:
            print("Prefix removal failed or state_dict was empty.")
            
    elif args.command == 'analyze_prefixes':
        print(f"Analyzing prefixes for model: {args.model_path}")
        if args.expected_prefix:
            print(f"Expecting prefix: {args.expected_prefix}")
            
        results = analyze_model_prefixes(args.model_path, args.expected_prefix)
        
        if results.get('error'):
            print(f"Error: {results['error']}")
        else:
            print(f"\n--- Prefix Analysis Results ---")
            print(f"Model Path: {results.get('model_path')}")
            print(f"Total Keys: {results.get('total_keys')}")
            print(f"Has 'module.' Prefix: {results.get('has_module_prefix')}")
            print(f"Most Common Prefix Found: {results.get('common_prefix') or 'N/A'}")
            print(f"Expected Prefix for Check: {results.get('expected_prefix') or 'Not specified'}")
            print(f"Prefix Consistency: {results.get('prefix_consistency')}")

            actual_sample_keys = results.get('actual_sample_keys', [])
            if actual_sample_keys:
                print(f"Sample Keys (first {len(actual_sample_keys)}):")
                for key in actual_sample_keys:
                    print(f"  - {key}")
            
            mismatched_count = results.get('mismatched_count', 0)
            if not results.get('prefix_consistency') or mismatched_count > 0 :
                print(f"Mismatched Keys ({mismatched_count} total):")
                for key in results.get('mismatched_keys', []):
                    print(f"  - {key}")
                if mismatched_count > len(results.get('mismatched_keys', [])):
                    print(f"  ... and {mismatched_count - len(results.get('mismatched_keys', []))} more.")
            print("--- End of Analysis ---")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()