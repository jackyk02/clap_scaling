#!/usr/bin/env python3
"""
Generate 128 OpenVLA actions for each original instruction using temperature 1.0
to analyze action generation variability.
"""

import requests
import json
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

# Define the ranges for NRMSE calculation (from collect.py)
min_values = np.array([-0.02872725307941437,
          -0.04170349963009357,
          -0.026093858778476715,
          -0.08092105075716972,
          -0.09288699507713317,
          -0.20718276381492615,
          0.0])
max_values = np.array([0.028309678435325586,
          0.040855254605412394,
          0.040161586627364146,
          0.08192047759890528,
          0.07792850524187081,
          0.20382574498653397,
          1.0])
ranges = max_values - min_values

def calculate_nrmse(action0, action1):
    """
    Calculate normalized root mean squared error between two actions
    """
    # Normalize the difference by the range
    normalized_diff = (action0 - action1) / ranges
    nrmse = np.sqrt(np.mean(normalized_diff**2))
    return nrmse

def get_batch_actions(instructions, image_path, temperature=1.0):
    """
    Get batch actions for multiple instructions using OpenVLA.
    
    Args:
        instructions: List of instruction strings or a single instruction string
        image_path: Path to the image file
        temperature: Temperature for sampling
    
    Returns:
        Tuple of (output_ids, actions) as numpy arrays
    """
    image_path = os.path.abspath(image_path)
    
    # Handle both single instruction and list of instructions
    if isinstance(instructions, str):
        instructions = [instructions]
    
    payload = {
        "instructions": instructions,
        "image_path": image_path,
        "temperature": temperature
    }

    res = requests.post(
        "http://localhost:3200/batch",
        data=json.dumps(payload),
        headers={'Content-Type': 'application/json'}
    )
    res.raise_for_status()
    result = json.loads(res.text)
    return np.array(result["output_ids"]), np.array(result["actions"])

def load_bridge_data():
    """Load bridge samples"""
    # Load bridge samples
    with open('bridge_samples.json', 'r') as f:
        bridge_data = json.load(f)
    
    return bridge_data

def calculate_action_statistics(actions):
    """
    Calculate statistics for a set of actions
    
    Args:
        actions: List of action arrays
    
    Returns:
        Dictionary with mean, std, min, max, and pairwise NRMSE statistics
    """
    actions = np.array(actions)
    
    # Basic statistics
    mean_action = np.mean(actions, axis=0)
    std_action = np.std(actions, axis=0)
    min_action = np.min(actions, axis=0)
    max_action = np.max(actions, axis=0)
    
    # Calculate pairwise NRMSE between all action pairs
    n_actions = len(actions)
    pairwise_nrmse = []
    
    for i in range(n_actions):
        for j in range(i + 1, n_actions):
            nrmse = calculate_nrmse(actions[i], actions[j])
            pairwise_nrmse.append(nrmse)
    
    pairwise_nrmse = np.array(pairwise_nrmse)
    
    return {
        'mean_action': mean_action.tolist(),
        'std_action': std_action.tolist(),
        'min_action': min_action.tolist(),
        'max_action': max_action.tolist(),
        'pairwise_nrmse_mean': float(np.mean(pairwise_nrmse)),
        'pairwise_nrmse_std': float(np.std(pairwise_nrmse)),
        'pairwise_nrmse_min': float(np.min(pairwise_nrmse)),
        'pairwise_nrmse_max': float(np.max(pairwise_nrmse)),
        'n_comparisons': len(pairwise_nrmse)
    }

def main():
    print("Loading bridge dataset...")
    bridge_data = load_bridge_data()
    
    samples = bridge_data['samples']
    
    print(f"Found {len(samples)} bridge samples")
    
    # Results list to store all datapoints
    results = []
    
    # Number of actions to generate per instruction
    n_actions_per_instruction = 128
    
    total_expected = len(samples) * n_actions_per_instruction
    print(f"Expected total datapoints: {total_expected}")
    
    # Process each sample
    with tqdm(total=total_expected, desc="Generating variability actions") as pbar:
        for sample_idx, sample in enumerate(samples):
            sample_id = sample['sample_id']
            original_instruction = sample['original_instruction']
            ground_truth_action = np.array(sample['current_ground_truth_action'])
            
            # Get OpenVLA processed image path
            image_filename = sample['state']['agent_view_image_file']
            base_name = image_filename.split('_')[0]  # Extract number from "N_clip.jpg"
            openvla_image_path = f"processed_images/openvla/{base_name}_openvla.jpg"
            
            if not os.path.exists(openvla_image_path):
                print(f"Warning: OpenVLA image not found: {openvla_image_path}")
                # Skip this sample and update progress bar
                pbar.update(n_actions_per_instruction)
                continue
            
            # Create list of the same instruction repeated 128 times
            repeated_instructions = [original_instruction] * n_actions_per_instruction
            
            try:
                # Get actions for all repeated instructions at once
                output_ids, generated_actions = get_batch_actions(
                    instructions=repeated_instructions,
                    image_path=openvla_image_path,
                    temperature=1.0
                )
                
                # Calculate statistics for this set of actions
                action_stats = calculate_action_statistics(generated_actions)
                
                # Process each generated action
                sample_actions = []
                for action_idx, (generated_action, output_id) in enumerate(zip(generated_actions, output_ids)):
                    # Calculate NRMSE against ground truth
                    nrmse_gt = calculate_nrmse(ground_truth_action, generated_action)
                    
                    # Create result entry
                    result_entry = {
                        'sample_id': sample_id,
                        'sample_index': sample_idx,
                        'action_index': action_idx,
                        'instruction': original_instruction,
                        'ground_truth_action': ground_truth_action.tolist(),
                        'generated_action': generated_action.tolist(),
                        'output_ids': output_id.tolist(),
                        'nrmse_vs_ground_truth': float(nrmse_gt),
                        'image_path': openvla_image_path,
                        'episode_id': sample['episode_id'],
                        'timestep': sample['timestep']
                    }
                    
                    sample_actions.append(result_entry)
                    pbar.update(1)
                
                # Add sample-level statistics
                sample_result = {
                    'sample_id': sample_id,
                    'sample_index': sample_idx,
                    'instruction': original_instruction,
                    'ground_truth_action': ground_truth_action.tolist(),
                    'image_path': openvla_image_path,
                    'episode_id': sample['episode_id'],
                    'timestep': sample['timestep'],
                    'action_statistics': action_stats,
                    'actions': sample_actions
                }
                
                results.append(sample_result)
                    
            except Exception as e:
                print(f"Error processing sample {sample_id}: {e}")
                # Skip this sample and update progress bar
                pbar.update(n_actions_per_instruction)
                continue
    
    print(f"\nGenerated actions for {len(results)} samples")
    total_actions = sum(len(sample['actions']) for sample in results)
    print(f"Total individual actions: {total_actions}")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"json/bridge_openvla_actions_variability_{timestamp}.json"
    
    # Calculate overall statistics
    all_nrmse_vs_gt = []
    all_pairwise_nrmse_means = []
    
    for sample_result in results:
        # Collect NRMSE vs ground truth
        sample_nrmse = [action['nrmse_vs_ground_truth'] for action in sample_result['actions']]
        all_nrmse_vs_gt.extend(sample_nrmse)
        
        # Collect pairwise NRMSE means
        all_pairwise_nrmse_means.append(sample_result['action_statistics']['pairwise_nrmse_mean'])
    
    # Prepare final data structure
    final_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_samples': len(results),
            'actions_per_sample': n_actions_per_instruction,
            'total_actions': total_actions,
            'source_bridge_file': 'bridge_samples.json',
            'model': 'OpenVLA',
            'temperature': 1.0,
            'api_endpoint': 'http://localhost:3200',
            'nrmse_ranges': {
                'min_values': min_values.tolist(),
                'max_values': max_values.tolist(),
                'ranges': ranges.tolist()
            },
            'overall_statistics': {
                'nrmse_vs_ground_truth': {
                    'mean': float(np.mean(all_nrmse_vs_gt)),
                    'std': float(np.std(all_nrmse_vs_gt)),
                    'min': float(np.min(all_nrmse_vs_gt)),
                    'max': float(np.max(all_nrmse_vs_gt))
                },
                'pairwise_nrmse_means': {
                    'mean': float(np.mean(all_pairwise_nrmse_means)),
                    'std': float(np.std(all_pairwise_nrmse_means)),
                    'min': float(np.min(all_pairwise_nrmse_means)),
                    'max': float(np.max(all_pairwise_nrmse_means))
                }
            }
        },
        'results': results
    }
    
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Results saved successfully!")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"Total samples processed: {len(results)}")
    print(f"Actions per sample: {n_actions_per_instruction}")
    print(f"Total actions generated: {total_actions}")
    print(f"NRMSE vs Ground Truth - Mean: {np.mean(all_nrmse_vs_gt):.4f}, Std: {np.std(all_nrmse_vs_gt):.4f}")
    print(f"Action Variability (Pairwise NRMSE) - Mean: {np.mean(all_pairwise_nrmse_means):.4f}, Std: {np.std(all_pairwise_nrmse_means):.4f}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
