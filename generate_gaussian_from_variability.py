#!/usr/bin/env python3
"""
Generate 128 Gaussian-perturbed actions for the first 5 samples from 
bridge_openvla_actions_variability_20250918_011122.json
"""

import json
import numpy as np
import os
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

def generate_gaussian_actions(original_actions, num_samples=128):
    """
    Generate Gaussian-perturbed actions based on the mean and variance of original actions.
    
    Args:
        original_actions: List of action arrays from the variability data
        num_samples: Number of Gaussian samples to generate
        
    Returns:
        NumPy array of shape (num_samples, 7) containing Gaussian-perturbed actions
    """
    # Convert to numpy array
    actions_array = np.array([action['generated_action'] for action in original_actions])
    
    # Calculate mean and variance for each dimension
    mean_values = np.mean(actions_array, axis=0)
    var_values = np.var(actions_array, axis=0)
    
    # Initialize output array to hold Gaussian samples
    gaussian_actions = np.zeros((num_samples, 7))
    
    # Generate num_samples Gaussian-perturbed samples
    for i in range(num_samples):
        # Generate values using the calculated mean and variance
        # For dimensions 0-5 (continuous values)
        gaussian_action = np.random.normal(mean_values, np.sqrt(var_values), size=7)
        
        # For the 7th dimension (binary), use probability based on mean
        p_gripper = mean_values[-1]  # Probability of gripper being 1
        gaussian_action[-1] = np.random.choice([0.0, 1.0], p=[1-p_gripper, p_gripper])
        
        # Clamp values to valid range for first six dimensions
        gaussian_action[:-1] = np.clip(gaussian_action[:-1], min_values[:-1], max_values[:-1])
        
        # Store the Gaussian action
        gaussian_actions[i] = gaussian_action
    
    return gaussian_actions

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
    # Load the variability data
    input_file = "json/bridge_openvla_actions_variability_20250918_022645.json"
    print(f"Loading variability data from {input_file}...")
    
    with open(input_file, 'r') as f:
        variability_data = json.load(f)
    
    # Get the first 5 samples
    first_3_samples = variability_data['results'][:5]
    print(f"Processing first 5 samples from {len(variability_data['results'])} total samples")
    
    # Results list to store all datapoints
    results = []
    
    # Number of Gaussian actions to generate per sample
    n_gaussian_per_sample = 128
    
    for sample_idx, sample in enumerate(first_3_samples):
        sample_id = sample['sample_id']
        instruction = sample['instruction']
        ground_truth_action = np.array(sample['ground_truth_action'])
        image_path = sample['image_path']
        episode_id = sample['episode_id']
        timestep = sample['timestep']
        
        print(f"Processing sample {sample_id}: '{instruction}'")
        
        # Get the original 128 generated actions from the variability data
        original_actions = sample['actions']
        print(f"  Found {len(original_actions)} original actions")
        
        # Generate 128 Gaussian-perturbed actions
        gaussian_actions = generate_gaussian_actions(original_actions, n_gaussian_per_sample)
        
        # Calculate statistics for the Gaussian actions
        gaussian_stats = calculate_action_statistics(gaussian_actions)
        
        # Process each Gaussian action
        sample_gaussian_actions = []
        for action_idx, gaussian_action in enumerate(gaussian_actions):
            # Calculate NRMSE against ground truth
            nrmse_gt = calculate_nrmse(ground_truth_action, gaussian_action)
            
            # Create result entry
            result_entry = {
                'sample_id': sample_id,
                'sample_index': sample_idx,
                'action_index': action_idx,
                'instruction': instruction,
                'ground_truth_action': ground_truth_action.tolist(),
                'gaussian_action': gaussian_action.tolist(),
                'nrmse_vs_ground_truth': float(nrmse_gt),
                'image_path': image_path,
                'episode_id': episode_id,
                'timestep': timestep
            }
            
            sample_gaussian_actions.append(result_entry)
        
        # Add sample-level result
        sample_result = {
            'sample_id': sample_id,
            'sample_index': sample_idx,
            'instruction': instruction,
            'ground_truth_action': ground_truth_action.tolist(),
            'image_path': image_path,
            'episode_id': episode_id,
            'timestep': timestep,
            'original_variability_stats': sample['action_statistics'],
            'gaussian_action_statistics': gaussian_stats,
            'gaussian_actions': sample_gaussian_actions
        }
        
        results.append(sample_result)
    
    print(f"\nGenerated Gaussian actions for {len(results)} samples")
    total_gaussian_actions = sum(len(sample['gaussian_actions']) for sample in results)
    print(f"Total Gaussian actions: {total_gaussian_actions}")
    
    # Calculate overall statistics
    all_nrmse_vs_gt = []
    all_pairwise_nrmse_means = []
    
    for sample_result in results:
        # Collect NRMSE vs ground truth
        sample_nrmse = [action['nrmse_vs_ground_truth'] for action in sample_result['gaussian_actions']]
        all_nrmse_vs_gt.extend(sample_nrmse)
        
        # Collect pairwise NRMSE means
        all_pairwise_nrmse_means.append(sample_result['gaussian_action_statistics']['pairwise_nrmse_mean'])
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"json/bridge_openvla_gaussian_actions_{timestamp}.json"
    
    # Prepare final data structure
    final_data = {
        'metadata': {
            'timestamp': timestamp,
            'total_samples': len(results),
            'gaussian_actions_per_sample': n_gaussian_per_sample,
            'total_gaussian_actions': total_gaussian_actions,
            'source_variability_file': input_file,
            'method': 'Gaussian perturbation based on variability statistics',
            'samples_used': 'First 5 samples from variability data',
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
    print(f"Gaussian actions per sample: {n_gaussian_per_sample}")
    print(f"Total Gaussian actions generated: {total_gaussian_actions}")
    print(f"NRMSE vs Ground Truth - Mean: {np.mean(all_nrmse_vs_gt):.4f}, Std: {np.std(all_nrmse_vs_gt):.4f}")
    print(f"Action Variability (Pairwise NRMSE) - Mean: {np.mean(all_pairwise_nrmse_means):.4f}, Std: {np.std(all_pairwise_nrmse_means):.4f}")
    print(f"Results saved to: {output_file}")
    
    # Print sample details
    print(f"\nProcessed samples:")
    for i, sample in enumerate(results):
        print(f"  Sample {sample['sample_id']}: '{sample['instruction'][:50]}...'")

if __name__ == "__main__":
    main()
