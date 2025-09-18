#!/usr/bin/env python3
"""
Simplified comparison plot showing how action error decreases with 1, 10, and 100 samples:
1. Instruction Rephrasing (Oracle Verifier)
2. Repeated Sampling (Temperature-based Variability)
3. Gaussian Perturbation

X-axis: Number of samples/instructions in proposal distribution
Y-axis: Action Error (Average NRMSE)

Each approach uses different strategies to generate multiple candidates and select the best one.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
import glob

# Set up the matplotlib parameters with larger fonts for a publication-quality figure
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.figsize': (12, 8),
    'text.usetex': False,
})

def load_instruction_rephrasing_data():
    """Load instruction rephrasing data from multiple files (different seeds)"""
    print("Loading instruction rephrasing data...")
    
    # Find all instruction rephrasing files
    rephrasing_files = glob.glob('json/bridge_openvla_actions_*.json')
    rephrasing_files = [f for f in rephrasing_files if 'variability' not in f and 'gaussian' not in f]
    print(f"Found {len(rephrasing_files)} instruction rephrasing files: {rephrasing_files}")
    
    all_samples = []
    all_original_nrmse = []
    
    for file_path in rephrasing_files:
        print(f"  Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Organize data by sample_id for this file
        samples = defaultdict(list)
        original_nrmse = []
        
        for result in data['results']:
            sample_id = result['sample_id']
            samples[sample_id].append(result)
            
            # Collect original instruction NRMSE
            if result['is_original']:
                original_nrmse.append(result['nrmse'])
        
        all_samples.append(samples)
        all_original_nrmse.extend(original_nrmse)
        print(f"    Found {len(samples)} unique samples, avg original NRMSE: {np.mean(original_nrmse):.4f}")
    
    print(f"Total original instruction average NRMSE: {np.mean(all_original_nrmse):.4f}")
    
    return all_samples, np.mean(all_original_nrmse)

def load_variability_data():
    """Load repeated sampling variability data from multiple files (different seeds)"""
    print("Loading repeated sampling variability data...")
    
    # Find all variability files
    variability_files = glob.glob('json/bridge_openvla_actions_variability_*.json')
    print(f"Found {len(variability_files)} variability files: {variability_files}")
    
    all_samples = []
    
    for file_path in variability_files:
        print(f"  Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_samples.append(data['results'])
        print(f"    Found {len(data['results'])} variability samples")
    
    return all_samples

def load_gaussian_data():
    """Load Gaussian perturbation data from multiple files (different seeds)"""
    print("Loading Gaussian perturbation data...")
    
    # Find all Gaussian files
    gaussian_files = glob.glob('json/bridge_openvla_gaussian_actions_*.json')
    print(f"Found {len(gaussian_files)} Gaussian files: {gaussian_files}")
    
    all_samples = []
    
    for file_path in gaussian_files:
        print(f"  Loading {file_path}...")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_samples.append(data['results'])
        print(f"    Found {len(data['results'])} Gaussian samples")
    
    return all_samples

def calculate_oracle_nrmse_rephrasing(all_samples, num_instructions):
    """
    Oracle verifier for instruction rephrasing: select the best instruction from proposal distribution
    Returns mean and std across multiple seeds
    """
    seed_results = []
    
    for samples in all_samples:
        sample_nrmse_values = []
        
        for sample_id, results in samples.items():
            # Separate original and rephrased instructions
            original = [r for r in results if r['is_original']]
            rephrased = [r for r in results if not r['is_original']]
            
            if not original:
                continue
                
            if num_instructions == 1:
                # Only consider original instruction
                candidate_pool = original
            else:
                # Consider original + (num_instructions - 1) rephrased instructions
                available_rephrased = rephrased[:num_instructions - 1] if len(rephrased) >= num_instructions - 1 else rephrased
                candidate_pool = original + available_rephrased
            
            # Oracle selects the SINGLE BEST instruction from the candidate pool
            if candidate_pool:
                best_instruction = min(candidate_pool, key=lambda x: x['nrmse'])
                sample_nrmse_values.append(best_instruction['nrmse'])
        
        if sample_nrmse_values:
            seed_results.append(np.mean(sample_nrmse_values))
    
    if seed_results:
        return np.mean(seed_results), np.std(seed_results)
    else:
        return None, None

def calculate_oracle_nrmse_variability(all_variability_samples, num_samples):
    """
    Oracle verifier for repeated sampling: select the best action from multiple temperature samples
    Returns mean and std across multiple seeds
    """
    seed_results = []
    
    for variability_samples in all_variability_samples:
        sample_nrmse_values = []
        
        for sample in variability_samples:
            actions = sample['actions']
            
            if len(actions) < num_samples:
                # If we don't have enough samples, use all available
                candidate_pool = actions
            else:
                # Use the first num_samples actions
                candidate_pool = actions[:num_samples]
            
            # Oracle selects the SINGLE BEST action from the candidate pool
            if candidate_pool:
                best_action = min(candidate_pool, key=lambda x: x['nrmse_vs_ground_truth'])
                sample_nrmse_values.append(best_action['nrmse_vs_ground_truth'])
        
        if sample_nrmse_values:
            seed_results.append(np.mean(sample_nrmse_values))
    
    if seed_results:
        return np.mean(seed_results), np.std(seed_results)
    else:
        return None, None

def calculate_oracle_nrmse_gaussian(all_gaussian_samples, num_samples):
    """
    Oracle verifier for Gaussian perturbation: select the best action from Gaussian samples
    Returns mean and std across multiple seeds
    """
    seed_results = []
    
    for gaussian_samples in all_gaussian_samples:
        sample_nrmse_values = []
        
        for sample in gaussian_samples:
            actions = sample['gaussian_actions']
            
            if len(actions) < num_samples:
                # If we don't have enough samples, use all available
                candidate_pool = actions
            else:
                # Use the first num_samples actions
                candidate_pool = actions[:num_samples]
            
            # Oracle selects the SINGLE BEST action from the candidate pool
            if candidate_pool:
                best_action = min(candidate_pool, key=lambda x: x['nrmse_vs_ground_truth'])
                sample_nrmse_values.append(best_action['nrmse_vs_ground_truth'])
        
        if sample_nrmse_values:
            seed_results.append(np.mean(sample_nrmse_values))
    
    if seed_results:
        return np.mean(seed_results), np.std(seed_results)
    else:
        return None, None

def generate_simple_comparison_plot():
    """Generate simplified comparison plot for 1, 10, 100 samples"""
    
    # Load all data
    rephrasing_samples, original_avg_nrmse = load_instruction_rephrasing_data()
    variability_samples = load_variability_data()
    gaussian_samples = load_gaussian_data()
    
    # Define the specific sample sizes to test
    num_samples_range = [1, 10, 100]
    
    # Calculate NRMSE for each approach
    print("Calculating oracle NRMSE for different approaches and sizes...")
    
    # Instruction Rephrasing
    rephrasing_nrmse_means = []
    rephrasing_nrmse_stds = []
    for num_instructions in num_samples_range:
        oracle_mean, oracle_std = calculate_oracle_nrmse_rephrasing(rephrasing_samples, num_instructions)
        rephrasing_nrmse_means.append(oracle_mean)
        rephrasing_nrmse_stds.append(oracle_std if oracle_std else 0)
        improvement = ((original_avg_nrmse - oracle_mean) / original_avg_nrmse * 100) if oracle_mean else 0
        print(f"  Rephrasing {num_instructions} instructions: {oracle_mean:.4f}±{oracle_std:.4f} ({improvement:.1f}% improvement)")
    
    # Repeated Sampling (Variability)
    variability_nrmse_means = []
    variability_nrmse_stds = []
    for num_samples in num_samples_range:
        oracle_mean, oracle_std = calculate_oracle_nrmse_variability(variability_samples, num_samples)
        variability_nrmse_means.append(oracle_mean)
        variability_nrmse_stds.append(oracle_std if oracle_std else 0)
        improvement = ((original_avg_nrmse - oracle_mean) / original_avg_nrmse * 100) if oracle_mean else 0
        print(f"  Variability {num_samples} samples: {oracle_mean:.4f}±{oracle_std:.4f} ({improvement:.1f}% improvement)")
    
    # Gaussian Perturbation
    gaussian_nrmse_means = []
    gaussian_nrmse_stds = []
    for num_samples in num_samples_range:
        oracle_mean, oracle_std = calculate_oracle_nrmse_gaussian(gaussian_samples, num_samples)
        gaussian_nrmse_means.append(oracle_mean)
        gaussian_nrmse_stds.append(oracle_std if oracle_std else 0)
        improvement = ((original_avg_nrmse - oracle_mean) / original_avg_nrmse * 100) if oracle_mean else 0
        print(f"  Gaussian {num_samples} samples: {oracle_mean:.4f}±{oracle_std:.4f} ({improvement:.1f}% improvement)")
    
    # Create the comparison plot
    fig, ax = plt.subplots()
    
    # Colors and styling
    color_baseline = 'black'
    color_rephrasing = '#1f77b4'  # Blue
    color_variability = '#ff7f0e'  # Orange
    color_gaussian = '#2ca02c'  # Green
    
    marker_rephrasing = 'o'  # Circle
    marker_variability = 's'  # Square
    marker_gaussian = 'D'  # Diamond
    
    # Add baseline for original instruction
    ax.axhline(y=original_avg_nrmse, color=color_baseline, linestyle='-', linewidth=2.0, 
               label='Original Instruction Baseline')
    
    # Plot each approach with variance regions
    # Instruction Rephrasing
    ax.plot(num_samples_range, rephrasing_nrmse_means, label='Instruction Rephrasing (Oracle)', 
            color=color_rephrasing, marker=marker_rephrasing, markersize=10, linestyle='-', linewidth=3.0)
    ax.fill_between(num_samples_range, 
                    np.array(rephrasing_nrmse_means) - np.array(rephrasing_nrmse_stds),
                    np.array(rephrasing_nrmse_means) + np.array(rephrasing_nrmse_stds),
                    color=color_rephrasing, alpha=0.2)
    
    # Repeated Sampling (Variability)
    ax.plot(num_samples_range, variability_nrmse_means, label='Repeated Sampling (Oracle)', 
            color=color_variability, marker=marker_variability, markersize=10, linestyle='-', linewidth=3.0)
    ax.fill_between(num_samples_range, 
                    np.array(variability_nrmse_means) - np.array(variability_nrmse_stds),
                    np.array(variability_nrmse_means) + np.array(variability_nrmse_stds),
                    color=color_variability, alpha=0.2)
    
    # Gaussian Perturbation
    ax.plot(num_samples_range, gaussian_nrmse_means, label='Gaussian Perturbation (Oracle)', 
            color=color_gaussian, marker=marker_gaussian, markersize=10, linestyle='-', linewidth=3.0)
    ax.fill_between(num_samples_range, 
                    np.array(gaussian_nrmse_means) - np.array(gaussian_nrmse_stds),
                    np.array(gaussian_nrmse_means) + np.array(gaussian_nrmse_stds),
                    color=color_gaussian, alpha=0.2)
    
    # Axis labels and scale
    ax.set_xlabel("Proposal Distribution Size (Number of Candidates)")
    ax.set_ylabel("Action Error (Average NRMSE)")
    ax.set_xscale('log')
    ax.set_xticks(num_samples_range)
    ax.set_xticklabels([str(t) for t in num_samples_range])
    
    # Set y-axis limits to specific range
    ax.set_ylim(0.06, 0.17)
    
    # Grid and border
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Legend
    legend = ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=14)
    legend.get_frame().set_linewidth(1.5)
    
    # Title
    ax.set_title('Action Generation Approaches: 1, 10, 100 Candidates\nwith Oracle Selection from Proposal Distributions', 
                 fontsize=18, pad=20)
    
    # Save plot
    output_file = 'action_generation_simple_comparison.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    plt.show()
    
    # Calculate and print detailed statistics
    print(f"\n=== Simple Comparison Analysis (1, 10, 100 samples) ===")
    print(f"Original instruction baseline NRMSE: {original_avg_nrmse:.4f}")
    
    approaches = [
        ("Instruction Rephrasing", rephrasing_nrmse_means, rephrasing_nrmse_stds),
        ("Repeated Sampling", variability_nrmse_means, variability_nrmse_stds),
        ("Gaussian Perturbation", gaussian_nrmse_means, gaussian_nrmse_stds)
    ]
    
    # Show detailed breakdown for each approach
    print(f"\n=== Results by Sample Size ===")
    for i, num_candidates in enumerate(num_samples_range):
        print(f"\n{num_candidates} candidates:")
        for approach_name, nrmse_means, nrmse_stds in approaches:
            if i < len(nrmse_means) and nrmse_means[i] is not None:
                improvement = ((original_avg_nrmse - nrmse_means[i]) / original_avg_nrmse * 100)
                print(f"  {approach_name:20s}: {nrmse_means[i]:.4f}±{nrmse_stds[i]:.4f} NRMSE ({improvement:5.1f}% improvement)")
    
    # Save numerical results
    results_df = pd.DataFrame({
        'proposal_distribution_size': num_samples_range,
        'rephrasing_nrmse_mean': rephrasing_nrmse_means,
        'rephrasing_nrmse_std': rephrasing_nrmse_stds,
        'variability_nrmse_mean': variability_nrmse_means,
        'variability_nrmse_std': variability_nrmse_stds,
        'gaussian_nrmse_mean': gaussian_nrmse_means,
        'gaussian_nrmse_std': gaussian_nrmse_stds,
        'rephrasing_improvement_percent': [((original_avg_nrmse - nrmse) / original_avg_nrmse * 100) if nrmse else 0 
                                         for nrmse in rephrasing_nrmse_means],
        'variability_improvement_percent': [((original_avg_nrmse - nrmse) / original_avg_nrmse * 100) if nrmse else 0 
                                          for nrmse in variability_nrmse_means],
        'gaussian_improvement_percent': [((original_avg_nrmse - nrmse) / original_avg_nrmse * 100) if nrmse else 0 
                                       for nrmse in gaussian_nrmse_means]
    })
    results_df.to_csv('action_generation_simple_comparison_results.csv', index=False)
    print(f"\nNumerical results saved as: action_generation_simple_comparison_results.csv")
    
    return results_df

def main():
    # Check if required file types exist
    required_patterns = [
        ('json/bridge_openvla_actions_*.json', 'instruction rephrasing'),
        ('json/bridge_openvla_actions_variability_*.json', 'variability'),
        ('json/bridge_openvla_gaussian_actions_*.json', 'gaussian')
    ]
    
    for pattern, description in required_patterns:
        files = glob.glob(pattern)
        files = [f for f in files if 'variability' not in f and 'gaussian' not in f] if description == 'instruction rephrasing' else files
        if not files:
            print(f"Error: No {description} files found matching pattern: {pattern}")
            return
        print(f"Found {len(files)} {description} files")
    
    # Generate the comparison plot
    results = generate_simple_comparison_plot()
    
    print(f"\n=== Summary ===")
    print("Generated simplified comparison of three action generation approaches for 1, 10, 100 samples:")
    print("1. Instruction Rephrasing: Using different phrasings of the same instruction")
    print("2. Repeated Sampling: Using temperature-based variability with the same instruction")
    print("3. Gaussian Perturbation: Using statistical perturbation of action distributions")
    print("\nAll approaches use oracle selection (perfect knowledge) to choose the best candidate.")

if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as e:
        print(f"Error: Missing required library: {e}")
        print("Please install matplotlib and pandas: pip install matplotlib pandas")
        exit(1)
    
    main()

