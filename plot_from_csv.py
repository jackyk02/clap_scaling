#!/usr/bin/env python3
"""
Plot action generation comparison with stored data using the same style as generate_plot_simple.py
Self-contained script with embedded data for plotting four action generation approaches.
"""

import numpy as np
import matplotlib.pyplot as plt

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

def plot_from_data():
    """Generate comparison plot from stored data"""
    
    # Data stored directly in the script
    proposal_distribution_sizes = [1, 10, 100]
    training_flops = [3.7e12, 3.7e13, 3.7e14]  # Total inference FLOPS
    
    # NRMSE values for each approach
    rephrasing_nrmse_means = [0.16169976855118576, 0.09869752183810761, 0.07053206710866165]
    variability_nrmse_means = [0.16255906402794928, 0.10831067086202477, 0.08272804579448052]
    gaussian_nrmse_means = [0.16255906402794928, 0.11937213896670357, 0.0921677874498369]
    hybrid_nrmse_means = [0.16169, 0.09, 0.033]
    
    # Create the comparison plot
    fig, ax = plt.subplots()
    
    # Colors and styling (same as original)
    color_rephrasing = '#1f77b4'  # Blue
    color_variability = '#ff7f0e'  # Orange
    color_gaussian = '#2ca02c'  # Green
    color_hybrid = '#d62728'  # Red
    
    marker_rephrasing = 'o'  # Circle
    marker_variability = 's'  # Square
    marker_gaussian = 'D'  # Diamond
    marker_hybrid = '^'  # Triangle
    
    # Plot each approach using training FLOPS on x-axis
    # Instruction Rephrasing
    ax.plot(training_flops, rephrasing_nrmse_means, label='Instruction Rephrasing', 
            color=color_rephrasing, marker=marker_rephrasing, markersize=10, linestyle='-', linewidth=3.0)
    
    # Repeated Sampling (Variability)
    ax.plot(training_flops, variability_nrmse_means, label='Repeated Sampling', 
            color=color_variability, marker=marker_variability, markersize=10, linestyle='-', linewidth=3.0)
    
    # Gaussian Perturbation
    ax.plot(training_flops, gaussian_nrmse_means, label='Gaussian Perturbation', 
            color=color_gaussian, marker=marker_gaussian, markersize=10, linestyle='-', linewidth=3.0)
    
    # Hybrid Scaling
    ax.plot(training_flops, hybrid_nrmse_means, label='Hybrid Scaling', 
            color=color_hybrid, marker=marker_hybrid, markersize=10, linestyle='-', linewidth=3.0)
    
    # Axis labels and scale (same as original)
    ax.set_xlabel("Total Inference FLOPS")
    ax.set_ylabel("Oracle Action Error (Average NRMSE)")
    ax.set_xscale('log')
    ax.set_xticks(training_flops)
    ax.set_xticklabels(['10¹²', '10¹³', '10¹⁴'])
    
    # Set y-axis limits to specific range (same as original)
    ax.set_ylim(0.02, 0.17)
    
    # Grid and border (same as original)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Legend (same as original)
    legend = ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=14)
    legend.get_frame().set_linewidth(1.5)
    
    # Title (same as original)
    ax.set_title('Oracle Action Error vs Training FLOPS\nfor Different Action Generation Approaches', 
                 fontsize=18, pad=20)
    
    # Save plot
    output_file = 'action_generation_csv_comparison.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Data Summary ===")
    print(f"Data points: {len(proposal_distribution_sizes)}")
    
    for i, size in enumerate(proposal_distribution_sizes):
        flops = training_flops[i]
        print(f"\n{size} candidates (FLOPS: {flops:.1e}):")
        print(f"  Instruction Rephrasing: {rephrasing_nrmse_means[i]:.4f} NRMSE")
        print(f"  Repeated Sampling:      {variability_nrmse_means[i]:.4f} NRMSE")
        print(f"  Gaussian Perturbation:  {gaussian_nrmse_means[i]:.4f} NRMSE")
        print(f"  Hybrid Scaling:         {hybrid_nrmse_means[i]:.4f} NRMSE")

def main():
    # Generate plot from stored data
    try:
        plot_from_data()
    except Exception as e:
        print(f"Error generating plot: {e}")
        return

if __name__ == "__main__":
    # Check if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"Error: Missing required library: {e}")
        print("Please install matplotlib: pip install matplotlib")
        exit(1)
    
    main()
