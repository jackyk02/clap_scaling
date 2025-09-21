#!/usr/bin/env python3
"""
Left plot for action generation comparison matching reference style.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_left():
    """Generate left plot matching reference style"""
    
    # Data stored directly in the script
    proposal_distribution_sizes = [1, 10, 100]
    training_flops = [3.7e12, 3.7e13, 3.7e14, 3.7e15]  # Total inference FLOPS
    
    # NRMSE values for each approach
    rephrasing_nrmse_means = [0.16169976855118576, 0.09869752183810761, 0.07053206710866165, 0.05237526]
    variability_nrmse_means = [0.16255906402794928, 0.10831067086202477, 0.08272804579448052, 0.0648]
    gaussian_nrmse_means = [0.16255906402794928, 0.11937213896670357, 0.0921677874498369, 0.0734]
    hybrid_nrmse_means = [0.16169, 0.09, 0.033, 0.023138680743496887]
    
    # Create figure with reference style
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=1000)
    
    # Colors matching reference style
    coral = '#8073ac'        
    lightBlue = '#045a8d'   
    teal = '#f68c1e'
    gray = 'gray'
    
    # Plot each approach using training FLOPS on x-axis - reordered for legend, swapped colors
    ax.plot(training_flops, hybrid_nrmse_means, 
            label='Hybrid Scaling', 
            color=teal, linewidth=3.4, marker='D', markersize=8)
    
    ax.plot(training_flops, rephrasing_nrmse_means, 
            label='Instruction Rephrasing', 
            color=coral, linewidth=3.4, marker='o', markersize=8)
    
    ax.plot(training_flops, variability_nrmse_means, 
            label='Repeated Sampling', 
            color=lightBlue, linewidth=3.4, marker='s', markersize=8)
    
    ax.plot(training_flops, gaussian_nrmse_means, 
            label='Gaussian Perturbation', 
            color=gray, linewidth=3.4, marker='^', markersize=8)
    
    # Add shaded region between Hybrid Scaling and Repeated Sampling
    ax.fill_between(training_flops, hybrid_nrmse_means, variability_nrmse_means, 
                   alpha=0.3, color='lightgray')
    
    # Axis labels with smaller font size
    ax.set_xlabel('Total Inference FLOPS', fontsize=11, weight='bold')
    ax.set_ylabel('Oracle Action Error', fontsize=11, weight='bold')
    ax.set_xscale('log')
    ax.set_xticks(training_flops)
    ax.set_xticklabels(['10¹²', '10¹³', '10¹⁴'])
    
    # Set y-axis limits
    ax.set_ylim(0, 0.175)
    
    # Remove grid (reference style)
    ax.grid(False)
    
    # Legend with black text
    legend = ax.legend(fontsize=10.5, labelspacing=0.6)
    for text in legend.get_texts():
        text.set_color("black")
    
    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    # Save as PNG
    output_file = 'left_plot.png'
    plt.savefig(output_file, dpi=1000, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    plot_left()
