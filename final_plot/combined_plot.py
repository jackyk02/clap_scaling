#!/usr/bin/env python3
"""
Combined plot matching reference style with left and right panels.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_combined_plot():
    """Create combined plot with reference style"""
    
    # LEFT PLOT DATA
    proposal_distribution_sizes = [1, 10, 100]
    training_flops = [3.7e12, 3.7e13, 3.7e14, 3.7e15]  # Total inference FLOPS
    
    # NRMSE values for each approach
    rephrasing_nrmse_means = [0.16169976855118576, 0.09869752183810761, 0.07053206710866165, 0.05237526]
    variability_nrmse_means = [0.16255906402794928, 0.10831067086202477, 0.08272804579448052, 0.0648]
    gaussian_nrmse_means = [0.16255906402794928, 0.11937213896670357, 0.0921677874498369, 0.0734]
    hybrid_nrmse_means = [0.16169, 0.09, 0.033, 0.0190138680743496887]
    
    # RIGHT PLOT DATA
    oracle_data = [
        [1, 1, 0.1603390303764664, 3.694852775178287],
        [1, 2, 0.14130218929586325, 15.129035573081179],
        [1, 4, 0.12666522365277375, 23.92050155533578],
        [1, 8, 0.053826734253342785, 67.66980843823055],
        [1, 16, 0.05109756767807141, 69.30903993548642],
        [1, 32, 0.04888225774632862, 70.63963142418494],
        [1, 64, 0.0476630487511084, 71.3719303629201],
        [1, 128, 0.045513571697352394, 72.66298035637654],
        [2, 1, 0.13858897921116284, 16.75868305220985],
        [2, 2, 0.12329264975444483, 25.946185663831105],
        [2, 4, 0.11098607717093063, 33.3379372648327],
        [2, 8, 0.048959123412499285, 70.59346325614926],
        [2, 16, 0.046182320123680135, 72.26130700521155],
        [2, 32, 0.04399364299240911, 73.57590191613096],
        [2, 64, 0.042649086341030526, 74.38348897686782],
        [2, 128, 0.041366332776590686, 75.15395497374013],
        [4, 1, 0.1204957902157559, 27.626075887740786],
        [4, 2, 0.10561786758856542, 36.56226893847154],
        [4, 4, 0.09467232407124185, 43.13653957873017],
        [4, 8, 0.042844546269913214, 74.26608900766878],
        [4, 16, 0.03916073554147445, 76.47871268212717],
        [4, 32, 0.03708827889955499, 77.72350156196816],
        [4, 64, 0.03658094209183112, 78.0282255324556],
        [4, 128, 0.03437798871578624, 79.35139524797208],
        [8, 1, 0.10531546139604514, 36.743904519178145],
        [8, 2, 0.09183788437021523, 44.83900174322205],
        [8, 4, 0.08022594395685306, 51.8135333245159],
        [8, 8, 0.03539002818935925, 78.74352946338465],
        [8, 16, 0.031486659935994796, 81.08802695367339],
        [8, 32, 0.030096892449701883, 81.92276919990913],
        [8, 64, 0.02997903788798603, 81.99355671116271],
        [8, 128, 0.028027947144968705, 83.16544904962885],
        [16, 1, 0.0919721794971851, 44.7583394619045],
        [16, 2, 0.07935814378093344, 52.334764016178696],
        [16, 4, 0.06993926090421514, 57.99206966412931],
        [16, 8, 0.031166097174692892, 81.28056799529931],
        [16, 16, 0.029066684240780183, 82.54154775307533],
        [16, 32, 0.028065591189234275, 83.14283873935928],
        [16, 64, 0.026668892646895828, 83.98174401671301],
        [16, 128, 0.025190781679560813, 84.86954840214301],
        [32, 1, 0.08070449270707433, 51.52610045345448],
        [32, 2, 0.07228534086630459, 56.58293318862652],
        [32, 4, 0.0642512246609487, 61.40850025212603],
        [32, 8, 0.027330583009847795, 83.58430998164546],
        [32, 16, 0.025475640464936094, 84.69845239888394],
        [32, 32, 0.023856411945122016, 85.67101684950576],
        [32, 64, 0.02283370663010111, 86.2852889060315],
        [32, 128, 0.022046465586119054, 86.75813300683531],
        [64, 1, 0.0737676267130893, 55.69262122672492],
        [64, 2, 0.06582479425358795, 60.463360126656816],
        [64, 4, 0.05941463791262198, 64.31352093093133],
        [64, 8, 0.02515645883021043, 84.89016388829198],
        [64, 16, 0.023138680743496887, 86.10211094355175],
        [64, 32, 0.022219584912184027, 86.65415157358724],
        [64, 64, 0.0210514016079542, 87.35580272387351],
        [64, 128, 0.020778986560857204, 87.51942458908835]
    ]
    
    # Create DataFrame from oracle data
    df = pd.DataFrame(oracle_data, columns=['num_rephrases', 'samples_per_rephrase', 'oracle_nrmse', 'improvement_percent'])
    
    # Create figure with reference style dimensions
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), dpi=1000)
    
    # Colors matching reference style
    coral = '#8073ac'        
    lightBlue = '#045a8d'   
    teal = '#f68c1e'
    gray = 'gray'
    colors_right = ['#f47f65', '#6aa7d6', '#45b39d', 'gray', '#9b59b6']
    
    # LEFT PLOT with different markers - reordered for legend, swapped colors
    axes[0].plot(training_flops, hybrid_nrmse_means, 
                label='Hybrid Scaling', 
                color=teal, linewidth=3.4, marker='D', markersize=8)
    
    axes[0].plot(training_flops, rephrasing_nrmse_means, 
                label='Instruction Rephrasing', 
                color=coral, linewidth=3.4, marker='o', markersize=8)
    
    axes[0].plot(training_flops, variability_nrmse_means, 
                label='Repeated Sampling', 
                color=lightBlue, linewidth=3.4, marker='s', markersize=8)
    
    axes[0].plot(training_flops, gaussian_nrmse_means, 
                label='Gaussian Perturbation', 
                color=gray, linewidth=3.4, marker='^', markersize=8)
    
    # Add shaded region between Hybrid Scaling and Repeated Sampling
    axes[0].fill_between(training_flops, hybrid_nrmse_means, variability_nrmse_means, 
                        alpha=0.3, color='lightgray')
    
    axes[0].set_xlabel('Total Inference FLOPS', fontsize=11, weight='bold')
    axes[0].set_ylabel('Oracle Action Error', fontsize=11, weight='bold')
    axes[0].set_xscale('log')
    axes[0].set_xticks(training_flops)
    axes[0].set_xticklabels(['10¹²', '10¹³', '10¹⁴', '10¹⁵'])
    axes[0].set_ylim(0, 0.175)
    axes[0].grid(False)
    axes[0].tick_params(axis='both', which='major', labelsize=11)
    
    # LEFT PLOT legend with black text
    legend = axes[0].legend(fontsize=10.5, labelspacing=0.6)
    for text in legend.get_texts():
        text.set_color("black")
    
    # RIGHT PLOT - keep 2, 4, 8, 16, 64 rephrases
    selected_rephrases = [2, 4, 8, 16, 64]
    markers = ['v', '<', '>', 'P', 's']  # Different markers from left plot
    
    # Plot each selected rephrase count as a separate curve
    for i, num_rephrases in enumerate(selected_rephrases):
        # Filter data for this number of rephrases
        rephrase_data = df[df['num_rephrases'] == num_rephrases].copy()
        rephrase_data = rephrase_data.sort_values('samples_per_rephrase')
        
        samples_x = rephrase_data['samples_per_rephrase'].values
        nrmse_y = rephrase_data['oracle_nrmse'].values
        
        # Plot with reference styling and different markers
        axes[1].plot(samples_x, nrmse_y, 
                    label=f'{num_rephrases} Rephrases', 
                    color=colors_right[i], linewidth=3.4, 
                    marker=markers[i], markersize=8)
    
    axes[1].set_xlabel('Number of Generated Actions per Rephrase', fontsize=11, weight='bold')
    # Remove y-axis label for right plot
    axes[1].set_xscale('log', base=2)
    
    # Set x-axis ticks for right plot
    samples_per_rephrase_list = sorted(df['samples_per_rephrase'].unique())
    axes[1].set_xticks(samples_per_rephrase_list)
    axes[1].set_xticklabels([str(t) for t in samples_per_rephrase_list])
    
    # Set y-axis limits for right plot
    y_max = 0.175
    axes[1].set_ylim(0, y_max)
    
    axes[1].grid(False)
    axes[1].tick_params(axis='both', which='major', labelsize=11)
    
    # RIGHT PLOT legend with black text
    legend = axes[1].legend(fontsize=11, labelspacing=0.6)
    for text in legend.get_texts():
        text.set_color("black")
    
    plt.tight_layout()
    
    # Save as PNG
    output_file = 'combined_plot.png'
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"Plot saved as: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    create_combined_plot()
