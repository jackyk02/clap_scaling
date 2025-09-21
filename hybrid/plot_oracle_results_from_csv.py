#!/usr/bin/env python3
"""
Plot oracle verifier results from embedded data.

This script contains the oracle verifier performance data embedded directly
and generates a plot showing oracle performance vs samples per rephrase for 
different numbers of rephrases.
"""

import pandas as pd
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

def plot_oracle_results():
    """
    Generate the oracle verifier plot from embedded data.
    """
    
    # Embedded oracle results data
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
    
    # Create DataFrame from embedded data
    df = pd.DataFrame(oracle_data, columns=['num_rephrases', 'samples_per_rephrase', 'oracle_nrmse', 'improvement_percent'])
    print(f"Using embedded oracle results data")
    
    print(f"Loaded {len(df)} data points")
    print(f"Number of rephrases: {sorted(df['num_rephrases'].unique())}")
    print(f"Samples per rephrase: {sorted(df['samples_per_rephrase'].unique())}")
    
    # Calculate baseline NRMSE (assuming ~16.65% based on previous results)
    # We can infer this from the improvement percentages
    sample_row = df.iloc[0]
    baseline_nrmse = sample_row['oracle_nrmse'] / (1 - sample_row['improvement_percent']/100)
    print(f"Inferred baseline NRMSE: {baseline_nrmse:.4f}")
    
    # Create the plot
    fig, ax = plt.subplots()
    
    # Colors for different rephrase counts
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Get unique rephrase counts and sort them, excluding 1 rephrase
    rephrase_counts = sorted([r for r in df['num_rephrases'].unique() if r != 1])
    
    
    # Plot each rephrase count as a separate curve (behind the main curves)
    for i, num_rephrases in enumerate(rephrase_counts):
        # Filter data for this number of rephrases
        rephrase_data = df[df['num_rephrases'] == num_rephrases].copy()
        rephrase_data = rephrase_data.sort_values('samples_per_rephrase')
        
        samples_x = rephrase_data['samples_per_rephrase'].values
        nrmse_y = rephrase_data['oracle_nrmse'].values
        
        # Plot the curve for this number of rephrases (with lower zorder so they appear behind)
        ax.plot(samples_x, nrmse_y, 'o-', linewidth=2.0, markersize=7, 
               color=colors[i % len(colors)], marker=markers[i % len(markers)], 
               label=f'{num_rephrases} Rephrases', markeredgewidth=1, markeredgecolor='white',
               zorder=3)
        
        print(f"\n{num_rephrases} Rephrases:")
        for _, row in rephrase_data.iterrows():
            augmentation_note = " (augmented)" if row['samples_per_rephrase'] > 4 else ""
            print(f"  {int(row['samples_per_rephrase']):3d} samples: {row['oracle_nrmse']:.4f} NRMSE ({row['improvement_percent']:.1f}% improvement){augmentation_note}")
    
    
    # Axis labels and formatting
    ax.set_xlabel("Number of Generated Actions per Rephrase")
    ax.set_ylabel("Action Error (Average NRMSE)")
    ax.set_xscale('log', base=2)
    
    # Set x-axis ticks
    samples_per_rephrase_list = sorted(df['samples_per_rephrase'].unique())
    ax.set_xticks(samples_per_rephrase_list)
    ax.set_xticklabels([str(t) for t in samples_per_rephrase_list])
    
    # Set y-axis limits to better show the improvement
    all_nrmse_values = list(df['oracle_nrmse'].values)
    y_min = min(all_nrmse_values) * 0.9
    y_max = 0.143
    ax.set_ylim(y_min, y_max)
    
    # Grid and border
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # Create legend
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.95, edgecolor='black', fontsize=10)
    legend.get_frame().set_linewidth(1.5)
    
    # Title
    ax.set_title('Oracle Verifier Performance Comparison\nAcross Different Sample Counts (with Gaussian Augmentation)', 
                 fontsize=14, pad=20)
    
    # Save plot
    output_file = 'oracle_results_from_csv.png'
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_file}")
    
    # # Also save as PDF
    # plt.savefig('/root/validate_clip_verifier/oracle_results_from_csv.pdf', bbox_inches='tight')
    # print(f"Plot also saved as: oracle_results_from_csv.pdf")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Oracle Results Summary ===")
    print(f"Baseline NRMSE: {baseline_nrmse:.4f}")
    
    # Find best overall result from oracle data
    best_result = df.loc[df['oracle_nrmse'].idxmin()]
    print(f"\nBest Oracle Performance:")
    print(f"  {int(best_result['num_rephrases'])} rephrases with {int(best_result['samples_per_rephrase'])} samples per rephrase")
    print(f"  NRMSE: {best_result['oracle_nrmse']:.4f} ({best_result['improvement_percent']:.1f}% improvement)")
    
    
    # Analyze the effect of augmentation (excluding 1 rephrase)
    print(f"\nAugmentation Effect Analysis:")
    for num_rephrases in rephrase_counts:
        rephrase_data = df[df['num_rephrases'] == num_rephrases].copy()
        rephrase_data = rephrase_data.sort_values('samples_per_rephrase')
        
        # Find the jump from 4 to 8 samples (where augmentation starts)
        four_samples = rephrase_data[rephrase_data['samples_per_rephrase'] == 4]
        eight_samples = rephrase_data[rephrase_data['samples_per_rephrase'] == 8]
        
        if len(four_samples) > 0 and len(eight_samples) > 0:
            improvement_jump = eight_samples.iloc[0]['improvement_percent'] - four_samples.iloc[0]['improvement_percent']
            print(f"  {int(num_rephrases):2d} rephrases: +{improvement_jump:.1f}% improvement jump (4→8 samples)")
    
    # Analyze scaling with number of rephrases at high sample count
    print(f"\nScaling with Number of Rephrases (at 128 samples per rephrase):")
    high_sample_data = df[df['samples_per_rephrase'] == 128].copy()
    high_sample_data = high_sample_data.sort_values('num_rephrases')
    
    for _, row in high_sample_data.iterrows():
        print(f"  {int(row['num_rephrases']):2d} rephrases: {row['oracle_nrmse']:.4f} NRMSE ({row['improvement_percent']:.1f}% improvement)")
    
    return df

def analyze_diminishing_returns(df):
    """
    Analyze diminishing returns in the oracle results.
    
    Args:
        df: DataFrame containing oracle results
    """
    print(f"\n=== Diminishing Returns Analysis ===")
    
    # For each number of rephrases, analyze the marginal benefit of additional samples (excluding 1 rephrase)
    rephrase_counts = sorted([r for r in df['num_rephrases'].unique() if r != 1])
    
    for num_rephrases in rephrase_counts:
        print(f"\n{int(num_rephrases)} Rephrases - Marginal Benefit per Additional Sample:")
        rephrase_data = df[df['num_rephrases'] == num_rephrases].copy()
        rephrase_data = rephrase_data.sort_values('samples_per_rephrase')
        
        prev_improvement = None
        prev_samples = None
        
        for _, row in rephrase_data.iterrows():
            if prev_improvement is not None:
                marginal_benefit = row['improvement_percent'] - prev_improvement
                additional_samples = int(row['samples_per_rephrase']) - int(prev_samples)
                efficiency = marginal_benefit / additional_samples if additional_samples > 0 else 0
                
                print(f"  {int(prev_samples):3d} → {int(row['samples_per_rephrase']):3d} samples: +{marginal_benefit:5.1f}% ({efficiency:5.2f}% per sample)")
            
            prev_improvement = row['improvement_percent']
            prev_samples = row['samples_per_rephrase']

if __name__ == "__main__":
    # Generate the plot from embedded data
    df = plot_oracle_results()
    
    # Perform additional analysis
    analyze_diminishing_returns(df)
    
    print(f"\nPlot generation complete!")
