import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


# First dataset
thresholds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 10000]
random_nrmse = [0.4002940512073371, 0.33174401789161556, 0.2717176088822886, 0.2314417335160592,
                0.20408128509190374, 0.18271187011553888, 0.15653528422327104, 0.13487999697838984,
                0.11998678544717495, 0.10961213369362265, 0.10069299261026529, 0.08964190620966579,
                0.08050728763717432, 0.07205178965558706, 0.07020216674473032]
vla_nrmse = [0.17012160864624573, 0.1395698519403141, 0.12382259063373457, 0.10865227458014033,
             0.09702733800476325, 0.09175774016003288, 0.08320682061341735, 0.07728755638551704,
             0.07227932028099078, 0.06884841617423933, 0.06478691782498859, 0.06080920489901476,
             0.0584520262201353, 0.05362861855018124, 0.052690499129027596]
gaussian_nrmse = [0.16328180389104205, 0.14627525175109124, 0.12950288748217434, 0.1130393644551584,
                  0.10673720815944918, 0.10036383805048667, 0.08745513045648391, 0.08271636576683072,
                  0.07830888606635437, 0.0758505702719317, 0.07336043783072331, 0.07108597876572688,
                  0.06899334908909349, 0.06673449881018208, 0.06641024481555212]

# Second dataset
data = {
    'CogAct': {1: 0.14657014921184422, 2: 0.13546936933717782, 4: 0.11325852910238227, 8: 0.1057954748926364,
               16: 0.09832846748572421, 32: 0.09268149555672446, 64: 0.08878742684557418, 128: 0.08494066558603759,
               256: 0.08100591941924964, 512: 0.07832660447680756, 1024: 0.07559469148743564, 2048: 0.07374491666625226,
               4096: 0.07147359574683111, 8192: 0.06950116608977153, 10000: 0.06853431022039472},
    'Octo': {1: 0.20385299367018228, 2: 0.1870831636716739, 4: 0.17272375252347982, 8: 0.15390941713860506,
             16: 0.14726222212820114, 32: 0.13970501717283487, 64: 0.1348948012117828, 128: 0.13012967241193288,
             256: 0.1263447627742102, 512: 0.12322787363823734, 1024: 0.12051496041831536, 2048: 0.1167944122525979,
             4096: 0.11415902149154256, 8192: 0.1114195326387485, 10000: 0.11040842387644544},
    'OpenVLA': {1: 0.16328180389104205, 2: 0.14627525175109124, 4: 0.12950288748217434, 8: 0.1130393644551584,
                16: 0.10673720815944918, 32: 0.10036383805048667, 64: 0.08745513045648391, 128: 0.08271636576683072,
                256: 0.07830888606635437, 512: 0.0758505702719317, 1024: 0.07336043783072331, 2048: 0.07108597876572688,
                4096: 0.06899334908909349, 8192: 0.06673449881018208, 10000: 0.06641024481555212},
    'SpatialVLA': {1: 0.13521099525216346, 2: 0.12605564459950971, 4: 0.11771072318411192, 8: 0.11150464387097633,
                   16: 0.10673824307759809, 32: 0.10316706858160037, 64: 0.10019826304141226, 128: 0.09730298094213591,
                   256: 0.09461648228261214, 512: 0.09262494589354095, 1024: 0.09070842064913645, 2048: 0.08878973239877454,
                   4096: 0.08732165277344606, 8192: 0.08560359018037329, 10000: 0.08522252241687622}
}

# Define model
def anchored_power_law(x, b, y0):
    return y0 * np.power(x, b)

def fit_with_fixed_y0(xdata, ydata, y0):
    def model(x, b):
        return anchored_power_law(x, b, y0)
    popt, _ = curve_fit(model, xdata, ydata, p0=[-0.5])
    return popt[0]

def compute_error(xdata, ydata, b, y0):
    y_fit = anchored_power_law(np.array(xdata), b, y0)
    errors = y_fit - np.array(ydata)
    rmse = np.sqrt(np.mean(errors ** 2))
    std = np.std(errors)
    return rmse, std

# Fit data
x_fit = np.logspace(np.log10(min(thresholds)), np.log10(max(thresholds)), 200)
x_fit_2 = np.logspace(0, np.log10(10000), 200)

# Custom colors
coral = '#8073ac'        
lightBlue = '#045a8d'   
teal = '#f68c1e'          
colors_right = ['#f47f65', '#6aa7d6', '#45b39d', 'gray']

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), dpi=1000)

# LEFT PLOT
y0_random, y0_vla, y0_gaussian = random_nrmse[0], vla_nrmse[0], gaussian_nrmse[0]
b_random = fit_with_fixed_y0(thresholds, random_nrmse, y0_random)
b_vla = fit_with_fixed_y0(thresholds, vla_nrmse, y0_vla)
b_gaussian = fit_with_fixed_y0(thresholds, gaussian_nrmse, y0_gaussian)

rmse_random, std_random = compute_error(thresholds, random_nrmse, b_random, y0_random)
rmse_vla, std_vla = compute_error(thresholds, vla_nrmse, b_vla, y0_vla)
rmse_gaussian, std_gaussian = compute_error(thresholds, gaussian_nrmse, b_gaussian, y0_gaussian)

# Use $\mathbf{}$ for bold method names only
axes[0].plot(x_fit, anchored_power_law(x_fit, b_vla, y0_vla), 
             label=f'$\\mathbf{{Policy\\ Sampling}}$ (a={y0_vla:.2f}, b={b_vla:.2f})\nError: {rmse_vla:.4f} ± {std_vla:.4f}', 
             color=coral, linewidth=3.4)
axes[0].plot(x_fit, anchored_power_law(x_fit, b_gaussian, y0_gaussian), 
             label=f'$\\mathbf{{Gaussian\\ Perturbation}}$ (a={y0_gaussian:.2f}, b={b_gaussian:.2f})\nError: {rmse_gaussian:.4f} ± {std_gaussian:.4f}', 
             color=teal, linewidth=3.4)
axes[0].plot(x_fit, anchored_power_law(x_fit, b_random, y0_random), 
             label=f'$\\mathbf{{Random\\ Sampling}}$ (a={y0_random:.2f}, b={b_random:.2f})\nError: {rmse_random:.4f} ± {std_random:.4f}', 
             color=lightBlue, linewidth=3.4)
axes[0].axhline(y=0.1607, color='#A9A9A9', linestyle='--', label='$\\mathbf{OpenVLA}$ (Single-Attempt)', linewidth=2.2)
axes[0].set_xscale('log')
axes[0].set_xlabel('Number of Samples (k)', fontsize=13.2, weight='bold')
axes[0].set_ylabel('Oracle Action Error (e)', fontsize=13.2, weight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(False)

# Make tick labels larger for left plot
axes[0].tick_params(axis='both', which='major', labelsize=12)

# RIGHT PLOT
for i, (label, values) in enumerate(data.items()):
    xdata = np.array(list(values.keys()))
    ydata = np.array(list(values.values()))
    y0 = ydata[0]
    b = fit_with_fixed_y0(xdata, ydata, y0)
    y_fit_dense = anchored_power_law(x_fit_2, b, y0)
    y_fit_actual = anchored_power_law(xdata, b, y0)
    errors = y_fit_actual - ydata
    rmse = np.sqrt(np.mean(errors ** 2))
    std = np.std(errors)

    # Make only the method name bold using $\mathbf{}$
    axes[1].plot(x_fit_2, y_fit_dense, 
                 label=f'$\\mathbf{{{label}}}$ (a={y0:.2f}, b={b:.2f})\nError: {rmse:.4f} ± {std:.4f}', 
                 color=colors_right[i], linewidth=3.4)

axes[1].set_xscale('log')
axes[1].set_xlabel('Number of Samples (k)', fontsize=13.2, weight='bold')
# axes[1].set_ylabel('Oracle Action Error (e)', fontsize=13.2, weight='bold')
axes[1].tick_params(axis='y', which='both', labelleft=True)
axes[1].legend(fontsize=9.2)
axes[1].grid(False)

# Make tick labels larger for right plot
axes[1].tick_params(axis='both', which='major', labelsize=12)

# LEFT PLOT legend
legend = axes[0].legend(fontsize=9.8)
for text in legend.get_texts():
    text.set_color("#535353")

# RIGHT PLOT legend
legend = axes[1].legend(fontsize=9.1)
for text in legend.get_texts():
    text.set_color("#535353")

plt.tight_layout()
plt.show()