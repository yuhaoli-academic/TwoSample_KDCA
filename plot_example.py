#%%
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Data: Only 5% significance level values
data_5_percent = {
    0.6: {
        'multi':   [0.163, 0.319, 0.877, 0.995],
        'single':  [0.198, 0.303, 0.865, 0.987],
        'mmd':     [0.049, 0.071, 0.086, 0.074]
    },
    0.8: {
        'multi':   [0.097, 0.122, 0.4, 0.675],
        'single':  [0.091, 0.109, 0.402, 0.666],
        'mmd':     [0.039, 0.062, 0.049, 0.057]
    },
    1.3: {
        'multi':   [0.169, 0.317, 0.887, 0.992],
        'single':  [0.161, 0.282, 0.85, 0.992],
        'mmd':     [0.057, 0.054, 0.068, 0.097]
    }
}

dimensions = [50, 100, 500, 1000]

# Directory to save figures
save_dir = "/home/lyh/Seafile/MEGAsync/Projects/TwoSample_KDCA/draft/arxiv/"

# Plot and save each figure as PDF
for idx, (sigma2, stats) in enumerate(data_5_percent.items()):
    pdf_path = f"{save_dir}ub_set_4_{idx + 1}.pdf"
    
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(dimensions, stats['multi'], marker='o', label=r'$\hat{T}_{N,\text{multi}}$')
        ax.plot(dimensions, stats['single'], marker='s', label=r'$\hat{T}_{N,\text{single}}$')
        ax.plot(dimensions, stats['mmd'], marker='^', label=r'$\hat{T}_{N,\text{MMD}}$')

        ax.set_title(rf'Rejection Rate at 5% Level ($\sigma^2={sigma2}$)')
        ax.set_xlabel('Dimension $d$')
        ax.set_ylabel('Rejection Rate (5%)')
        ax.set_xscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()
        plt.tight_layout()

        pdf.savefig(fig)  # Save the current figure to the PDF
        plt.close()

print("PDF figures saved successfully.")
# %%
