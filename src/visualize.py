import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from IPython.display import display, HTML
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def violin_historgrams(sub_path, energy_fam, energy_nov, energy_test_nov, base_fam_color = 'C3', base_nov_color = 'C4', test_color = 'C2', base_class = [4], test_class=[5], n_layers=2):
    fig, ax = plt.subplots(n_layers+1, 1, figsize=(2, 7))

    for l in range(n_layers+1):
        energy_base_fam = energy_fam[:, l]
        energy_base_nov = energy_nov[:, l]
        energy_test = energy_test_nov[:, l]
        
        data = [energy_base_fam, energy_base_nov, energy_test]
        
        # Create violin plots
        parts = ax[l].violinplot(data, showmeans=False, showmedians=True)

        # Coloring each violin plot
        for i, body in enumerate(parts['bodies']):
            body.set_facecolor([base_fam_color, base_nov_color, test_color][i])
            body.set_alpha(0.75)  # Set transparency

        # Set color for the other components: medians, caps, etc.
        parts['cmedians'].set_edgecolor('black')  # Set color of medians
        parts['cmaxes'].set_edgecolor('black')    # Set color of the max caps
        parts['cmins'].set_edgecolor('black')     # Set color of the min caps

        if l == 0:
            ax[l].legend([f"familiar '{base_class[0]}'", f"novel '{base_class[0]}'", f"novel '{test_class[0]}'"], loc='upper left')
        
        ax[l].set_xticks([])
        ax[l].set_yticks([])

    plt.savefig(sub_path + f'/PCN_energy_distibutions_layer_{n_layers-l}.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def add_panel_letters(axes, x_nudge = 0.12, y_nudge = 0.05, fontsize=24, font='Arial', fontweight='bold'):
    panels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] # Add more letters if needed

    for i, ax in enumerate(axes):
        if isinstance(ax, plt.Axes):
            ax.text(-x_nudge, 1+y_nudge, panels[i], transform=ax.transAxes,
                    fontsize=fontsize, fontname=font, fontweight=fontweight)

def plot_separability(sub_path, sep_12, sep_23, n_layers=2):
    plt.figure(figsize=(4, 3))
    plt.title(f"d' separability between classes of digits by layers")
    
    avg12, yerr12 = calculate_errorbars(sep_12, axis=0)
    avg23, yerr23 = calculate_errorbars(sep_23, axis=0)
    
    # x-axis locations for the groups
    layers = np.arange(n_layers + 1)
    
    # Width of the bars
    bar_width = 0.35
    
    # Plotting the bars
    # plt.bar(layers - bar_width / 2, avg12, bar_width, yerr=yerr12,
    #         label=f'sep. between familiar {base_class} and novel {base_class}', capsize=5)
    # plt.bar(layers + bar_width / 2, avg23, bar_width, yerr=yerr23,
    #         label=f'sep. between novel {base_class} and novel {test_class}', capsize=5)
    plt.bar(layers - bar_width / 2, avg12, bar_width, yerr=yerr12,
            label=f'sensory novelty', capsize=3)
    plt.bar(layers + bar_width / 2, avg23, bar_width, yerr=yerr23,
            label=f'semantic novelty', capsize=3)
    
    plt.ylabel("d'")
    plt.xlabel('layer')
    plt.xticks(layers, [f'${i}$' for i in range(n_layers + 1)])
    
    plt.legend()
    plt.savefig(sub_path + 'separability_by_layers.pdf', format = 'pdf' , bbox_inches='tight')
    plt.show()

def plot_mses(sub_path, train_mses):
    plt.figure()
    # Convert to a list of Python floats for plotting
    mses = [float(x.cpu().numpy()) for x in train_mses]
    # Create an array for the x-axis (epochs or iterations)
    epochs = list(range(1, len(mses) + 1))
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mses, marker='o', linestyle='-')
    # Add labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training MSE Over Epochs')
    # Show grid
    plt.grid(True)
    plt.savefig(sub_path+'/train_mses', dpi=300)
    plt.show()

# axis specifies the random/seed dimension. e.g., axis = 1 means averaging across seeds for each sample size
def calculate_errorbars(data, axis, num_seeds=16, n_layers=2):
    avg = np.mean(data, axis=axis)
    error_values = (np.std(data, axis=axis, ddof=1))/np.sqrt(num_seeds)
    return avg, error_values

def plot_weights(result_path, pcn, n_layers=2):
    accumulated_weight_np = pcn.layers[n_layers-1].weight.cpu().detach().numpy()
    for k in range(n_layers):
        j = layer_convert(k)
        weight_matrix_np = pcn.layers[j].weight.cpu().detach().numpy()
        if j < n_layers-1:
            accumulated_weight_np = accumulated_weight_np @ weight_matrix_np
            weight_matrix_np = accumulated_weight_np
        d_out = len(weight_matrix_np[0])
        d_in = len(weight_matrix_np)
        h = int(np.sqrt(d_in))
        w = h
        # Wmin, Wmax = np.min(weight_matrix_np), np.max(weight_matrix_np)
        # fig, axes = plt.subplots(d_out // 32, 32, figsize=(8, (d_out // 32) // 4))
        # fig, axes = plt.subplots(d_out // 20, 20, figsize=(80, (d_out // 80)))
        fig, axes = plt.subplots(d_out // 20, 20)
        # fig, axes = plt.subplots(d_out // 16, 16, figsize=(8, (d_out // 16) // 4))
        for i, ax in enumerate(axes.flatten()):
            f = weight_matrix_np[:, i]
            Wmin, Wmax = -np.max(np.abs(f)), np.max(np.abs(f))
            im = ax.imshow(f.reshape((h, w)), cmap='gray', vmin=Wmin, vmax=Wmax)
            ax.axis('off')
        fig.colorbar(im, ax=axes.ravel().tolist())
        fig.suptitle(f'receptive fields of layer {int(2-j)}')
        # fig.tight_layout()
        plt.savefig(result_path + f'/receptive fields of layer {int(2-j)}.pdf', format = 'pdf', bbox_inches='tight')
        plt.close()

def layer_convert(l_in, n_layers=2):
    l_out = round(n_layers -1 - l_in)
    return l_out