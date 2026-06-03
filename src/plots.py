from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

def plot_chain_animation(data):
    data_array = np.array(data) 
    chain_length = len(data_array[0]) // 2
    plt.figure(figsize=(10, 6))

    n_qubits = data_array.shape[1]
    time_steps = np.arange(data_array.shape[0])

    upper_chain = data_array[:, :chain_length]
    lower_chain = data_array[:, chain_length:]

    site_indices = np.arange(chain_length)

    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # y_min = float(data_array.min()) - 0.1
    # y_max = float(data_array.max()) + 0.1

    y_min = -1.1
    y_max = +1.1

    line_up, = ax_up.plot(site_indices, upper_chain[0], marker='o', color='tab:blue')
    ax_up.set_ylim(y_min, y_max)
    ax_up.set_xticks(site_indices)
    ax_up.set_ylabel(r'$\langle Z \rangle$')
    ax_up.set_title('Upper chain')
    ax_up.grid(True, linestyle='--', alpha=0.6)

    line_down, = ax_down.plot(site_indices, lower_chain[0], marker='o', color='tab:red')
    ax_down.set_ylim(y_min, y_max)
    ax_down.set_xticks(site_indices)
    ax_down.set_xlabel('Site index')
    ax_down.set_ylabel(r'$\langle Z \rangle$')
    ax_down.set_title('Lower chain')
    ax_down.grid(True, linestyle='--', alpha=0.6)

    time_text = fig.suptitle(f't = {time_steps[0]:.2f}')

    stride = 5 # keep 1 of every stride frames
    frame_indices = np.arange(0, len(time_steps), stride)

    def update(i):
        frame = frame_indices[i]
        line_up.set_ydata(upper_chain[frame])
        line_down.set_ydata(lower_chain[frame])
        time_text.set_text(f't = {time_steps[frame]:.2f}')
        return line_up, line_down, time_text

    anim = FuncAnimation(fig, update, frames=len(frame_indices), interval=50, blit=False)

    return anim, fig

