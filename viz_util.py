import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colormaps
cmap = colormaps.get_cmap("jet")

def plot_agent_history(agent):
    n = len(agent.q_history)

    fig, axs = plt.subplots(n, 2, figsize=(12, 2*n))

    for i, (episode, (q_values, path)) in enumerate(agent.q_history.items()):
        for ax in axs[i, :]:
            setup_axis(ax, agent.env.env_map)
            agent.env.render(ax)
        plot_path(axs[i, 0], path, episode, agent.env.env_map)
        plot_policy(axs[i, 1], q_values, episode, agent.env.env_map)

    plt.tight_layout()
    plt.show()


def setup_axis(ax, env_layout):
    xticks = np.arange(0, env_layout.shape[1]) - 0.5
    yticks = np.arange(0, env_layout.shape[0]) - 0.5
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim(xticks[0], xticks[-1] + 1)
    ax.set_ylim(yticks[0], yticks[-1] + 1)
    ax.grid(lw=1, c='k')

    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plot_policy(ax, q_values, episode,  env_layout):
    actions = {
        0: "v",
        1: ">",
        2: "^",
        3: "<"
    }
    policy = q_values.argmax(axis=-1)
    xs = np.arange(env_layout.shape[1])
    ys = np.arange(env_layout.shape[0])
    xs, ys = np.meshgrid(xs, ys, indexing="xy")

    for x, y, action in zip(xs.flatten(), ys.flatten(), policy):
        ax.scatter(x, y, marker=actions[action], c='k')
    ax.set_title(f"Optimal policy after episode {episode}")


def plot_path(ax, path, episode, env_layout):
    path = np.array(path)
    xs = path % env_layout.shape[1]
    ys = path // env_layout.shape[0]

    dxs = np.diff(xs)
    dys = np.diff(ys)
    colours = np.linspace(0, 1, len(dxs))
    offsets = np.linspace(-0.2, 0.2, len(dxs))[::-1]
    for x, y, dx, dy, c, offset in zip(xs, ys, dxs, dys, colours, offsets):
        ax.arrow(
            x + 0 - offset, y + 0 - offset, 0.9*dx, 0.9*dy,
            head_length=0.2, length_includes_head=True,
            width=0.1,
            fc=cmap(c), alpha=0.5,
            zorder=3
        )
    ax.set_title(f"Path during episode {episode}")