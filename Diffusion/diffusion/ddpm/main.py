import os

# not sure if this is necessary
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from typing import Any, List

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import animation

from fire import Fire
from tqdm import tqdm
from pydantic import BaseModel

from diffusers import DDPM
from models import BasicDiscreteTimeModel
import pandas as pd


class TrainResult(BaseModel):
    losses: List[float]
    samples: List[Any]


def train(
    model: nn.Module,
    ddpm: DDPM,
    batch_size: int = 128,
    n_epochs: int = 400,
    sample_size: int = 512,
    steps_between_sampling: int = 20,
    seed: int = 42,
) -> TrainResult:
    np.random.seed(seed)
    torch.manual_seed(seed)

    assert batch_size > 0 and steps_between_sampling > 0 and sample_size > 0

    N = 1 << 10
    # Load the data from the CSV file
    data_path = "/Users/ayushchaudhary/models_review/data_normal.csv"
    df = pd.read_csv(data_path)

    # Extract the 'Age' and 'Score' columns for 2D data
    X = df[["age", "score"]].to_numpy()

    optim = torch.optim.Adam(model.parameters(), 1e-3)

    losses: List[float] = []
    samples: List[Any] = []
    step = 0
    avg_loss = None  # exponential moving average
    with tqdm(total=n_epochs * (len(X) // batch_size)) as pbar:
        for _ in range(n_epochs):
            ids = np.random.choice(N, N, replace=False)
            for i in range(0, len(ids), batch_size):
                x = torch.tensor(X[ids[i : i + batch_size]], dtype=torch.float32)
                optim.zero_grad()
                loss = ddpm.diffusion_loss(model, x)
                loss.backward()
                optim.step()

                pbar.update(1)
                losses.append(loss.item())
                if avg_loss is None:
                    avg_loss = losses[-1]
                else:
                    avg_loss = 0.95 * avg_loss + 0.05 * losses[-1]
                if not step % 10:
                    pbar.set_description(f"Iter: {step}. Average Loss: {avg_loss:.04f}")
                if not step % steps_between_sampling:
                    samples.append(ddpm.sample(model, n_samples=sample_size))
                step += 1
    return TrainResult(losses=losses, samples=samples)


def animate(samples: List[Any], save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set(xlim=(0, 100), ylim=(-1, 1))
    scat = ax.scatter(*samples[0].detach().numpy().T, c="k", alpha=0.3)

    def animate(i):
        scat.set_offsets(samples[i].detach().numpy())

    anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(samples) - 1)
    if save:
        anim.save(filename="animation.gif", writer=animation.PillowWriter(fps=5))
    return anim


def main(
    n_steps: int = 100,
    d_model: int = 128,
    n_layers: int = 2,
    batch_size: int = 128,
    n_epochs: int = 400,
    sample_size: int = 512,
    steps_between_sampling: int = 20,
    seed: int = 42,
):
    print("Creating model")
    model = BasicDiscreteTimeModel(d_model=d_model, n_layers=n_layers)
    ddpm = DDPM(n_steps=n_steps)

    print("Training")
    result = train(
        model=model,
        ddpm=ddpm,
        batch_size=batch_size,
        n_epochs=n_epochs,
        sample_size=sample_size,
        steps_between_sampling=steps_between_sampling,
        seed=seed,
    )

    print("Animating")
    animate(result.samples, save=True)
    print("Animation complete")

    print("Plotting final result")
    final_samples = result.samples[-1].detach().numpy()  # Get the last set of samples
    plt.figure(figsize=(8, 5))
    plt.scatter(final_samples[:, 0], final_samples[:, 1], alpha=0.5, c="blue")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Final Generated Samples")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    import sys

    if any(arg.startswith("--f=") for arg in sys.argv):
        # Running in Jupyter, avoid executing Fire
        print("Running in Jupyter Notebook, skipping Fire CLI.")
    else:
        from fire import Fire

        Fire(main)
