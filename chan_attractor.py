#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 23:26:27 2024

@author: seba
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import nengo
import numpy as np
import imageio
import os

# Parameters for Chen's system
alpha, beta, delta = 5, -10, -0.38
tau                  = 0.1  # Synaptic time constant
num_trajectories     = 40   # Number of trajectories to simulate
simulation_time      = 5.0  # Fixed simulation duration in seconds
frames_per_rotation  = 200  # Number of frames for the GIF (controls smoothness)
time_steps_per_frame = None # Progress independently of frames

# Chen's system equations
def feedback(x):
    dx_dt = alpha * x[0] - x[1] * x[2]
    dy_dt = beta * x[1]  + x[0] * x[2]
    dz_dt = delta * x[2] + (x[0] * x[1]) / 3
    return [
        dx_dt * tau + x[0],
        dy_dt * tau + x[1],
        dz_dt * tau + x[2],
    ]

# Directory to store frames for the GIF
if not os.path.exists("frames"):
    os.makedirs("frames")

# Initialize storage for trajectories
trajectories = []

# Run multiple trajectories with different initial conditions
for i in range(num_trajectories):
    # Add small random noise to the initial state for each trajectory
    initial_state = np.random.normal(loc=[1.0, 1.0, 1.0], scale=0.1, size=3)
    
    # Nengo model setup
    model = nengo.Network(label=f"Chen Attractor Trajectory {i+1}")
    
    with model:
        # Neural ensemble with 2000 neurons and 3 dimensions to represent (x, y, z)
        state = nengo.Ensemble(2000, 3, radius=60)
        
        # Connect the state ensemble to itself with the feedback function
        nengo.Connection(state, state, function=feedback, synapse=tau)
        
        # Probe the state for recording
        state_probe = nengo.Probe(state, synapse=tau)

    # Run the simulation for the specified time
    with nengo.Simulator(model, seed=i) as sim:
        sim.run(simulation_time)
    
    trajectories.append(sim.data[state_probe])

# Determine total time steps and steps per frame
num_time_steps = len(trajectories[0])
steps_per_frame = num_time_steps // frames_per_rotation

# LaTeX formatted equations for the top-right corner
equation_text = (
    r"$\frac{{dx}}{{dt}} = \alpha x - yz$" "\n"
    r"$\frac{{dy}}{{dt}} = \beta y + xz$" "\n"
    r"$\frac{{dz}}{{dt}} = \delta z + \frac{{xy}}{3}$" "\n"
    r"$\alpha = 5, \; \beta = -10, \; \delta = -0.38$"
)

# Neuron information for the bottom-left corner
neuron_info = "2000 Neurons\nLeaky Integrate-and-Fire"

# Create frames for the rotating 3D plot with fixed progression
for frame in range(frames_per_rotation):
    fig = plt.figure(figsize=(16, 12), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    ax.set_facecolor("white")
    ax.grid(False)
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1])
    
    # Set the title with LaTeX formatting
    ax.set_title(r"$Chen\ Attractor$", fontsize=25, pad=20)

    # Set the camera view
    ax.view_init(elev=30, azim=frame * (360 / frames_per_rotation))

    # Current time step to display in this frame
    current_time_step = min((frame + 1) * steps_per_frame, num_time_steps)

    # Plot each trajectory up to the current time step
    for i, trajectory in enumerate(trajectories):
        color = plt.cm.Blues(i / num_trajectories)
        ax.plot(trajectory[:current_time_step, 0], 
                trajectory[:current_time_step, 1], 
                trajectory[:current_time_step, 2], 
                color=color, alpha=0.7, linewidth=1.5)
    
    # Add the equations to the top-right corner
    fig.text(0.75, 0.8, equation_text, ha="left", va="top", fontsize=14, color="black")

    # Add neuron information to the bottom-left corner
    fig.text(0.05, 0.1, neuron_info, ha="left", va="top", fontsize=14, color="black")

    # Save the frame
    plt.savefig(f"frames/frame_{frame:03d}.png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Create the GIF from the frames
with imageio.get_writer("chen_att.gif", mode="I", duration=0.1) as writer:
    for frame in range(frames_per_rotation):
        filename = f"frames/frame_{frame:03d}.png"
        image = imageio.imread(filename)
        writer.append_data(image)

# Clean up the frames
for frame in range(frames_per_rotation):
    os.remove(f"frames/frame_{frame:03d}.png")

print("GIF saved as chen_att.gif")
