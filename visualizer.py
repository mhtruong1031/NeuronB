import numpy as np
import os, requests

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from vedo import Plotter, Volume, settings, Mesh, screenshot

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Fix veda issue
settings.use_depth_peeling = False

def mpl_plot_volume(volume: np.stack) -> None:
    verts, faces, normals, _ = measure.marching_cubes(volume, level=0)

    # Step 2: Plot with surface shading
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor('red')
    mesh.set_edgecolor('none')

    ax.add_collection3d(mesh)

    # Set limits to match volume shape
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Tumor Surface Mesh")
    plt.tight_layout()
    plt.show()
    
def plotly_plot_volume(volume: np.stack, flair: np.ndarray) -> None:
    coords = np.argwhere(volume > 0) # Grab all points where volume value > 0 (tumor+)
    brain_coords = np.argwhere(flair > 0)  # Everything non-zero in FLAIR

    # Normalize FLAIR intensities [0, 1]
    brain_intensity = flair[brain_coords[:, 0], brain_coords[:, 1], brain_coords[:, 2]]
    brain_intensity = (brain_intensity - brain_intensity.min()) / (brain_intensity.max() - brain_intensity.min())

    # Downsample brain for speed
    brain_coords = brain_coords[::30]
    brain_intensity = brain_intensity[::10]

    x_b, y_b, z_b = brain_coords[:, 0], brain_coords[:, 1], brain_coords[:, 2]
    x_t, y_t, z_t = coords[:, 0], coords[:, 1], coords[:, 2]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_b, y=y_b, z=z_b,
        mode='markers',
        marker=dict(size=1, color=brain_intensity, colorscale='gray', opacity=0.3),
        name='Brain'
    ))

    fig.add_trace(go.Scatter3d(
        x=x_t, y=y_t, z=z_t,
        mode='markers',
        marker=dict(size=2, color='red', opacity=0.05),
        name='Tumor'
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
    ), title="Tumor Segmentation with Brain Overlay")

    fig.show()

def plot_surface_vedo(pred_volume: np.ndarray, flair: np.ndarray):
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min())
    flair_norm = (flair_norm * 255).astype(np.uint8)

    # Tumor surface mesh
    verts, faces, _, _ = measure.marching_cubes(pred_volume, level=0)
    tumor_mesh = Mesh([verts, faces])
    tumor_mesh.color("red").alpha(0.6).lighting("plastic")

    # Brain volume (grayscale)
    brain_volume = Volume(flair_norm)
    brain_volume.mode('composite')
    brain_volume.alpha([0, 0.005, 0.01, 0.015, 0.02, 0.03])
    brain_volume.cmap("gray")

    # Render both
    plt = Plotter(bg="white", title="Brain + Tumor Mesh", axes=1)
    plt.show(brain_volume, tumor_mesh, viewup="z", interactive=True)

def render_from_angles(pred_volume, flair_volume, out_dir="NeuronB/public/"):
    os.makedirs(out_dir, exist_ok=True)

    # Normalize brain volume
    flair_norm = (flair_volume - flair_volume.min()) / (flair_volume.max() - flair_volume.min())
    flair_norm = (flair_norm * 255).astype(np.uint8)

    # Build mesh
    verts, faces, _, _ = measure.marching_cubes(pred_volume, level=0)
    tumor_mesh = Mesh([verts, faces]).color("red").alpha(0.7).lighting("plastic")

    # Build brain volume
    brain_volume = Volume(flair_norm).mode('max')
    brain_volume.alpha([0, 0.01, 0.02, 0.03, 0.04, 0.06])
    brain_volume.cmap("gray")

    # Set up plotter
    plt = Plotter(offscreen=True, size=(1024, 768), bg='white')
    plt.show(brain_volume, tumor_mesh, interactive=False)

    # Center of the volume
    center = np.array(pred_volume.shape) / 2

    # Define views
    views = {
        "front":     (center + [0, -500, 0]),
        "side":      (center + [500, 0, 0]),
        "bottom":    (center + [0, 0, -500]),
        "isometric": (center + [500, 500, 500]),
    }

    # Loop through views
    for name, pos in views.items():
        plt.camera.SetPosition(pos.tolist())
        plt.camera.SetFocalPoint(center.tolist())
        plt.reset_camera()
        screenshot(os.path.join(out_dir, f"{name}.png"), scale=1)
        print(f"Saved {name}.png")

    plt.close()
    