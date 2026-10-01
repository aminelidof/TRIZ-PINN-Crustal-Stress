import os
import time
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.set_device(0)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde

dde.config.set_random_seed(SEED)

OUTPUT_DIR = "./PINN_Phase1_Baseline_Results"
OS_DIR_FIGS = os.path.join(OUTPUT_DIR, "figures")
OS_DIR_DATA = os.path.join(OUTPUT_DIR, "data_exports")
OS_DIR_MODELS = os.path.join(OUTPUT_DIR, "saved_models")

for directory in [OS_DIR_FIGS, OS_DIR_DATA, OS_DIR_MODELS]:
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(OUTPUT_DIR, "execution.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

logging.info("=================== DÉBUT DU RUN PINN ===================")
device_str = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Périphérique de calcul : {device_str.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

C_VELOCITY = 3.5
GAMMA_DAMP = 0.1

X_MIN, X_MAX = 0.0, 10.0
T_MIN, T_MAX = 0.0, 2.0

NEURONS_PER_LAYER = 64
NUM_LAYERS = 5
ACTIVATION = "tanh"
LEARNING_RATE_ADAM = 1e-3
ITERATIONS_ADAM = 15000

data_files = [f"SYNTH_ALG_00{i}_BHZ.csv" for i in range(1, 6)]
data_list = []

for idx, fname in enumerate(data_files):
    if os.path.exists(fname):
        logging.info(f"Chargement de la station : {fname}")
        df = pd.read_csv(fname)
        data_list.append(df)
    else:
        logging.warning(f"Fichier {fname} introuvable. Génération de secours pour station x = {idx * 2.0} km.")
        t_dummy = np.linspace(T_MIN, T_MAX, 200)
        x_dummy = np.full_like(t_dummy, idx * 2.0)
        u_dummy = np.sin(2 * np.pi * (t_dummy - x_dummy / C_VELOCITY)) * np.exp(-0.5 * GAMMA_DAMP * t_dummy)
        df_dummy = pd.DataFrame({"t": t_dummy, "x": x_dummy, "u": u_dummy})
        data_list.append(df_dummy)

df_all = pd.concat(data_list, ignore_index=True)
X_data = df_all[["x", "t"]].values.astype(np.float32)
y_data = df_all[["u"]].values.astype(np.float32)

observe_u = dde.icbc.PointSetBC(X_data, y_data, component=0)

def pde_damped_wave(x, u):
    u_tt = dde.grad.hessian(u, x, i=1, j=1)
    u_t  = dde.grad.jacobian(u, x, i=0, j=1)
    u_xx = dde.grad.hessian(u, x, i=0, j=0)
    return u_tt + GAMMA_DAMP * u_t - (C_VELOCITY ** 2) * u_xx

geom = dde.geometry.Interval(X_MIN, X_MAX)
timedomain = dde.geometry.TimeDomain(T_MIN, T_MAX)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

data = dde.data.TimePDE(
    geomtime,
    pde_damped_wave,
    [observe_u],
    num_domain=3000,
    num_boundary=400,
    num_initial=400
)

net = dde.nn.FNN(
    [2] + [NEURONS_PER_LAYER] * NUM_LAYERS + [1],
    ACTIVATION,
    "Glorot normal"
)
model = dde.Model(data, net)

checkpoint_path = os.path.join(OS_DIR_MODELS, "model_checkpoint.pt")
checker = dde.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_better_only=True,
    period=500
)

logging.info("=== PHASE 1: ADAM OPTIMIZER ===")
model.compile("adam", lr=LEARNING_RATE_ADAM)
t0 = time.time()
losshistory_adam, train_state_adam = model.train(iterations=ITERATIONS_ADAM, callbacks=[checker])
t_adam = time.time() - t0

logging.info("=== PHASE 2: L-BFGS FINE-TUNING ===")
model.compile("L-BFGS")
t0 = time.time()
losshistory_lbfgs, train_state_lbfgs = model.train(callbacks=[checker])
t_lbfgs = time.time() - t0

t_total = t_adam + t_lbfgs
logging.info(f"Entraînement terminé en {t_total:.2f} s (Adam: {t_adam:.2f}s, L-BFGS: {t_lbfgs:.2f}s)")

final_weights_path = os.path.join(OS_DIR_MODELS, "final_pinn_weights.pt")
torch.save(net.state_dict(), final_weights_path)
logging.info(f"Poids du réseau enregistrés : {final_weights_path}")

N_RES = 200
x_test = np.linspace(X_MIN, X_MAX, N_RES)
t_test = np.linspace(T_MIN, T_MAX, N_RES)
X_grid, T_grid = np.meshgrid(x_test, t_test)
X_flat = np.hstack((X_grid.flatten()[:, None], T_grid.flatten()[:, None]))

u_pred_flat = model.predict(X_flat)
u_pred_grid = u_pred_flat.reshape(N_RES, N_RES)

f_res_flat = model.predict(X_flat, operator=pde_damped_wave)
f_res_grid = np.abs(f_res_flat).reshape(N_RES, N_RES)

np.save(os.path.join(OS_DIR_DATA, "u_pred_matrix.npy"), u_pred_grid)
np.save(os.path.join(OS_DIR_DATA, "pde_residual_matrix.npy"), f_res_grid)
np.savez(os.path.join(OS_DIR_DATA, "grid_coords.npz"), x=x_test, t=t_test)

df_export = pd.DataFrame({
    "x": X_flat[:, 0],
    "t": X_flat[:, 1],
    "u_pred": u_pred_flat.flatten(),
    "pde_residual": np.abs(f_res_flat).flatten()
})
df_export.to_csv(os.path.join(OS_DIR_DATA, "full_field_predictions.csv"), index=False)

y_pred_obs = model.predict(X_data)
rmse_obs = np.sqrt(np.mean((y_pred_obs - y_data) ** 2))
mae_obs = np.mean(np.abs(y_pred_obs - y_data))
max_err_obs = np.max(np.abs(y_pred_obs - y_data))

metrics = {
    "Execution_Environment": {
        "Device": device_str,
        "PyTorch_Version": torch.__version__,
        "Seed": SEED
    },
    "Hyperparameters": {
        "Activation": ACTIVATION,
        "Layers": NUM_LAYERS,
        "Neurons_Per_Layer": NEURONS_PER_LAYER,
        "Adam_Iterations": ITERATIONS_ADAM,
        "c_velocity": C_VELOCITY,
        "gamma_damp": GAMMA_DAMP
    },
    "Performance": {
        "Adam_Time_sec": round(t_adam, 2),
        "LBFGS_Time_sec": round(t_lbfgs, 2),
        "Total_Time_sec": round(t_total, 2),
        "Data_RMSE": float(rmse_obs),
        "Data_MAE": float(mae_obs),
        "Data_MaxError": float(max_err_obs),
        "PDE_Residual_Mean": float(np.mean(f_res_grid)),
        "PDE_Residual_Max": float(np.max(f_res_grid))
    }
}

with open(os.path.join(OS_DIR_DATA, "run_metadata_summary.json"), "w") as f:
    json.dump(metrics, f, indent=4)

loss_history_combined = np.array(losshistory_lbfgs.loss_train if losshistory_lbfgs.loss_train else losshistory_adam.loss_train)
steps_combined = np.array(losshistory_lbfgs.steps if losshistory_lbfgs.steps else losshistory_adam.steps)
df_loss = pd.DataFrame(loss_history_combined, columns=[f"loss_comp_{i}" for i in range(loss_history_combined.shape[1])])
df_loss.insert(0, "step", steps_combined)
df_loss.to_csv(os.path.join(OS_DIR_DATA, "convergence_history.csv"), index=False)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)

plt.figure(figsize=(6.5, 5))
im0 = plt.pcolormesh(
    X_grid, T_grid, u_pred_grid, cmap="seismic", shading="auto"
)
plt.scatter(
    X_data[::15, 0],
    X_data[::15, 1],
    c="black",
    s=14,
    marker="x",
    linewidths=0.9,
    label="Sensors",
    zorder=3,
)
plt.xlabel("Distance $x$ (km)")
plt.ylabel("Time $t$ (s)")
cbar0 = plt.colorbar(im0, label="Displacement $u(x,t)$")
plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()

plt.savefig(
    os.path.join(OS_DIR_FIGS, "fig_predicted_wavefield.png"), dpi=600
)
plt.savefig(
    os.path.join(OS_DIR_FIGS, "fig_predicted_wavefield.pdf")
)
plt.close()

plt.figure(figsize=(6.5, 5))
im1 = plt.pcolormesh(X_grid, T_grid, f_res_grid, cmap="inferno", shading="auto")
plt.xlabel("Distance $x$ (km)")
plt.ylabel("Time $t$ (s)")
cbar1 = plt.colorbar(im1, label=r"Absolute Residual $|\mathcal{F}(u)|$")
plt.tight_layout()

plt.savefig(os.path.join(OS_DIR_FIGS, "fig_pde_residual.png"), dpi=600)
plt.savefig(os.path.join(OS_DIR_FIGS, "fig_pde_residual.pdf"))
plt.close()

plt.figure(figsize=(6.5, 4.8))
plt.semilogy(
    df_loss["step"],
    np.sum(df_loss.iloc[:, 1:], axis=1),
    "b-",
    lw=1.8,
    label="Total Loss",
)
plt.axvline(
    x=ITERATIONS_ADAM,
    color="r",
    linestyle="--",
    lw=1.4,
    label="Switch to L-BFGS",
)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True, which="both", ls=":", alpha=0.6)
plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()

plt.savefig(os.path.join(OS_DIR_FIGS, "fig_convergence_history.png"), dpi=600)
plt.savefig(os.path.join(OS_DIR_FIGS, "fig_convergence_history.pdf"))
plt.close()

idx_x1 = int(N_RES * 0.4)
idx_x2 = int(N_RES * 0.8)

plt.figure(figsize=(6.5, 4.8))
plt.plot(
    t_test,
    u_pred_grid[:, idx_x1],
    "b-",
    lw=1.8,
    label=f"$x = {x_test[idx_x1]:.1f}$ km",
)
plt.plot(
    t_test,
    u_pred_grid[:, idx_x2],
    "r--",
    lw=1.8,
    label=f"$x = {x_test[idx_x2]:.1f}$ km",
)
plt.xlabel("Time $t$ (s)")
plt.ylabel("Displacement $u$")
plt.grid(True, ls=":", alpha=0.6)
plt.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
plt.tight_layout()

plt.savefig(os.path.join(OS_DIR_FIGS, "fig_seismogram_traces.png"), dpi=600)
plt.savefig(os.path.join(OS_DIR_FIGS, "fig_seismogram_traces.pdf"))
plt.close()

logging.info(
    f"=== FIGURES INDIVIDUELLES GÉNÉRÉES (600 DPI & PDF) DANS : {OS_DIR_FIGS} ==="
)