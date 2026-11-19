from vae.experiment import run_experiment

BETAS = [0.5, 1.0, 4.0]
OVERWRITE = False # Allow overwriting old runs


for beta in BETAS:
    run_experiment("vae", run_name=f"vae_beta{beta}", beta=beta, overwrite=OVERWRITE)
run_experiment("ae", run_name="ae", n_epochs=5, overwrite=OVERWRITE)
run_experiment("vqvae", run_name="vqvae", overwrite=OVERWRITE)
