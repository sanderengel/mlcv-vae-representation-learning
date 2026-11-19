import argparse

from vae.experiment import run_experiment

BETAS = [0.5, 1.0, 4.0]

parser = argparse.ArgumentParser(description="Train all models.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runs")
args = parser.parse_args()

for beta in BETAS:
    run_experiment("vae", run_name=f"vae_beta{beta}", beta=beta, overwrite=args.overwrite)
run_experiment("ae", run_name="ae", overwrite=args.overwrite)
run_experiment("vqvae", run_name="vqvae", overwrite=args.overwrite)
