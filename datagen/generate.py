import argparse

from py2d.Py2D_solver import Py2D_solver
from py2d.datamanager import gen_path

from compute_stats import compute_stats


def main():
    parser = argparse.ArgumentParser(description="Generate 2D turbulence data using py2d")
    parser.add_argument("--Re", type=float, default=5000, help="Reynolds number")
    parser.add_argument("--fkx", type=int, default=0, help="Forcing wavenumber x")
    parser.add_argument("--fky", type=int, default=4, help="Forcing wavenumber y")
    parser.add_argument("--alpha", type=float, default=0.1, help="Rayleigh friction coefficient")
    parser.add_argument("--beta", type=float, default=20, help="Beta-plane parameter")
    parser.add_argument("--NX", type=int, default=256, help="Grid resolution")
    parser.add_argument("--dt", type=float, default=0.0002, help="Time step")
    parser.add_argument("--tSAVE", type=float, default=0.02, help="Save interval")
    parser.add_argument("--tTOTAL", type=float, default=500, help="Total simulation time")
    parser.add_argument("--ICnum", type=int, default=1, help="Initial condition number")
    args = parser.parse_args()

    Py2D_solver(
        Re=args.Re,
        fkx=args.fkx,
        fky=args.fky,
        alpha=args.alpha,
        beta=args.beta,
        NX=args.NX,
        forcing_filter=None,
        SGSModel_string="NoSGS",
        eddyViscosityCoeff=0.0,
        dt=args.dt,
        dealias=True,
        saveData=True,
        tSAVE=args.tSAVE,
        tTotal=args.tTOTAL,
        readTrue=False,
        ICnum=args.ICnum,
        resumeSim=False,
    )

    # Compute normalization stats after data generation
    save_dir, _, _ = gen_path(
        args.NX, args.dt, args.ICnum, args.Re,
        args.fkx, args.fky, args.alpha, args.beta,
        "NoSGS", True,
    )
    print(f"Computing normalization stats for {save_dir}...")
    compute_stats(save_dir)


if __name__ == "__main__":
    main()
