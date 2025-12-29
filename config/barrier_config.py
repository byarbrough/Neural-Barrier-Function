"""
Extends alpha-beta-CROWN ConfigHandler with barrier function arguments.

This module imports the upstream arguments module and adds barrier-specific
configuration options needed for neural barrier function verification.
"""
import setup_paths  # noqa: F401 - must be first to setup sys.path

# Import upstream arguments module
import arguments as upstream_arguments

# Get references to the upstream Config and Globals
Config = upstream_arguments.Config
Globals = upstream_arguments.Globals


def add_barrier_options(config_handler):
    """Add barrier function specific arguments to the config handler."""

    # Algorithm options
    h = ["alg_options"]
    config_handler.add_argument(
        "--train_method", type=str, default="verification-only",
        choices=["verification-only", "fine-tuning"],
        help='Training framework for barrier function.',
        hierarchy=h + ["train_method"])
    config_handler.add_argument(
        "--verification_method", type=str, default="bab",
        choices=["bab", "mip"],
        help="Verification method in training loop.",
        hierarchy=h + ["verification_method"])

    # Dynamics function options
    h = ["alg_options", "dynamics_fcn"]
    config_handler.add_argument(
        "--train_nn_dynamics", action="store_true",
        help="Train the NN dynamics.",
        hierarchy=h + ["train_nn_dynamics"])

    # Barrier function options
    h = ["alg_options", "barrier_fcn"]
    config_handler.add_argument(
        "--barrier_output_dim", type=int, default=5,
        help="Output dimension of NN barrier function.",
        hierarchy=h + ["barrier_output_dim"])

    # Dataset options
    h = ["alg_options", "barrier_fcn", "dataset"]
    config_handler.add_argument(
        "--collect_samples", action='store_true',
        help="Collect samples to train barrier function.",
        hierarchy=h + ["collect_samples"])
    config_handler.add_argument(
        "--num_samples_x0", type=int, default=5000,
        help="Number of samples from initial set X0.",
        hierarchy=h + ["num_samples_x0"])
    config_handler.add_argument(
        "--num_samples_xu", type=int, default=5000,
        help="Number of samples from unsafe set Xu.",
        hierarchy=h + ["num_samples_xu"])
    config_handler.add_argument(
        "--num_samples_x", type=int, default=5000,
        help="Number of samples from workspace X.",
        hierarchy=h + ["num_samples_x"])

    # Training options
    h = ["alg_options", "barrier_fcn", "train_options"]
    config_handler.add_argument(
        "--num_epochs", type=int, default=50,
        help="Number of training epochs for barrier function.",
        hierarchy=h + ["num_epochs"])
    config_handler.add_argument(
        "--l1_lambda", type=float, default=0.0,
        help="L1 regularization weight for barrier function training.",
        hierarchy=h + ["l1_lambda"])
    config_handler.add_argument(
        "--early_stopping_tol", type=float, default=1e-9,
        help="Tolerance for early stopping in barrier training.",
        hierarchy=h + ["early_stopping_tol"])
    config_handler.add_argument(
        "--update_A_freq", type=int, default=1,
        help="Frequency to update A matrix during training.",
        hierarchy=h + ["update_A_freq"])
    config_handler.add_argument(
        "--samples_pool_size", type=int, default=100000,
        help="Maximum number of samples to maintain in pool.",
        hierarchy=h + ["samples_pool_size"])
    config_handler.add_argument(
        "--B_batch_size", type=int, default=50,
        help="Batch size for barrier function training.",
        hierarchy=h + ["B_batch_size"])
    config_handler.add_argument(
        "--num_iter", type=int, default=20,
        help="Number of iterations for barrier training loop.",
        hierarchy=h + ["num_iter"])
    config_handler.add_argument(
        "--scaling_factor", type=float, default=1.0,
        help="Scaling factor for barrier function output.",
        hierarchy=h + ["scaling_factor"])
    config_handler.add_argument(
        "--noise_weight", type=float, default=0.0,
        help="Weight for noise in barrier training.",
        hierarchy=h + ["noise_weight"])
    config_handler.add_argument(
        "--train_timeout", type=float, default=3600,
        help="Timeout in seconds for barrier training.",
        hierarchy=h + ["train_timeout"])
    config_handler.add_argument(
        "--condition_tol", type=float, default=1e-5,
        help="Tolerance for barrier condition verification.",
        hierarchy=h + ["condition_tol"])

    # Counterexample sampling options
    h = ["alg_options", "ce_sampling"]
    config_handler.add_argument(
        "--num_ce_samples", type=int, default=20,
        help="Number of counterexample samples to generate.",
        hierarchy=h + ["num_ce_samples"])
    config_handler.add_argument(
        "--num_ce_samples_accpm", type=int, default=5,
        help="Number of counterexample samples for ACCPM.",
        hierarchy=h + ["num_ce_samples_accpm"])
    config_handler.add_argument(
        "--opt_iter", type=int, default=100,
        help="Number of optimization iterations for CE sampling.",
        hierarchy=h + ["opt_iter"])
    config_handler.add_argument(
        "--radius", type=float, default=0.1,
        help="Sampling radius for counterexample generation.",
        hierarchy=h + ["radius"])

    # ACCPM options
    h = ["alg_options", "ACCPM"]
    config_handler.add_argument(
        "--max_iter", type=int, default=30,
        help="Maximum ACCPM iterations for fine-tuning.",
        hierarchy=h + ["max_iter"])
    config_handler.add_argument(
        "--cvxpy_solver", type=str, default='ECOS',
        help="CVXPY solver for ACCPM (ECOS, MOSEK, etc.).",
        hierarchy=h + ["cvxpy_solver"])
    config_handler.add_argument(
        "--num_ce_thresh", type=int, default=5,
        help="Counterexample threshold for ACCPM termination.",
        hierarchy=h + ["num_ce_thresh"])

    # BAB-specific options for barrier verification
    h = ["alg_options", "bab"]
    config_handler.add_argument(
        "--bab_yaml_path", type=str, default='double_integrator.yaml',
        help="Path to BAB configuration YAML file.",
        hierarchy=h + ["bab_yaml_path"])
    config_handler.add_argument(
        "--adv_sample_filter_radius_x0", type=float, default=0.05,
        help="Filter radius for adversarial samples from X0.",
        hierarchy=h + ["adv_sample_filter_radius_x0"])
    config_handler.add_argument(
        "--adv_sample_filter_radius_xu", type=float, default=0.05,
        help="Filter radius for adversarial samples from Xu.",
        hierarchy=h + ["adv_sample_filter_radius_xu"])
    config_handler.add_argument(
        "--adv_sample_filter_radius_x", type=float, default=0.05,
        help="Filter radius for adversarial samples from X.",
        hierarchy=h + ["adv_sample_filter_radius_x"])
    config_handler.add_argument(
        '--adv_samples_pool_size', type=int, default=50,
        help="Maximum number of adversarial samples stored.",
        hierarchy=h + ["adv_samples_pool_size"])
    config_handler.add_argument(
        "--get_upper_bound_samples", type=int, default=2048,
        help="Number of samples for upper bound estimation.",
        hierarchy=h + ["get_upper_bound_samples"])

    # MIP options
    h = ["alg_options", "mip"]
    config_handler.add_argument(
        "--time_limit", type=float, default=0.0,
        help="Time limit for MIP solver (0 = no limit).",
        hierarchy=h + ["time_limit"])
    config_handler.add_argument(
        "--MIPFocus", type=int, default=0,
        help="Gurobi MIPFocus parameter (0-3).",
        hierarchy=h + ["MIPFocus"])

    # Update default args after adding new arguments
    config_handler.default_args = vars(config_handler.defaults_parser.parse_args([]))


# Add barrier options to the Config singleton
add_barrier_options(Config)
