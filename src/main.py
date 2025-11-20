import os
import json
import random
import numpy as np
from tqdm import tqdm
import argparse

from helper import (
    FastLogger,
    compute_metrics,
    observed_m_ids,
    uav_position,
)
from orthomap import Field
from mapper_LBP import OccupancyMap as OM
from planner import planning

from uav_camera import Camera

# from new_camera import Camera  # Updated import for new camera model
from viewer import plot_metrics, plot_terrain, plot_terrain_2d

from helper import create_run_folder, make_param_tag
import matplotlib

matplotlib.use("Agg")
from conf_space import Pyramid3D, GridPoint


# -----------------------------------------------------------------------------
# Load Experiment Configuration from JSON File
# -----------------------------------------------------------------------------
def load_config(config_file):
    """Load experiment configuration from a JSON file and filter out comment keys."""
    with open(config_file, "r") as f:
        config = json.load(f)
    # Remove any keys starting with "_" (used for comments)
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    return config


# -----------------------------------------------------------------------------
# Build Global Folder Paths from Config
# -----------------------------------------------------------------------------
def load_global_paths(config):
    """
    Build global path variables using the base 'project_path' directory provided
    in the config.
    """
    PROJECT_PATH = config["project_path"].rstrip("/")  # Ensure no trailing slash
    ANNOTATION_PATH = os.path.join(PROJECT_PATH, "data", "annotation.txt")
    ORTHOMAP_PATH = "/media/bota/BOTA/wheat/example-run-001_20241014T1739_ortho_dsm.tif"
    TILE_PIXEL_PATH = os.path.join(PROJECT_PATH, "data", "tiles_to_pixels.txt")
    MODEL_PATH = os.path.join(
        PROJECT_PATH,
        "binary_classifier",
        "models",
        "best_model_auc91_lr1_-05_bs128_wd_2.5-04.pth",
    )
    CACHE_DIR = os.path.join(PROJECT_PATH, "data", "predictions_cache")
    return (
        PROJECT_PATH,
        ANNOTATION_PATH,
        ORTHOMAP_PATH,
        TILE_PIXEL_PATH,
        MODEL_PATH,
        CACHE_DIR,
    )


# -----------------------------------------------------------------------------
# Parse Command-Line Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run active sensing experiments using a configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main Experiment Code
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    config = load_config(args.config)

    # Extract configuration parameters
    (
        PROJECT_PATH,
        ANNOTATION_PATH,
        ORTHOMAP_PATH,
        TILE_PIXEL_PATH,
        MODEL_PATH,
        CACHE_DIR,
    ) = load_global_paths(config)
    # base_dir = create_run_folder(os.path.join(PROJECT_PATH, "results"))
    base_dir = os.path.join(PROJECT_PATH, "trials_new_conf_space")
    run_base = f"{config['field_type'].lower()}_{config['start_position']}"
    # if config.get("action_strategy") == "mcts" and config.get("params_in_path", True):
    # run_base = run_base + "__" + make_param_tag(config.get("mcts_params", {}))
    # # TODO remove this later when planning local change
    border_comp = False
    sample_obs = False
    most_visited = False
    run_base = run_base + "_" + config["action_strategy"].upper()
    # if border_comp:
    #     run_base = run_base + "_border_comp"
    # else:
    #     run_base = run_base + "_wout_border_comp"
    if sample_obs:
        run_base = run_base + "_sample_obs"
    else:
        run_base = run_base + "_expd_obs"
    if most_visited:
        run_base = run_base + "_most_visited"
    else:
        run_base = run_base + "_most_rewarded"

    results_folder = os.path.join(base_dir, run_base)
    print(f"Results will be saved to: {results_folder}")

    ENABLE_STEPWISE_PLOTTING = config["enable_plotting"]
    ENABLE_LOGGING = config["enable_logging"]
    mcts_params = config.get("mcts_params", {})

    field_type = config["field_type"]
    start_position = config["start_position"]
    action_strategy = config["action_strategy"]
    correlation_types = config["correlation_types"]
    n_steps = config["n_steps"]
    iters = config["iters"]

    if isinstance(iters, int):
        iters = [0, iters]
    error_margins = [None if e == "None" else e for e in config["error_margins"]]
    if action_strategy == "sweep":
        error_margins = [None]
        iters = [0, 1]

    # -----------------------------------------------------------------------------
    # Setup Grid and Field Parameters Based on Field Type
    # -----------------------------------------------------------------------------

    # desktop += f"results_{field_type.lower()}_{start_position}_trial"
    if field_type == "Ortomap":
        grf_r = "orto"
        min_alt = 19.5
        overlap = 0.8
        optimal_alt = min_alt

        class grid_info:
            x = 60
            y = 110
            length = 1
            shape = (int(y / length), int(x / length))
            center = True

        use_sensor_model = False
    else:
        grf_r = 4
        field_type = grf_r
        min_alt = None
        overlap = None
        optimal_alt = 21.5

        class grid_info:
            # x = 50
            # y = 50
            x = 60
            y = 110
            length = 1
            # length = 0.125
            shape = (int(y / length), int(x / length))
            center = True

        use_sensor_model = True

    seed = 123
    rng = np.random.default_rng(seed)
    xy_step_in_cells = 5  # grid cells in xy per camera step
    # TODO: adjust pyramid3D parameters as needed
    configuration_space = Pyramid3D(
        (grid_info.x, grid_info.y),
        xy_step_in_cells,
        grid_info.length,
        np.deg2rad(60),
        center_on_origin=grid_info.center,
    )
    # TODO: conf space none to test original camera
    # configuration_space = None
    #    # Uncomment the following line to use the original Camera class
    camera1 = Camera(
        grid_info,
        60,
        rng=rng,
        camera_altitude=min_alt,
        f_overlap=overlap,
        s_overlap=overlap,
        conf_space=configuration_space,
    )
    map = Field(
        grid_info,
        field_type,
        sweep=action_strategy,
        h_range=camera1.get_hrange(),
        annotation_path=ANNOTATION_PATH,
        ortomap_path=ORTHOMAP_PATH,
        tile_pixel_path=TILE_PIXEL_PATH,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
    )

    # -----------------------------------------------------------------------------
    # Main Experiment Loop
    # -----------------------------------------------------------------------------

    for corr_type in tqdm(correlation_types, desc="Pairwise", position=0):
        for e_margin in tqdm(
            error_margins, desc=f"Error Margins (pairwise = {corr_type})", position=1
        ):
            for iter in tqdm(
                range(iters[0], iters[-1]),
                desc=f"Iters (e={e_margin})",
                position=2,
                leave=False,
            ):
                log_folder = os.path.join(results_folder, "txt")
                # log_folder = (
                #     results_folder
                #     # + f"/txt/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}"
                # )

                map.reset()
                ground_truth_map = map.get_ground_truth()
                # Initialize belief map with a uniform probability (0.5)
                belief_map = np.full((grid_info.shape[0], grid_info.shape[1], 2), 0.5)
                assert ground_truth_map.shape == belief_map[:, :, 0].shape

                if e_margin is not None:
                    conf_dict = map.init_s0_s1(
                        e=e_margin,
                        sensor=use_sensor_model,
                    )
                else:
                    conf_dict = None

                occupancy_map = OM(
                    grid_info.shape, conf_dict=conf_dict, correlation_type=corr_type
                )

                planner = planning(
                    grid_info,
                    camera1,
                    action_strategy,
                    conf_dict=conf_dict,
                    optimal_alt=optimal_alt,
                    # seed=seed,
                    mcts_params=mcts_params,
                    ground_truth_map=None,
                    local=False,
                    border_comp=border_comp,
                    sample_obs=sample_obs,
                    most_visited=most_visited,
                )
                # Select initial UAV starting position
                # TODO: update startpos for POINT
                if configuration_space is not None:
                    start_pos = configuration_space.sample_start_position(
                        position=start_position, rng=rng
                    )
                else:
                    if start_position == "edge":
                        # w = 2 * min_alt * np.tan(np.deg2rad(fov * 0.5))
                        real_border = [
                            (
                                -grid_info.x / 2,
                                random.uniform(-grid_info.y / 2, grid_info.y / 2),
                            ),  # Left border
                            (
                                grid_info.x / 2,
                                random.uniform(-grid_info.y / 2, grid_info.y / 2),
                            ),  # Right border
                            (
                                random.uniform(-grid_info.x / 2, grid_info.x / 2),
                                grid_info.y / 2,
                            ),  # Top border
                            (
                                random.uniform(-grid_info.x / 2, grid_info.x / 2),
                                -grid_info.y / 2,
                            ),  # Bottom border
                        ]
                        start_pos = random.choice(real_border)

                    elif start_position == "corner":
                        start_pos = random.choice(
                            [
                                (-grid_info.x / 2, -grid_info.y / 2),
                                (-grid_info.x / 2, grid_info.y / 2),
                                (grid_info.x / 2, -grid_info.y / 2),
                                (grid_info.x / 2, grid_info.y / 2),
                            ]
                        )

                # Initialize UAV position and list for tracking path and actions
                if type(start_pos) is GridPoint:
                    uav_pos = uav_position((start_pos.coord[0:2], start_pos.coord[2]))
                else:
                    uav_pos = uav_position((start_pos, camera1.get_hrange()[0]))

                uav_positions, actions = [uav_pos], []
                # Update camera settings based on UAV initial state
                camera1.set_altitude(uav_pos.altitude)
                camera1.set_position(uav_pos.position)
                # Initialize observed cell ids set and metric lists
                observed_ids = set()
                entropy, mse, height, coverage = [], [], [], []
                if ENABLE_LOGGING:

                    logger = FastLogger(
                        log_folder,
                        strategy=action_strategy,
                        pairwise=corr_type,
                        grid=grid_info,
                        init_x=uav_pos,
                        r=grf_r,
                        n_agent=iter,
                        e=e_margin,
                        conf_dict=conf_dict,
                        header_extras=[
                            ("mcts_params", json.dumps(mcts_params, sort_keys=True))
                        ],
                    )
                # Create directory for saving step-by-step results
                if ENABLE_STEPWISE_PLOTTING:
                    os.makedirs(
                        results_folder
                        + f"/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter}/steps/",
                        exist_ok=True,
                    )
                # also store mcts params as JSON alongside results (once per (corr, e) group)
                if action_strategy == "mcts":
                    params_out_dir = os.path.join(
                        results_folder,
                        f"{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}",
                    )
                    os.makedirs(params_out_dir, exist_ok=True)
                    try:
                        with open(
                            os.path.join(params_out_dir, "mcts_params.json"), "w"
                        ) as f:
                            json.dump(mcts_params, f, indent=2, sort_keys=True)
                    except Exception:
                        pass

                # -------------------------------------------------------------------------
                # Mapping and Planning Loop (per step)
                # -------------------------------------------------------------------------

                for step in tqdm(
                    range(0, n_steps),
                    desc=f"steps",
                    position=3,
                    leave=False,
                ):
                    # print(f"\n=== mapping {[step]} ===")
                    sigmas = None

                    if conf_dict is not None:
                        s0, s1 = conf_dict[np.round(uav_pos.altitude, decimals=2)]
                        sigmas = [s0, s1]

                    fp_vertices_ij, submap = map.get_observations(
                        uav_pos,
                        sigmas,
                    )

                    observed_field_range = camera1.get_range(
                        index_form=False,
                    )
                    # Update occupancy map with new observation and propagate messages
                    occupancy_map.update_belief_OG(fp_vertices_ij, submap, uav_pos)
                    occupancy_map.propagate_messages(
                        fp_vertices_ij, submap, max_iterations=1
                    )

                    # Update the belief map from the occupancy map's belief
                    belief_map[:, :, 1] = occupancy_map.get_belief().copy()
                    belief_map[:, :, 0] = 1 - belief_map[:, :, 1]

                    # Update observed cell IDs and compute metrics
                    observed_ids.update(observed_m_ids(camera1, uav_pos))
                    entropy_val, mse_val, coverage_val = compute_metrics(
                        ground_truth_map, belief_map, observed_ids, grid_info
                    )
                    entropy.append(entropy_val)
                    mse.append(mse_val)
                    coverage.append(coverage_val)
                    height.append(uav_pos.altitude)
                    if ENABLE_LOGGING and logger is not None:
                        # Log current metrics and actions
                        logger.log_data(
                            entropy[-1],
                            mse[-1],
                            height[-1],
                            coverage[-1],
                            step=step,
                            action=actions[-1] if len(actions) > 0 else None,
                            ig=(
                                info_gain_action[actions[-1]]
                                if len(actions) > 0
                                else None
                            ),
                        )
                        # logger.log("actions: " + str(actions))
                    if ENABLE_STEPWISE_PLOTTING:
                        # Save metrics plot for current iteration
                        plot_metrics(
                            f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/iter_{iter}.png",
                            entropy,
                            mse,
                            coverage,
                            height,
                        )
                    # Planning: select the next action based on current belief
                    next_action, info_gain_action = planner.select_action(
                        belief_map, uav_positions
                    )
                    print(f"Step {step}: Selected action {next_action}")
                    print(f"Current UAV position: {uav_pos}")
                    igs_sorted = dict(
                        sorted(
                            info_gain_action.items(), key=lambda kv: kv[1], reverse=True
                        )
                    )

                    for a, ig in igs_sorted.items():
                        print(f"{a}\t - {ig:.12f}")

                    print("________________________________")

                    # Update UAV position based on the next action
                    uav_pos = uav_position(camera1.x_future(next_action))
                    actions.append(next_action)
                    uav_positions.append(uav_pos)
                    # Update camera with the new UAV state
                    camera1.set_altitude(uav_pos.altitude)
                    camera1.set_position(uav_pos.position)
                    if ENABLE_STEPWISE_PLOTTING:
                        # Plot and save the terrain visualization for this step
                        plot_terrain(
                            f"{results_folder}/{corr_type}_{action_strategy}_e{e_margin}_r{grf_r}/{iter}/steps/step_{step}.png",
                            belief_map,
                            grid_info,
                            uav_positions[0:-1],
                            ground_truth_map,
                            submap,
                            observed_field_range,
                            fp_vertices_ij,
                            camera1.get_hrange(),
                        )

                    # plot_terrain_2d(
                    #     f"{results_folder}/steps/step_{step}.png",
                    #     grid_info,
                    #     ground_truth_map,
                    # )


if __name__ == "__main__":
    main()
