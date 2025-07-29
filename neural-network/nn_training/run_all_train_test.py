import os
import sys
import subprocess
import time
import argparse
import gc

sys.path.append("../")
sys.path.append("../../")
sys.path.append("./source_code/")

import project_config as CONFIG

def run_command(command):
    """Run a command with subprocess, ensuring proper termination."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Command output: {result.stdout}")
        if result.stderr:
            print(f"Command error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command {command}: {e}")
        return False

def main(args):
    use_time_hour = False
    use_onehot = True
    upsample_enabled = False
    epochs = 3
    batch_size = 8
    num_labels = 1
    time_window = 10

    train_file_path = CONFIG.SOURCE_CODE_DIRECTORY + "nn_training/train_nn.py"
    results_file_path = CONFIG.SOURCE_CODE_DIRECTORY + "nn_training/generate_results.py"
    model_analysis_file_path = (
        CONFIG.SOURCE_CODE_DIRECTORY + "nn_model_analysis/compare_models.py"
    )

    nn_model_list = [
        # "dense", 
        "cnn",
        #  "lstm", "trans", 
        # "aen"
        ]
    architecture_list = [
        "one_model_with_correlation", "one_model_without_correlation", 
        # "multiple_models_with_correlation", "multiple_models_without_correlation"
        ]

    # Run 1
    if args.run_1:
        use_metadata = False
        metadata_path = CONFIG.OUTPUT_DIRECTORY
        metadata_metric = "NOT_USED"
        num_selected_nodes_for_correlation = 0
        run_number = 0
        run_final_report = True
        for nn_model in nn_model_list:
            for architecture in architecture_list:
                train_command = (
                    f"python3 {train_file_path} {nn_model} {architecture} {use_metadata} "
                    f"{metadata_path} {metadata_metric} {num_selected_nodes_for_correlation} "
                    f"{run_number} {use_time_hour} {use_onehot} {upsample_enabled} "
                    f"{epochs} {batch_size} {num_labels} {time_window}"
                )
                print(f"Run train command: {train_command}")
                run_command(train_command)
                print(f"[DONE] Run train command: {train_command}")
                results_command = (
                    f"python3 {results_file_path} {nn_model} {architecture} {metadata_metric} "
                    f"{run_number} {run_final_report} {use_time_hour} {use_onehot} "
                    f"{upsample_enabled} {num_labels} {time_window}"
                )
                print(f"Run result command: {results_command}")
                run_command(results_command)
                print(f"[DONE] Run result command: {results_command}")

                # Clear memory
                # psutil.virtual_memory()  # Refresh memory stats
                gc.collect()

    # Run 2
    # if args.run_2:
    #     architecture_list = ["multiple_models_with_correlation"]
    #     nn_model_list = ["lstm"]
    #     use_metadata = True
    #     metadata_path_list = [
    #         CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/metadata/distances.csv",
    #         CONFIG.OUTPUT_DIRECTORY + "pre_process/Output/metadata/correlations.csv",
    #         CONFIG.OUTPUT_DIRECTORY,
    #         CONFIG.OUTPUT_DIRECTORY,
    #     ]
    #     metadata_metric_list = [
    #         # "DISTANCE",
    #         # "CORRELATION",
    #         # "RANDOM", 
    #         "SHAP"
    #         ]
    #     num_random_trains = 10 
    #     num_selected_nodes_for_correlation = 5
    #     for metadata_index in range(len(metadata_metric_list)):
    #         metadata_path = metadata_path_list[metadata_index]
    #         metadata_metric = metadata_metric_list[metadata_index]
    #         for nn_model in nn_model_list:
    #             for architecture in architecture_list:
    #                 if metadata_metric != "RANDOM":
    #                     run_number = 0
    #                     run_final_report = True
    #                     train_command = (
    #                         f"python3 {train_file_path} {nn_model} {architecture} {use_metadata} "
    #                         f"{metadata_path} {metadata_metric} {num_selected_nodes_for_correlation} "
    #                         f"{run_number} {use_time_hour} {use_onehot} {upsample_enabled} "
    #                         f"{epochs} {batch_size} {num_labels} {time_window}"
    #                     )
    #                     print(f"Running: {train_command}")
    #                     run_command(train_command)
    #                     results_command = (
    #                         f"python3 {results_file_path} {nn_model} {architecture} {metadata_metric} "
    #                         f"{run_number} {run_final_report} {use_time_hour} {use_onehot} "
    #                         f"{upsample_enabled} {num_labels} {time_window}"
    #                     )
    #                     print(f"Running: {results_command}")
    #                     run_command(results_command)
    #                 else:
    #                     run_final_report = False
    #                     for run_number in range(num_random_trains):
    #                         train_command = (
    #                             f"python3 {train_file_path} {nn_model} {architecture} {use_metadata} "
    #                             f"{metadata_path} {metadata_metric} {num_selected_nodes_for_correlation} "
    #                             f"{run_number} {use_time_hour} {use_onehot} {upsample_enabled} "
    #                             f"{epochs} {batch_size} {num_labels} {time_window}"
    #                         )
    #                         print(f"Running: {train_command}")
    #                         run_command(train_command)
    #                         results_command = (
    #                             f"python3 {results_file_path} {nn_model} {architecture} {metadata_metric} "
    #                             f"{run_number} {run_final_report} {use_time_hour} {use_onehot} "
    #                             f"{upsample_enabled} {num_labels} {time_window}"
    #                         )
    #                         print(f"Running: {results_command}")
    #                         run_command(results_command)
    #                     run_final_report = True
    #                     run_number = -1
    #                     results_command = (
    #                         f"python3 {results_file_path} {nn_model} {architecture} {metadata_metric} "
    #                         f"{run_number} {run_final_report} {use_time_hour} {use_onehot} "
    #                         f"{upsample_enabled} {num_labels} {time_window}"
    #                     )
    #                     print(f"Running: {results_command}")
    #                     run_command(results_command)
    #                 gc.collect()

    # Run model analysis
    # if args.run_analysis:
    #     print(f"Running model analysis: {model_analysis_file_path}")
    #     run_command(f"python3 {model_analysis_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural network training and evaluation.")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with reduced models and runs.")
    # parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP metadata runs.")
    parser.add_argument("--run-1", action="store_true", default=True, help="Run the first set of experiments.")
    parser.add_argument("--run-2", action="store_true", default=True, help="Run the second set of experiments.")
    parser.add_argument("--run-analysis", action="store_true", default=True, help="Run model analysis.")
    args = parser.parse_args()
    main(args)