from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
# parser.add_argument('--config', type=str, default='sample_hello_world')
parser.add_argument('--config', type=str, default='tag_experiments')
args = parser.parse_args()

run_experiment(f"config_files/{args.config}.yml")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
