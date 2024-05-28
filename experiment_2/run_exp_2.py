import os
import pandas as pd
import argparse
from models_exp_2 import ClaudeExperiment, GPTExperiment
import re

def load_prompts_from_excel(file_path, dimensions=None):
    df = pd.read_excel(file_path)
    if dimensions:
        df = df[df['Dimension'].isin(dimensions)]
    return df['Revised Question (Round 2) '].tolist()


MODEL_MAP = {
    'gpt_3_5': ("gpt-3.5-turbo", "GPT"),
    'gpt_4': ("gpt-4-turbo", "GPT"),
    'claude_2': ("claude-2.1", "Claude"),
    'claude_3': ("claude-3-opus-20240229", "Claude")
}

DIMENSIONS = ['Individualism/Collectivism', 'Power Distance']


def save_results_to_text(results, experiment_type, model_name, dimension, num_round):
    # Replace slashes with underscores in dimension names
    safe_dimension = re.sub(r'[ /]', '_', dimension)

    # Ensure the directory structure
    dir_path = os.path.join('results', f'Round {num_round}' , experiment_type, model_name)
    os.makedirs(dir_path, exist_ok=True)

    # Save to a text file
    file_path = os.path.join(dir_path, f'{safe_dimension}.txt')
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")


def run_experiments(models=None, num_trials=1, dimensions=None, num_round=1):
    if models is None:
        models = ['gpt_3_5', 'gpt_4', 'claude_2', 'claude_3']
    if dimensions is None:
        dimensions = DIMENSIONS

    # Construct the file path based on the current script location
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'exp_2_dataset.xlsx')

    for model_key in models:
        if model_key not in MODEL_MAP:
            print(f"Invalid model choice: {model_key}. Choices are: 'gpt_3_5', 'gpt_4', 'claude_2', 'claude_3'.")
            continue

        model_name, experiment_type = MODEL_MAP[model_key]
        for dimension in dimensions:
            prompts = load_prompts_from_excel(file_path, [dimension])
            results = []
            if experiment_type == "GPT":
                experiment = GPTExperiment(model_name, dimension, prompts)
            elif experiment_type == "Claude":
                experiment = ClaudeExperiment(model_name, dimension, prompts)
            else:
                raise ValueError("Unsupported experiment type. Use 'GPT' or 'Claude'.")

            experiment.process_prompts(num_trials, results)

            # Save results to a text file
            save_results_to_text(results, experiment_type, model_name, dimension, num_round)

    if results:
        experiment.save_results_to_excel(results, file_path, num_round)


def main():
    parser = argparse.ArgumentParser(description="Run experiment 2.")
    parser.add_argument('-m', '--models', type=str, nargs='+', default=list(MODEL_MAP.keys()),
                        help=f"List of models to run experiments on. Choices are: {', '.join(MODEL_MAP.keys())}. Runs all if none are specified.")
    parser.add_argument('-t', '--num_trials', type=int, default=3, help='Number of trials to run')
    parser.add_argument('-d', '--dimensions', type=str, nargs='+', choices=DIMENSIONS,
                        help='List of dimensions to examine. Choices are: Individualism/Collectivism, Masculinity/Femininity, Power Distance.')
    parser.add_argument('-r', '--round', type=int, required=True, help='Experiment round number.')

    args = parser.parse_args()

    run_experiments(args.models, args.num_trials, args.dimensions)


if __name__ == "__main__":
    # main()

    # For direct running without CLI
    run_experiments(num_round=1)
