from models_exp_1 import GPTExperiment, ClaudeExperiment
import argparse
from datasets import load_dataset
import warnings
from pathlib import Path
from evaluation.evaluate import EvaluateExperiment

warnings.filterwarnings('ignore')

MODEL_MAP = {
    'gpt_3_5': ("gpt-3.5-turbo", "GPT"),
    'gpt_4': ("gpt-4o", "GPT"),
    'claude_2': ("claude-2.1", "Claude"),
    'claude_3': ("claude-3-opus-20240229", "Claude")
}

def run_experiments(models=None, num_data_points=None, num_trials=5, temperature=0.0):
    if models is None:
        models = ['gpt_3_5', 'gpt_4', 'claude_2', 'claude_3']
    dataset = load_dataset("Anthropic/llm_global_opinions")
    topic_file_path = Path(__file__).resolve().parent / "evaluation" / "question_categories.json"
    results_path = Path(__file__).resolve().parent / "results"

    num_rows, num_cols = dataset['train'].shape
    if num_data_points is None:
        num_data_points = num_rows
    else:
        num_data_points = int(num_data_points)

    experiment_manager = EvaluateExperiment(dataset, num_data_points, num_trials, topic_file_path, results_path)

    results = {}
    for model_key in models:
        print("Model:", model_key)
        if model_key in MODEL_MAP:
            model_name, model_type = MODEL_MAP[model_key]
            if model_type == "GPT":
                experiment_class = GPTExperiment
            elif model_type == "Claude":
                experiment_class = ClaudeExperiment
            else:
                raise ValueError("Unsupported experiment type. Use 'GPT' or 'Claude'.")

            results[model_key] = experiment_manager.run_experiment(model_name, model_type, experiment_class, temperature)

            experiment_manager.print_results(results[model_key])
            experiment_manager.plot_results(results[model_key])


def main():
    parser = argparse.ArgumentParser(description='Run experiment 1')
    parser.add_argument('-m', '--models', type=str, nargs='+', default=list(MODEL_MAP.keys()),
                        help=f"List of models to run experiments on. Choices are: {', '.join(MODEL_MAP.keys())}. Runs all if none are specified.")
    parser.add_argument('-n', '--num_data_points', type=str, default=None,
                        help='Number of data points to test, default is the number of rows in the dataset')
    parser.add_argument('-t', '--num_trials', type=int, default=5, help='Number of trials to run')
    parser.add_argument('-temp', '--temperature', type=float, default=0, help='Temperature of model')

    args = parser.parse_args()
    run_experiments(models=args.models, num_data_points=args.num_data_points, num_trials=args.num_trials, temperature=args.temperature)


if __name__ == "__main__":
    # main()
    # For direct running without CLI
    run_experiments(models=['claude_3'])
