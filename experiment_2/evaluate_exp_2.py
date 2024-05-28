import os
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, skew
import ast
from itertools import combinations
from tabulate import tabulate
import re
import argparse

MODEL_MAP = {
    'gpt_3_5': ("gpt-3.5-turbo", "GPT"),
    'gpt_4': ("gpt-4-turbo", "GPT"),
    'claude_2': ("claude-2.1", "Claude"),
    'claude_3': ("claude-3-opus-20240229", "Claude")
}

DIMENSIONS = ['Individualism/Collectivism', 'Masculinity/Femininity', 'Power Distance']


def load_results(model_name, experiment_type, dimension, num_round):
    # Replace slashes with underscores in dimension names
    safe_dimension = re.sub(r'[ /]', '_', dimension)

    file_path = os.path.join('results',  f"Round {num_round}", experiment_type, model_name, f'{safe_dimension}.txt')
    print(file_path)
    if not os.path.exists(file_path):
        print(f"Results file for {model_name} and {dimension} not found.")
        return []

    results = []
    with open(file_path, 'r') as file:
        for line in file:
            # Safely evaluate the tuple string and extract the numerical response value
            try:
                response_tuple = ast.literal_eval(line.strip())
                response = float(response_tuple[1])
                results.append(response)
            except (ValueError, SyntaxError):
                print(f"Could not parse line: {line.strip()}")
    return results


def calculate_statistics(results):
    mean = sum(results) / len(results)
    stddev = (sum([(x - mean) ** 2 for x in results]) / len(results)) ** 0.5
    skewness = skew(results)
    return mean, stddev, skewness


def extreme_response_analysis(results, scale_max=7):
    midpoint = (scale_max + 1) / 2
    proportion_extreme_low = results.count(1) / len(results)
    proportion_extreme_high = results.count(scale_max) / len(results)
    mean_deviation = sum([abs(x - midpoint) for x in results]) / len(results)
    return proportion_extreme_low, proportion_extreme_high, mean_deviation


def compare_models(model_results, model_names, dimension, num_round):
    if len(model_results) < 2:
        print("At least two models are needed for comparison.")
        return

    safe_dimension = dimension.replace('/', '_')

    # Create side-by-side histograms
    fig, axes = plt.subplots(1, len(model_results), sharey=True, figsize=(15, 5))
    for ax, results, name in zip(axes, model_results, model_names):
        ax.hist(results, bins=7, edgecolor='black', range=(1, 7))
        ax.set_title(f'{name}')
        ax.set_xlabel('Response')
    axes[0].set_ylabel('Frequency')
    fig.suptitle(f'Combined Histograms for {dimension}')

    results_path = os.path.join('results', f'Round {num_round}' )

    combined_file_path = os.path.join(results_path, f'{safe_dimension}_combined_histogram_side_by_side.png')
    plt.savefig(combined_file_path)
    plt.close()

    # Create overlapping histogram
    plt.figure(figsize=(10, 6))
    for results, name in zip(model_results, model_names):
        plt.hist(results, bins=7, alpha=0.5, edgecolor='black', range=(1, 7), label=name)
    plt.title(f'Overlapping Histograms for {dimension}')
    plt.xlabel('Response')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    overlapping_file_path = os.path.join(results_path, f'{safe_dimension}_combined_histogram_overlapping.png')
    plt.savefig(overlapping_file_path)
    plt.close()

    # Perform pairwise t-tests
    t_test_results = []
    statistics_results = []
    pairs = list(combinations(range(len(model_results)), 2))
    print(f"Number of pairs to compare: {len(pairs)}")

    for (i, j) in pairs:
        t_stat, p_val = ttest_ind(model_results[i], model_results[j])
        t_test_results.append([model_names[i], model_names[j], t_stat, p_val])
        print(f'Two-sample t-test between {model_names[i]} and {model_names[j]}:')
        print(f'T-statistic: {t_stat}, P-value: {p_val}')
        print("***********************************************************************************")
        print()

    # Calculate statistics and extreme response analysis for each model
    for results, name in zip(model_results, model_names):
        mean, stddev, skewness = calculate_statistics(results)
        prop_low, prop_high, mean_dev = extreme_response_analysis(results)
        statistics_results.append([name, mean, stddev, skewness, prop_low, prop_high, mean_dev])

    # Save t-test results to text file
    t_test_file_path = os.path.join(results_path, f'{safe_dimension}_t_test_results.txt')
    with open(t_test_file_path, 'w') as f:
        f.write(tabulate(t_test_results, headers=['Model 1', 'Model 2', 'T-statistic', 'P-value'], tablefmt='grid'))

    # Save statistics results to text file
    stats_file_path = os.path.join(results_path, f'{safe_dimension}_statistics_results.txt')
    with open(stats_file_path, 'w') as f:
        f.write(tabulate(statistics_results,
                         headers=['Model', 'Mean', 'StdDev', 'Skewness', 'Prop Low (1)', 'Prop High (7)',
                                  'Mean Deviation'], tablefmt='grid', missingval='N/A'))


def analyze_results(models=None, dimensions=None, num_round=1):
    if models is None:
        models = ['gpt_3_5', 'gpt_4', 'claude_2', 'claude_3']
    if dimensions is None:
        dimensions = DIMENSIONS

    experiment_types = [MODEL_MAP[model][1] for model in models]
    model_names = [MODEL_MAP[model][0] for model in models]

    for dimension in dimensions:
        model_results = []
        for model_name, experiment_type in zip(model_names, experiment_types):
            results = load_results(model_name, experiment_type, dimension, num_round)
            if results:
                model_results.append(results)

        if len(model_results) > 1:
            compare_models(model_results, model_names, dimension, num_round)

def main():
    parser = argparse.ArgumentParser(description='Analyze AI model responses across cultural dimensions.')
    parser.add_argument('-m', '--models', nargs='+', required=True, choices=['gpt_3_5', 'gpt_4', 'claude_2', 'claude_3'],
                        help='List of models to analyze.')
    parser.add_argument('-d', '--dimensions', nargs='+', required=True, choices=DIMENSIONS,
                        help='List of cultural dimensions to analyze.')
    parser.add_argument('-r', '--round', type=int, required=True, help='Experiment round number.')

    args = parser.parse_args()
    analyze_results(args.models, args.dimensions, args.round)


if __name__ == "__main__":
    '''
    Mean: The average response value on the Likert scale.
    StdDev (Standard Deviation): Indicates the variability or spread of the responses around the mean. 
                                 A higher standard deviation means more variability.
    Skewness: Measures the asymmetry of the distribution of responses. 
              Negative skewness indicates that responses are skewed towards the higher end (collectivist),
              while positive skewness indicates skewness towards the lower end (individualist).
    Prop Low (1): The proportion of responses at the lowest end of the Likert scale (1). 
                  Indicates how often the model produces strongly individualist responses.
    Prop High (7): The proportion of responses at the highest end of the Likert scale (7). 
                   Indicates how often the model produces strongly collectivist responses.
    Mean Deviation: The average deviation of responses from the midpoint of the Likert scale (4). 
                    Higher values indicate responses are more extreme (either towards 1 or 7).

    '''

    '''
    T-statistic: A value that indicates the size of the difference between the two means in terms of standard errors. 
                 A larger absolute value suggests a larger difference.
    P-value: The probability that the observed difference between the two means occurred by chance.
             A smaller p-value (typically less than 0.05) indicates that the difference is statistically significant
    '''
    # main()
    analyze_results(num_round=1)
