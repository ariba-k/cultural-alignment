from .topics import calculate_topic_level_similarity
from .representation import  calculate_representativeness
from .viz import ResultsPlotter
from .misc import calculate_similarity, parse_dict_from_string
from collections import defaultdict
import numpy as np
from tabulate import tabulate
import pickle
import os
from pathlib import Path


def evaluate_responses(model_selections, dataset):
    raw_similarities = defaultdict(list)
    question_probs = {}
    new_skipped_questions = []
    new_model_selections = []

    for row in dataset["train"]:
        question = row["question"]
        human_responses = parse_dict_from_string(row["selections"])

        for model_selection in model_selections:
            if model_selection["question"] == question:
                options = model_selection["options"]
                num_options = len(options)
                model_responses = model_selection["model_response"]
                model_probs = [model_responses.count(i) / len(model_responses) for i in range(num_options)]

                if sum(model_probs) == 0:
                    new_skipped_questions.append(question)
                    continue

                all_country_probs_zero = True
                for country, country_probs in human_responses.items():
                    if sum(country_probs) != 0:
                        all_country_probs_zero = False
                        break

                if all_country_probs_zero:
                    new_skipped_questions.append(question)
                    continue

                question_probs[question] = model_probs
                new_model_selections.append(model_selection)

                for country, country_probs in human_responses.items():
                    if sum(country_probs) == 0:
                        continue
                    similarity = calculate_similarity(model_probs, country_probs)
                    raw_similarities[country].append(similarity)

    avg_similarities = {
        country: np.mean(scores) for country, scores in raw_similarities.items() if scores
    }

    sorted_avg_similarities = {k: v for k, v in sorted(avg_similarities.items(), key=lambda item: item[1], reverse=True)}

    assert len(question_probs) == len(new_model_selections), "Mismatch between question_probs and new_model_selections sizes"
    assert len(question_probs) + len(new_skipped_questions) == len(model_selections), "Mismatch between question_probs + skipped_questions and original model_selections sizes"

    return sorted_avg_similarities, question_probs, new_skipped_questions, new_model_selections



class EvaluationMetrics:
    def __init__(self, model_name, model_type, model_selections, avg_similarities, question_probs, skipped_questions, representativeness_measures, topic_similarities):
        self.model_name = model_name
        self.model_type = model_type
        self.model_selections = model_selections
        self.avg_similarities = avg_similarities
        self.question_probs = question_probs
        self.skipped_questions = skipped_questions
        self.representativeness_measures = representativeness_measures
        self.topic_similarities = topic_similarities

class EvaluateExperiment:
    def __init__(self, dataset, num_data_points, num_trials, topic_file_path, results_path):
        self.results_model_path = None
        self.dataset = dataset
        self.num_data_points = num_data_points
        self.num_trials = num_trials
        self.topic_file_path = topic_file_path
        self.results_path = results_path
        self.experiments = []

    def run_experiment(self, model_name, model_type, experiment_class, temperature):
        self.results_model_path = self.results_path / model_type / model_name / Path(f"temperature_{temperature}")
        os.makedirs(self.results_model_path, exist_ok=True)

        experiment = experiment_class(model_name, temperature)
        model_selections = experiment.generate_model_selections(self.dataset, self.num_trials, self.num_data_points)
        avg_similarities, question_probs, new_skipped_questions, new_model_selections = evaluate_responses(model_selections, self.dataset)
        model_selections = new_model_selections
        skipped_questions = experiment.skipped_questions + new_skipped_questions
        representativeness_measures = calculate_representativeness(avg_similarities)
        topic_similarities = calculate_topic_level_similarity(question_probs, self.dataset, self.topic_file_path)
        experiment_results = EvaluationMetrics(
            model_name, model_type, model_selections, avg_similarities, question_probs, skipped_questions, representativeness_measures,
            topic_similarities
        )
        self.experiments.append(experiment_results)

        return experiment_results

    def print_results(self, experiment_results):
        if self.results_model_path is None:
            self.results_model_path = Path(self.results_path) / experiment_results.model_type / experiment_results.model_name

        results_pkl_file_path = self.results_model_path / f"eval_metrics_{experiment_results.model_name}.pkl"

        with open(results_pkl_file_path, 'wb') as output:
            pickle.dump(experiment_results, output)

        results_txt_file_path = self.results_model_path / f"results_{experiment_results.model_name}.txt"

        with open(results_txt_file_path, 'w') as file:
            file.write(
                "############################################################################################\n")
            file.write(f"Results for {experiment_results.model_name}:\n")
            file.write(
                "############################################################################################\n\n")

            # Skipped Questions
            file.write("Skipped Questions:\n")
            skipped_table = tabulate(
                [[len(experiment_results.skipped_questions), experiment_results.skipped_questions]],
                headers=['Count', 'Questions'], tablefmt='grid')
            file.write(skipped_table + "\n\n")

            # Average Similarities
            file.write("Average Similarities:\n")
            similarities_data = [[k, v] for k, v in experiment_results.avg_similarities.items()]
            similarities_table = tabulate(similarities_data, headers=['Country', 'Similarity'], tablefmt='grid')
            file.write(similarities_table + "\n\n")

            # Representativeness Measures
            file.write("Representativeness Measures:\n")
            for measure, values in experiment_results.representativeness_measures.items():
                file.write(f"{measure}:\n")
                measure_data = [[k, v] for k, v in values.items()]
                representativeness_table = tabulate(measure_data, headers=['Category', 'Average Similarity'],
                                                    tablefmt='grid')
                file.write(representativeness_table + "\n\n")

            # Topic Similarities
            file.write("Topic Similarities:\n")
            for topic, countries in experiment_results.topic_similarities.items():
                file.write(f"{topic}:\n")
                topic_data = [[country, sim] for country, sim in countries.items()]
                topic_table = tabulate(topic_data, headers=['Country', 'Similarity'], tablefmt='grid')
                file.write(topic_table + "\n\n")

            file.flush()

    def plot_results(self, experiment_results):
        results_plotter = ResultsPlotter(experiment_results, self.results_model_path)
        results_plotter.plot_similarities_worldmap()
        results_plotter.plot_representativeness_measures()
        results_plotter.plot_topic_similarities()
        results_plotter.plot_topic_representativeness_matrix()







