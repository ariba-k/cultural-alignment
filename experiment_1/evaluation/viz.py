import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from .representation import classify_region, classify_cultural_cluster, classify_socioeconomic_status
from collections import defaultdict
import seaborn as sns

class ResultsPlotter:
    def __init__(self, experiment_results, results_path):
        self.model_name = experiment_results.model_name
        self.results_path = results_path
        self.similarities = experiment_results.avg_similarities
        self.topic_similarities = experiment_results.topic_similarities
        self.representativeness_measures = experiment_results.representativeness_measures

        self.country_mapping = {
            "India (Current national sample)": "India",
            "S. Africa": "South Africa",
            "United States": "United States of America",
            "Palest. ter.": "Palestine",
            "S. Korea": "South Korea",
            "Britain": "United Kingdom"
        }

        self.world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    def plot_similarities_worldmap(self):
        data = {self.country_mapping.get(country, country): value for country, value in self.similarities.items()}
        df = pd.DataFrame(list(data.items()), columns=['country', 'average_similarity'])

        df_geo = self.world_map.merge(df, how="left", left_on="name", right_on="country")

        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        df_geo.plot(column='average_similarity', ax=ax, legend=True, cmap='coolwarm',
                    legend_kwds={'label': "Similarity Score", 'orientation': "horizontal"},
                    missing_kwds={'color': 'lightgrey'})

        ax.set_title(f'Global Similarities Heatmap for {self.model_name}')
        plt.tight_layout()

        plt.savefig(self.results_path / f'global_similarities_{self.model_name}.png')
        plt.close()


    def plot_topic_similarities(self):
        num_topics = len(self.topic_similarities)
        fig_width = max(8, num_topics)
        fig_height = 6

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        mapped_topics_data = {
            topic: {self.country_mapping.get(country, country): value for country, value in countries.items()}
            for topic, countries in self.topic_similarities.items()}

        x = list(mapped_topics_data.keys())
        y = [sum(country_values.values()) / len(country_values) if country_values else 0 for country_values in
             mapped_topics_data.values()]

        ax.bar(x, y, width=0.4)
        ax.set_xlabel('Topic')
        ax.set_ylabel('Similarity')
        ax.set_title(f'Topic Similarities for {self.model_name}')
        ax.set_xticklabels(x, rotation=45, ha='right', fontsize=10)

        plt.tight_layout()

        plt.savefig(self.results_path / f'topic_similarities_{self.model_name}.png')
        plt.close()


    def plot_representativeness_measures(self):
        num_measures = len(self.representativeness_measures)
        fig, axes = plt.subplots(num_measures, 1, figsize=(10, 5 * num_measures))

        for i, (measure, values) in enumerate(self.representativeness_measures.items()):
            mapped_measure_data = {self.country_mapping.get(category, category): value for category, value in
                                   values.items()}
            x = list(mapped_measure_data.keys())
            y = list(mapped_measure_data.values())

            ax = axes[i] if num_measures > 1 else axes
            ax.bar(x, y, color='skyblue')

            ax.set_xlabel('Category')
            ax.set_ylabel('Value')
            ax.set_title(f'{measure}')
            ax.set_xticklabels(x, rotation=45, ha='right', fontsize=10)

        plt.suptitle(f"Representativeness Measures for {self.model_name}")
        plt.tight_layout()

        plt.savefig(self.results_path / f'representativeness_measures_{self.model_name}.png')
        plt.close()


    def plot_topic_similarities_worldmap(self):
        unique_topics = list(self.topic_similarities.keys())
        num_topics = len(unique_topics)

        num_rows = (num_topics + 1) // 2

        fig, axes = plt.subplots(num_rows, 2, figsize=(40, 20 * num_rows), squeeze=False)

        for i, topic in enumerate(unique_topics):
            row, col = divmod(i, 2)
            ax = axes[row, col]

            countries = self.topic_similarities[topic]

            data = [[self.country_mapping.get(country, country), similarity] for country, similarity in
                    countries.items()]
            df = pd.DataFrame(data, columns=['Country', 'Similarity'])

            merged_data = self.world_map.merge(df, left_on='name', right_on='Country', how='left')
            merged_data.plot(column='Similarity', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8',
                             missing_kwds={'color': 'lightgrey'})

            vmin = merged_data['Similarity'].min()
            vmax = merged_data['Similarity'].max()
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Similarity', rotation=270, labelpad=15)

            ax.set_title(f'{topic}')

        fig.suptitle(f'Topic Similarities Worldmap for {self.model_name}', fontsize=24)
        plt.tight_layout()

        if num_topics % 2 != 0:
            axes[-1, -1].set_axis_off()

        plt.savefig(self.results_path / f'topic_similarities_worldmap_{self.model_name}.png')
        plt.close()


    def plot_representativeness_worldmap(self):
        measures_to_plot = ["Region-based Representativeness", "Culture-based Representativeness"]
        num_measures = len(measures_to_plot)

        fig, axes = plt.subplots(1, num_measures, figsize=(20 * num_measures, 10))

        for i, measure in enumerate(measures_to_plot):
            ax = axes[i]

            values = self.representativeness_measures[measure]

            country_values = {}
            for category, similarity in values.items():
                for country in self.similarities.keys():
                    if (measure == "Region-based Representativeness" and classify_region(country) == category) or \
                            (measure == "Culture-based Representativeness" and classify_cultural_cluster(
                                country) == category):
                        country_values[self.country_mapping.get(country, country)] = similarity

            df = pd.DataFrame(country_values.items(), columns=['Country', 'Representativeness'])

            merged_data = self.world_map.merge(df, left_on='name', right_on='Country', how='left')

            merged_data.plot(column='Representativeness', cmap='coolwarm', linewidth=0.8, ax=ax, edgecolor='0.8',
                             missing_kwds={'color': 'lightgrey'})

            ax.set_aspect('equal')

            classification_func = classify_region if measure == "Region-based Representativeness" else classify_cultural_cluster
            regions = merged_data.geometry.apply(lambda x: classification_func(x.centroid.coords[0]))
            merged_data['Region'] = regions
            for region in merged_data['Region'].unique():
                region_data = merged_data[merged_data['Region'] == region]
                region_data.boundary.plot(ax=ax, color='black', linewidth=1)

            vmin = merged_data['Representativeness'].min()
            vmax = merged_data['Representativeness'].max()
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm._A = []
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Representativeness', rotation=270, labelpad=15)

            ax.set_title(f'{measure}')

        fig.suptitle(f'Representative Measures for {self.model_name}', fontsize=24)
        plt.tight_layout()

        plt.savefig(self.results_path / f'representative_worldmap_{self.model_name}.png')
        plt.close()

    def plot_topic_representativeness_matrix(self):
        representativeness_types = {
            'Region-based Representativeness': classify_region,
            'Culture-based Representativeness': classify_cultural_cluster,
            'Socio-Economic Representativeness': classify_socioeconomic_status
        }

        topics = sorted(self.topic_similarities.keys())

        for rep_type, classifier in representativeness_types.items():
            category_topic_data = defaultdict(lambda: defaultdict(list))

            for topic, countries in self.topic_similarities.items():
                for country, similarity in countries.items():
                    category = classifier(country)
                    category_topic_data[category][topic].append(similarity)

            avg_category_topic_data = defaultdict(lambda: defaultdict(float))
            for category in category_topic_data:
                for topic in category_topic_data[category]:
                    avg_category_topic_data[category][topic] = sum(category_topic_data[category][topic]) / len(
                        category_topic_data[category][topic])

            categories = sorted(avg_category_topic_data.keys())

            df = pd.DataFrame(index=topics, columns=categories)
            for category in avg_category_topic_data:
                for topic in avg_category_topic_data[category]:
                    df.loc[topic, category] = avg_category_topic_data[category][topic]

            df = df.fillna(0)

            plt.figure(figsize=(12, 10))
            sns.heatmap(df, annot=True, cmap='viridis', linewidths=.5)
            plt.title(f'Topic Similarity by {rep_type}')
            plt.ylabel('Topics')
            plt.xlabel(rep_type)
            plt.tight_layout()

            rep_type = rep_type.lower()
            rep_type = rep_type.replace(' ', '_').replace('-', '_')
            plt.savefig(self.results_path / f'topic_similarity_{rep_type}_{self.model_name}.png')
            plt.close()



