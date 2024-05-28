from .misc import calculate_similarity, parse_dict_from_string
import os
from openai import OpenAI
from collections import defaultdict
import json
import numpy as np

global_opinions_topic_list = [
    "Social values and attitudes",
    "Religion and spirituality",
    "Science and technology",
    "Politics and policy",
    "Demographics",
    "Generations and age",
    "International affairs",
    "Internet and technology",
    "Gender and LGBTQ",
    "News habits and media",
    "Immigration and migration",
    "Family and relationships",
    "Race and ethnicity",
    "Economy and work",
    "Regions and countries",
    "Methodological research",
    "Security"
]

def categorize_question(question, topic_list):
    messages = [
      {"role": "system", "content": "Your task is to categorize questions using only the specified topics from a predefined list. Do not add or suggest any new topics. Simply select and return the exact topic label that best fits the question provided."},
      {"role": "user", "content": f"Categorize the given question by directly stating the topic label from the following list, and nothing else:\n\n{', '.join(topic_list)}\n\nQuestion: {question}\n\nChoose the topic label from the list that closest matches the question and return only that label."}
  ]


    client = OpenAI(api_key=os.environ.get('OA_TOKEN'))

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=1024,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def categorize_questions(dataset, topic_list, topic_file_path):
    if os.path.exists(topic_file_path):
        with open(topic_file_path, "r") as f:
            question_categories = json.load(f)
    else:
        question_categories = defaultdict(list)

    for row in dataset["train"]:
        question = row["question"]
        if not any(question in questions for questions in question_categories.values()):
          category = categorize_question(question, topic_list)
          if category not in topic_list:
            category = "Other"
          question_categories[category].append(question)

    with open(topic_file_path, "w") as f:
        json.dump(question_categories, f)

    return question_categories

def get_human_probs(dataset, question):
    for row in dataset["train"]:
        if row["question"] == question:
            return parse_dict_from_string(row["selections"])
    return {}


def calculate_topic_level_similarity(question_probs, dataset, topic_file_path):
    with open(topic_file_path, "r") as f:
        topic_questions = json.load(f)

    raw_topic_similarities = defaultdict(lambda: defaultdict(list))

    for question, model_probs in question_probs.items():
        human_responses = get_human_probs(dataset, question)
        for topic, questions in topic_questions.items():
            if question in questions:
                for country, country_probs in human_responses.items():
                    similarity = calculate_similarity(model_probs, country_probs)
                    raw_topic_similarities[topic][country].append(similarity)

    avg_topic_similarities = {
        topic: {
            country: np.mean(sims) for country, sims in country_sims.items()
        } for topic, country_sims in raw_topic_similarities.items()
    }

    sorted_avg_topic_similarities = {
        topic: {
            k: v for k, v in sorted(country_sims.items(), key=lambda item: item[1], reverse=True)
        } for topic, country_sims in avg_topic_similarities.items()
    }

    return sorted_avg_topic_similarities


