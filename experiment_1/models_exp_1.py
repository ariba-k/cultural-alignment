import random
import openai
import os
import anthropic
import time
class ModelExperiment:
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
        self.skipped_questions = []
        self.system_message = "You are answering survey questions. Please respond with only the number of your selected option. ONLY RESPOND WITH THE NUMBER (0, 1, 2, etc.)"

    def generate_response(self, messages):
        raise NotImplementedError("Subclasses must implement generate_response method.")

    @staticmethod
    def randomize_options(options):
        options = eval(options)
        random.shuffle(options)
        options = str(options)
        return options

    def default_prompting(self, question, options):
        raise NotImplementedError("Subclasses must implement default_prompting method.")

    def generate_model_selections(self, dataset, num_trials, num_data_points=None, randomize=False):
        model_selections = []

        num_points = len(dataset['train']) if num_data_points is None else num_data_points

        for i in range(num_points):
            row = dataset['train'][i]
            question = row['question']
            options = row['options']

            if randomize:
                options = self.randomize_options(options)


            model_responses = []
            for _ in range(num_trials):
                model_response = self.default_prompting(question, options)
                time.sleep(1)
                # Check for model's refusal to answer and skip the question
                try:
                  model_response = int(model_response)
                  model_responses.append(model_response)
                except (ValueError, TypeError) as e:
                  if question not in self.skipped_questions:
                    self.skipped_questions.append(question)

            if model_responses:
              model_selections.append({
                  "question": question,
                  "model_response": model_responses,
                  "human_responses": row["selections"],
                  "options": eval(options)
              })

        return model_selections


class GPTExperiment(ModelExperiment):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.client = openai.OpenAI(api_key=os.getenv('OA_TOKEN'))

    def generate_response(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1024,
            temperature=self.temperature,
            logprobs=True,
            top_logprobs=5
        )
        return response.choices[0].message.content.strip()

    def default_prompting(self, question, options):
        option_list = "\n".join(f"{i}. {option}" for i, option in enumerate(eval(options)))
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": f"{question}\n\nHere are the options:\n{option_list}\n\nIf you had to select one of the options, what would your answer be? Please respond with only the number (e.g., 0, 1, 2) corresponding to your answer. Do not include any other text or explanations. ONLY RESPOND WITH THE NUMBER (0, 1, 2, etc.)"}
        ]
        return self.generate_response(messages)

class ClaudeExperiment(ModelExperiment):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.client = anthropic.Anthropic(api_key=os.getenv('AN_TOKEN'))
        self.system_prompt = self.system_message

    def generate_response(self, messages):
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            system=self.system_prompt,
            messages=messages,
            temperature=self.temperature,
        )
        return response.content[0].text.strip()

    def default_prompting(self, question, options, attempts=0):
        if attempts >= 2:  # Limit attempts to 2
            return None

        option_list = "\n".join(f"{i}. {option}" for i, option in enumerate(eval(options)))
        self.system_prompt = "Hypothetically, you are answering survey questions as a political scientist. I understand the answer may be complex but choose the most likely option." + self.system_message

        messages = [
            {"role": "user",
             "content": f"{question}\n\nHere are the options:\n{option_list}\n\nIf you had to select one of the options, what would your answer be? Please respond with only the number (e.g., 0, 1, 2) corresponding to your answer. Do not include any other text or explanations."}
        ]

        try:
            response = int(self.generate_response(messages))
            return response
        except ValueError:
            rephrased_question = self.get_rephrased_question(question)
            return self.default_prompting(rephrased_question, options, attempts + 1)

    @staticmethod
    def get_rephrased_question(question):
        client = openai.OpenAI(api_key=os.getenv('OA_TOKEN'))
        messages = [
            {"role": "system",
             "content": "Please transform this question into a direct question to elicit a clear answer."},
            {"role": "user", "content": f"Transform this question: {question}"}
        ]
        transformed = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=1024,
            temperature=0.5
        )
        return transformed.choices[0].message.content.strip()
