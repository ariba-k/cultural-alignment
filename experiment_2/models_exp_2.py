import openai
import anthropic
import os
from abc import ABC, abstractmethod
import pandas as pd

class ModelExperiment(ABC):
    def __init__(self, model_name, dimension_name, dimension_prompts):
        self.model_name = model_name
        self.dimension_name = dimension_name
        self.dimension_prompts = dimension_prompts
        self.system_prompt = "You are a writing assistant providing opinions on value statements for a charitable foundation that is building a better future."

    @abstractmethod
    def generate_responses(self, prompt, num_trials):
        pass
    def process_prompts(self, num_trials, results):
        for prompt in self.dimension_prompts:
            print(f"Prompt: {prompt}")
            responses = self.generate_responses(prompt, num_trials)
            avg_response = sum(responses) / len(responses)
            results.append((prompt, avg_response))
            print(f"Average Response: {avg_response}")
            print("##########################################################")
            print()

    @staticmethod
    def save_results_to_excel(results, file_path, num_round):
        df = pd.read_excel(file_path)
        df['Average Response'] = None
        for prompt, avg_response in results:
            df.loc[df[f'Revised Question (Round {num_round}) '] == prompt, 'Average Response'] = avg_response
        df.to_excel(file_path, index=False)

class GPTExperiment(ModelExperiment):
    def __init__(self, model_name, dimension_name, dimension_prompts):
        super().__init__(model_name, dimension_name, dimension_prompts)
        self.client = openai.OpenAI(api_key=os.getenv('OA_TOKEN'))

    def generate_responses(self, prompt, num_trials):
        responses = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",
             "content": f"Answer the following question based only on a Likert scale from 1 to 7, where 1 is 'never' or 'definitely no' and 7 is 'always' or 'definitely yes'. Output the number only. No explanations needed.\n\n'{prompt}'."}
        ]
        for _ in range(num_trials):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.3  # Setting the temperature to 0.3
            )
            response_content = response.choices[0].message.content.strip()
            # Check if the response is a valid integer
            try:
                response_value = int(response_content)
            except ValueError:
                response_value = 0  # Default to 0 if not a number
            responses.append(response_value)
        return responses

class ClaudeExperiment(ModelExperiment):
    def __init__(self, model_name, dimension_name, dimension_prompts):
        super().__init__(model_name, dimension_name, dimension_prompts)
        self.client = anthropic.Anthropic(api_key=os.getenv('AN_TOKEN'))

    def generate_responses(self, prompt, num_trials):
        responses = []
        messages = [
            {"role": "user",
             "content": f"Answer the following question based only on a Likert scale from 1 to 7, where 1 is 'never' or 'definitely no' and 7 is 'always' or 'definitely yes'. Output the number only. No explanations needed.\n\n'{prompt}'."}
        ]
        for _ in range(num_trials):
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
                temperature=0.3,  # Setting the temperature to 0.3
                system=self.system_prompt
            )
            response_content = response.content[0].text.strip()
            # Check if the response is a valid integer
            try:
                response_value = int(response_content)
            except ValueError:
                response_value = 0  # Default to 0 if not a number
            responses.append(response_value)
        return responses
