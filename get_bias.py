from typing import Dict, List, Union, Tuple
import time
import openai
import pandas as pd

class Generator:
    def __init__(self, data, target):
        df = data

        schema_info = "\n".join([f"{col}: {df[col].dtype}" for col in df.columns])
        self.key = ''
        sample_data = df.head(20).to_string(index=False)
        self.prompt = f"""
            Here is a sample of data with 20 rows and its schema:

            Schema:
            {schema_info} 

            Sample Data:
            {sample_data}
            
            Target column to be predicted:
            {target}
            
            Bias Factors:
            Consider these bias factors when analyzing the data for potential socially biased columns:
            - Race
            - Ethnicity
            - Gender
            - Sexuality
            - Religion
            - Wealth
            - Income

            Goal:
            Based on the schema, sample data, and these bias factors, identify columns that may introduce social bias. Use the column names, descriptions, and patterns in the sample data to recognize these columns.

            Output:
            Please return a list of max 5 column names that may be socially biased with a brief explanation of why it may be biased.
            "Respond in the following format:\n"
                "- Column Name: Explanation.\n"
            """

    def call_openai_api(self,
            engine: str,
            max_tokens,
            temperature: float,
            top_p: float,
            n: int,
            stop: List[str],
            is_chat=True
    ):
        prompt = self.prompt
        start_time = time.time()
        result = None
        while result is None:
            try:
                key = self.key
                if is_chat:
                    choices = []
                    if isinstance(prompt, str):
                        prompt = [prompt]
                    for prompt_item in prompt:
                        re = openai.ChatCompletion.create(
                            model=engine,
                            messages=[
                                {"role": "system",
                                 "content": "Given a dataset schema, sample data, and specified bias factors, identify columns that may introduce social bias based on the schema and provided examples. Respond with only the list of column names that may be biased, and explain why they are biased."},
                                {"role": "user", "content": prompt_item},
                            ],
                            api_key=key,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            n=n,
                            stop=stop,
                            logprobs=True
                        )
                        choices += re.choices
                    result = {"choices": choices}
                    print('Openai api inference time:', time.time() - start_time)
                    return result

                else:
                    choices = []
                    if isinstance(prompt, str):
                        prompt = [prompt]
                    for prompt_item in prompt:
                        re = openai.Completion.create(engine=engine,
                                                      prompt=prompt_item,
                                                      api_key=key,
                                                      max_tokens=max_tokens,
                                                      temperature=temperature,
                                                      top_p=top_p,
                                                      n=n,
                                                      stop=stop,
                                                      logprobs=1)
                        choices += re.choices
                    result = {"choices": choices}
                    print('Openai api inference time:', time.time() - start_time)
                    return result
            except openai.InvalidRequestError as e:
                # fixme: hardcoded, fix when refactoring
                if "This model's maximum context length is" in str(e):
                    print(e)
                    print("Set a place holder, and skip this example")
                    result = {"choices": [{"message": {"content": "PLACEHOLDER"}}]} if is_chat \
                        else {"choices": [{"text": "PLACEHOLDER"}]}
                    print('Openai api inference time:', time.time() - start_time)
                else:
                    print(e, 'Retry.')
                    time.sleep(3)
            except Exception as e:
                print(e, 'Retry.')
                time.sleep(3)




