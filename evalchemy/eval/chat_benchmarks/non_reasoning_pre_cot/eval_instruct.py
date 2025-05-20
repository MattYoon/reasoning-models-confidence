import logging
from typing import Any, Dict, List, Optional
import os
import re

from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
import lm_eval.models
from lm_eval.models.vllm_causallms import VLLM

from eval.task import BaseBenchmark
from eval.utils import extract_confidence_10_bin, JUDGE_PROMPT


dataset_path = os.getenv("DS_PATH", "DKYoon/triviaqa_val_1k")
sample_size = int(os.getenv("DS_SIZE", 1000))

MAX_LENGTH = int(os.getenv("MAX_LENGTH", -1))

PROMPT = """First, reason through the question step by step to arrive at an answer.  
Then, thoroughly assess your confidence in that answer by evaluating your thinking process so far.  
Finally, classify your confidence into one of the following classes based on how likely your answer is to be correct:

- "Almost no chance" (0.0–0.1)  
- "Highly unlikely" (0.1–0.2)  
- "Chances are slight" (0.2–0.3)  
- "Unlikely" (0.3–0.4)  
- "Less than even" (0.4–0.5)  
- "Better than even" (0.5–0.6)  
- "Likely" (0.6–0.7)  
- "Very good chance" (0.7–0.8)  
- "Highly likely" (0.8–0.9)  
- "Almost certain" (0.9–1.0)

Each category reflects the probability that your answer is correct.

At the very end of your output, format your answer and confidence as  
**Answer**: $ANSWER
**Confidence**: $CLASS
where CLASS is one of the names (only the names without the probability ranges) of the classes above.\n\n\n\n
"""
ANSWER_PROMPT = '''Repeat your answer and confidence in the requested format: 
**Answer**: $ANSWER
**Confidence**: $CLASS
The answer should be concise. If the question was multiple choice, just output the correct option. If you didn't find an answer yet, give me your best guess. You must give an answer and its confidence. Do NOT give further explanations.'''


class Benchmark(BaseBenchmark):

    def __init__(
        self,
        data_file: str = "",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(logger)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = MAX_LENGTH
        self.judge = OpenAI()

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()

        # Prepare instances for model
        all_instances = []
        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]
        for idx, example in enumerate(examples):
            message = PROMPT + example["question"]
            messages = [
                {"role": "user", "content": message}
            ]

            templated_messages = model.apply_chat_template(messages)
            templated_messages += example["prompt"]
            templated_messages += "\n\n**Answer**:"

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": 64,
                            "temperature": 0.0,
                            "seed": self.seed,
                            "until": ["<|im_end|>"]
                        },
                    ),
                    idx,
                )
            )

        # Generate model responses
        outputs = self.compute(model, all_instances)

        all_instances_answer = []
        for idx, example in enumerate(examples):
            message = PROMPT + example["question"]
            assistant_message = example['prompt'] + \
                "\n\n**Answer**:" + outputs[idx]
            messages = [
                {"role": "user", "content": message},
                {"role": "assistant", "content": assistant_message},
                {"role": "user", "content": ANSWER_PROMPT},
            ]

            templated_messages = model.apply_chat_template(messages)

            all_instances_answer.append(
                Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": 64,
                            "temperature": 0.0,
                            "seed": self.seed,
                            "until": ["<|im_end|>"]
                        },
                    ),
                    idx,
                )
            )

        # Generate model responses
        outputs_answer = self.compute(model, all_instances_answer)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, output, output_answer in zip(examples, outputs, outputs_answer):

            example["model_think"] = output
            example["model_output"] = "\n\n**Answer**:" + output_answer
            example["model_confidence"] = extract_confidence_10_bin(
                output_answer)

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)

        # convert 0,1,2,3 to A,B,C,D
        for example in tqdm(examples, desc='judging'):
            grade_letter = self.prompt_judge(example)
            is_correct = grade_letter == "A"
            is_incorrect = grade_letter == "B"
            is_not_attempted = grade_letter == "C"

            example["grade_letter"] = grade_letter
            example["correct"] = 1 if is_correct else 0
            example['not_attempted'] = 1 if is_not_attempted else 0

        total = len(results["examples"])
        solved = sum(example["correct"] for example in results["examples"])
        not_attempted = sum(example["not_attempted"]
                            for example in results["examples"])

        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "num_not_attempted": not_attempted,
                "accuracy": solved / total,
                "prompt": PROMPT,
                "prompt2": ANSWER_PROMPT,
            }
        )

        return results

    def load_questions(self) -> List[Dict[str, str]]:
        print(f"Loading {dataset_path} dataset...")
        dataset = load_dataset(dataset_path)
        questions = dataset["validation"]
        if sample_size > 0:
            questions = questions.select(range(sample_size))
        df = questions.to_pandas()

        questions = df.to_dict(orient="records")
        return questions

    def prompt_judge(self, example):

        if type(example['answers']) == str:
            answer = example['answers'] + \
                " (Any of these answers are acceptable.)"
        else:
            if len(example['answers']) == 1:
                answer = example['answers'][0]
            else:
                answer = ", or ".join(example['answers'])
                answer += " (Any of these answers are acceptable.)"

        content = JUDGE_PROMPT.format(
            question=example["question"],
            target=answer,
            predicted_answer=example["model_output"],
        )

        completion = self.judge.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "user", "content": content}
            ],
            temperature=0.0,
        )
        response = completion.choices[0].message.content

        match = re.search(r"(A|B|C)", response)
        # Default to "NOT_ATTEMPTED" if no match
        return match.group(0) if match else "C"
