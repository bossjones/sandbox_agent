from __future__ import annotations

import json


import langsmith



# Load the example inputs from q_a.json
with open("scripts/q_a.json") as file:
    example_inputs = [
        (item["question"], item["answer"])
        for item in json.load(file)
    ]

client = langsmith.Client()
dataset_name = "Climate Change Q&A"

# Storing inputs in a dataset lets us
# run chains and LLMs over a shared set of examples.
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="Questions and answers about climate change.",
)
for input_prompt, output_answer in example_inputs:
    client.create_example(
        inputs={"question": input_prompt},
        outputs={"answer": output_answer},
        metadata={"source": "Various"},
        dataset_id=dataset.id,
    )
