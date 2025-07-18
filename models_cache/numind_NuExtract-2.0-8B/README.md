---
library_name: transformers
license: mit
base_model:
- Qwen/Qwen2.5-VL-8B-Instruct
pipeline_tag: image-text-to-text
---

<p align="center">
    <a href="https://nuextract.ai/">
        <img src="logo_nuextract.svg" width="200"/>
    </a>
</p>
<p align="center">
        🖥️ <a href="https://nuextract.ai/">API / Platform</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://numind.ai/blog">Blog</a>&nbsp&nbsp | &nbsp&nbsp🗣️ <a href="https://discord.gg/3tsEtJNCDe">Discord</a>
</p>

# NuExtract 2.0 8B by NuMind 🔥

NuExtract 2.0 is a family of models trained specifically for structured information extraction tasks. It supports both multimodal inputs and is multilingual.

We provide several versions of different sizes, all based on pre-trained models from the QwenVL family.
| Model Size | Model Name | Base Model | License | Huggingface Link |
|------------|------------|------------|---------|------------------|
| 2B | NuExtract-2.0-2B | [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | MIT | 🤗 [NuExtract-2.0-2B](https://huggingface.co/numind/NuExtract-2.0-2B) |
| 4B | NuExtract-2.0-4B | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Qwen Research License | 🤗 [NuExtract-2.0-4B](https://huggingface.co/numind/NuExtract-2.0-4B) |
| 8B | NuExtract-2.0-8B | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | MIT | 🤗 [NuExtract-2.0-8B](https://huggingface.co/numind/NuExtract-2.0-8B) |

❗️Note: `NuExtract-2.0-2B` is based on Qwen2-VL rather than Qwen2.5-VL because the smallest Qwen2.5-VL model (3B) has a more restrictive, non-commercial license. We therefore include `NuExtract-2.0-2B` as a small model option that can be used commercially.

## Benchmark
Performance on collection of ~1,000 diverse extraction examples containing both text and image inputs.
<a href="https://nuextract.ai/">
    <img src="nuextract2_bench.png" width="500"/>
</a>

## Overview

To use the model, provide an input text/image and a JSON template describing the information you need to extract. The template should be a JSON object, specifying field names and their expected type.

Support types include:
* `verbatim-string` - instructs the model to extract text that is present verbatim in the input.
* `string` - a generic string field that can incorporate paraphrasing/abstraction.
* `integer` - a whole number.
* `number` - a whole or decimal number.
* `date-time` - ISO formatted date.
* Array of any of the above types (e.g. `["string"]`)
* `enum` - a choice from set of possible answers (represented in template as an array of options, e.g. `["yes", "no", "maybe"]`).
* `multi-label` - an enum that can have multiple possible answers (represented in template as a double-wrapped array, e.g. `[["A", "B", "C"]]`).

If the model does not identify relevant information for a field, it will return `null` or `[]` (for arrays and multi-labels).

The following is an example template:
```json
{
  "first_name": "verbatim-string",
  "last_name": "verbatim-string",
  "description": "string",
  "age": "integer",
  "gpa": "number",
  "birth_date": "date-time",
  "nationality": ["France", "England", "Japan", "USA", "China"],
  "languages_spoken": [["English", "French", "Japanese", "Mandarin", "Spanish"]]
}
```
An example output:
```json
{
  "first_name": "Susan",
  "last_name": "Smith",
  "description": "A student studying computer science.",
  "age": 20,
  "gpa": 3.7,
  "birth_date": "2005-03-01",
  "nationality": "England",
  "languages_spoken": ["English", "French"]
}
```

⚠️ We recommend using NuExtract with a temperature at or very close to 0. Some inference frameworks, such as Ollama, use a default of 0.7 which is not well suited to many extraction tasks.

## Using NuExtract with 🤗 Transformers

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

model_name = "numind/NuExtract-2.0-2B"
# model_name = "numind/NuExtract-2.0-8B"

model = AutoModelForVision2Seq.from_pretrained(model_name, 
                                               trust_remote_code=True, 
                                               torch_dtype=torch.bfloat16,
                                               attn_implementation="flash_attention_2",
                                               device_map="auto")
processor = AutoProcessor.from_pretrained(model_name, 
                                          trust_remote_code=True, 
                                          padding_side='left',
                                          use_fast=True)

# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
```

You will need the following function to handle loading of image input data:
```python
def process_all_vision_info(messages, examples=None):
    """
    Process vision information from both messages and in-context examples, supporting batch processing.
    
    Args:
        messages: List of message dictionaries (single input) OR list of message lists (batch input)
        examples: Optional list of example dictionaries (single input) OR list of example lists (batch)
    
    Returns:
        A flat list of all images in the correct order:
        - For single input: example images followed by message images
        - For batch input: interleaved as (item1 examples, item1 input, item2 examples, item2 input, etc.)
        - Returns None if no images were found
    """
    from qwen_vl_utils import process_vision_info, fetch_image
    
    # Helper function to extract images from examples
    def extract_example_images(example_item):
        if not example_item:
            return []
            
        # Handle both list of examples and single example
        examples_to_process = example_item if isinstance(example_item, list) else [example_item]
        images = []
        
        for example in examples_to_process:
            if isinstance(example.get('input'), dict) and example['input'].get('type') == 'image':
                images.append(fetch_image(example['input']))
                
        return images
    
    # Normalize inputs to always be batched format
    is_batch = messages and isinstance(messages[0], list)
    messages_batch = messages if is_batch else [messages]
    is_batch_examples = examples and isinstance(examples, list) and (isinstance(examples[0], list) or examples[0] is None)
    examples_batch = examples if is_batch_examples else ([examples] if examples is not None else None)
    
    # Ensure examples batch matches messages batch if provided
    if examples and len(examples_batch) != len(messages_batch):
        if not is_batch and len(examples_batch) == 1:
            # Single example set for a single input is fine
            pass
        else:
            raise ValueError("Examples batch length must match messages batch length")
    
    # Process all inputs, maintaining correct order
    all_images = []
    for i, message_group in enumerate(messages_batch):
        # Get example images for this input
        if examples and i < len(examples_batch):
            input_example_images = extract_example_images(examples_batch[i])
            all_images.extend(input_example_images)
        
        # Get message images for this input
        input_message_images = process_vision_info(message_group)[0] or []
        all_images.extend(input_message_images)
    
    return all_images if all_images else None
```

E.g. To perform a basic extraction of names from a text document:
```python
template = """{"names": ["string"]}"""
document = "John went to the restaurant with Mary. James went to the cinema."

# prepare the user message content
messages = [{"role": "user", "content": document}]
text = processor.tokenizer.apply_chat_template(
    messages,
    template=template, # template is specified here
    tokenize=False,
    add_generation_prompt=True,
)

print(text)
""""<|im_start|>user
# Template:
{"names": ["string"]}
# Context:
John went to the restaurant with Mary. James went to the cinema.<|im_end|> 
<|im_start|>assistant"""

image_inputs = process_all_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# we choose greedy sampling here, which works well for most information extraction tasks
generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048}

# Inference: Generation of the output
generated_ids = model.generate(
    **inputs,
    **generation_config
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)
# ['{"names": ["John", "Mary", "James"]}']
```

<details>
<summary>In-Context Examples</summary>
Sometimes the model might not perform as well as we want because our task is challenging or involves some degree of ambiguity. Alternatively, we may want the model to follow some specific formatting, or just give it a bit more help. In cases like this it can be valuable to provide "in-context examples" to help NuExtract better understand the task.

To do so, we can provide a list examples (dictionaries of input/output pairs). In the example below, we show to the model that we want the extracted names to be in captial letters with `-` on either side (for the sake of illustration). Usually providing multiple examples will lead to better results.
```python
template = """{"names": ["string"]}"""
document = "John went to the restaurant with Mary. James went to the cinema."
examples = [
    {
        "input": "Stephen is the manager at Susan's store.",
        "output": """{"names": ["-STEPHEN-", "-SUSAN-"]}"""
    }
]

messages = [{"role": "user", "content": document}]
text = processor.tokenizer.apply_chat_template(
    messages,
    template=template,
    examples=examples, # examples provided here
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs = process_all_vision_info(messages, examples)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# we choose greedy sampling here, which works well for most information extraction tasks
generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048}

# Inference: Generation of the output
generated_ids = model.generate(
    **inputs,
    **generation_config
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
# ['{"names": ["-JOHN-", "-MARY-", "-JAMES-"]}']
```
</details>

<details>
<summary>Image Inputs</summary>
If we want to give image inputs to NuExtract, instead of text, we simply provide a dictionary specifying the desired image file as the message content, instead of a string. (e.g. `{"type": "image", "image": "file://image.jpg"}`).

You can also specify an image URL (e.g. `{"type": "image", "image": "http://path/to/your/image.jpg"}`) or base64 encoding (e.g. `{"type": "image", "image": "data:image;base64,/9j/..."}`).
```python
template = """{"store": "verbatim-string"}"""
document = {"type": "image", "image": "file://1.jpg"}

messages = [{"role": "user", "content": [document]}]
text = processor.tokenizer.apply_chat_template(
    messages,
    template=template,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs = process_all_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048}

# Inference: Generation of the output
generated_ids = model.generate(
    **inputs,
    **generation_config
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
# ['{"store": "Trader Joe\'s"}']
```
</details>

<details>
<summary>Batch Inference</summary>

```python
inputs = [
    # image input with no ICL examples
    {
        "document": {"type": "image", "image": "file://0.jpg"},
        "template": """{"store_name": "verbatim-string"}""",
    },
    # image input with 1 ICL example
    {
        "document": {"type": "image", "image": "file://0.jpg"},
        "template": """{"store_name": "verbatim-string"}""",
        "examples": [
            {
                "input": {"type": "image", "image": "file://1.jpg"},
                "output": """{"store_name": "Trader Joe's"}""",
            }
        ],
    },
    # text input with no ICL examples
    {
        "document": {"type": "text", "text": "John went to the restaurant with Mary. James went to the cinema."},
        "template": """{"names": ["string"]}""",
    },
    # text input with ICL example
    {
        "document": {"type": "text", "text": "John went to the restaurant with Mary. James went to the cinema."},
        "template": """{"names": ["string"]}""",
        "examples": [
            {
                "input": "Stephen is the manager at Susan's store.",
                "output": """{"names": ["STEPHEN", "SUSAN"]}"""
            }
        ],
    },
]

# messages should be a list of lists for batch processing
messages = [
    [
        {
            "role": "user",
            "content": [x['document']],
        }
    ]
    for x in inputs
]

# apply chat template to each example individually
texts = [
    processor.tokenizer.apply_chat_template(
        messages[i],  # Now this is a list containing one message
        template=x['template'],
        examples=x.get('examples', None),
        tokenize=False, 
        add_generation_prompt=True)
    for i, x in enumerate(inputs)
]

image_inputs = process_all_vision_info(messages, [x.get('examples') for x in inputs])
inputs = processor(
    text=texts,
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generation_config = {"do_sample": False, "num_beams": 1, "max_new_tokens": 2048}

# Batch Inference
generated_ids = model.generate(**inputs, **generation_config)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
for y in output_texts:
    print(y)
# {"store_name": "WAL-MART"}
# {"store_name": "Walmart"}
# {"names": ["John", "Mary", "James"]}
# {"names": ["JOHN", "MARY", "JAMES"]}
```
</details>

<details>
<summary>Template Generation</summary>
If you want to convert existing schema files you have in other formats (e.g. XML, YAML, etc.) or start from an example, NuExtract 2.0 models can automatically generate this for you.

E.g. convert XML into a NuExtract template:
```python
xml_template = """<SportResult>
    <Date></Date>
    <Sport></Sport>
    <Venue></Venue>
    <HomeTeam></HomeTeam>
    <AwayTeam></AwayTeam>
    <HomeScore></HomeScore>
    <AwayScore></AwayScore>
    <TopScorer></TopScorer>
</SportResult>"""

messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": xml_template}],
        }
    ]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)

image_inputs = process_all_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(
    **inputs,
    **generation_config
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
# {
#     "Date": "date-time",
#     "Sport": "verbatim-string",
#     "Venue": "verbatim-string",
#     "HomeTeam": "verbatim-string",
#     "AwayTeam": "verbatim-string",
#     "HomeScore": "integer",
#     "AwayScore": "integer",
#     "TopScorer": "verbatim-string"
# }
```

E.g. generate a template from natural language description:
```python
description = "I would like to extract important details from the contract."

messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": description}],
        }
    ]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True,
)

image_inputs = process_all_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

generated_ids = model.generate(
    **inputs,
    **generation_config
)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text[0])
# {
#     "Contract": {
#         "Title": "verbatim-string",
#         "Description": "verbatim-string",
#         "Terms": [
#             {
#                 "Term": "verbatim-string",
#                 "Description": "verbatim-string"
#             }
#         ],
#         "Date": "date-time",
#         "Signatory": "verbatim-string"
#     }
# }
```
</details>

## Fine-Tuning
You can find a fine-tuning tutorial notebook in the [cookbooks](https://github.com/numindai/nuextract/tree/main/cookbooks) folder of the [GitHub repo](https://github.com/numindai/nuextract/tree/main).

## vLLM Deployment
Run the command below to serve an OpenAI-compatible API:
```bash
vllm serve numind/NuExtract-2.0-8B --trust_remote_code --limit-mm-per-prompt image=6 --chat-template-content-format openai
```
If you encounter memory issues, set `--max-model-len` accordingly.

Send requests to the model as follows:
```python
import json
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

chat_response = client.chat.completions.create(
    model="numind/NuExtract-2.0-8B",
    temperature=0,
    messages=[
        {
            "role": "user", 
            "content": [{"type": "text", "text": "Yesterday I went shopping at Bunnings"}],
        },
    ],
    extra_body={
        "chat_template_kwargs": {
            "template": json.dumps(json.loads("""{\"store\": \"verbatim-string\"}"""), indent=4)
        },
    }
)
print("Chat response:", chat_response)
```
For image inputs, structure requests as shown below. Make sure to order the images in `"content"` as they appear in the prompt (i.e. any in-context examples before the main input).
```python
import base64

def encode_image(image_path):
    """
    Encode the image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("0.jpg")
base64_image2 = encode_image("1.jpg")

chat_response = client.chat.completions.create(
    model="numind/NuExtract-2.0-8B",
    temperature=0,
    messages=[
        {
            "role": "user", 
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}, # first ICL example image
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image2}"}}, # real input image
            ],
        },
    ],
    extra_body={
        "chat_template_kwargs": {
            "template": json.dumps(json.loads("""{\"store\": \"verbatim-string\"}"""), indent=4),
            "examples": [
                {
                    "input": "<image>",
                    "output": """{\"store\": \"Walmart\"}"""
                }
            ]
        },
    }
)
print("Chat response:", chat_response)
```