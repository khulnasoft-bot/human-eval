import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from azure import azure_complete
from prompts import gpt4_prompt, gpt_3_5_prompt, azure_prompt, claude_prompt
from claude import claude_complete

load_dotenv()

# Constants
HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}", "Content-Type": "application/json"}
HUMAN_EVAL = Path(__file__).parent / "data/HumanEval.jsonl"
OUT_FILE = Path(__file__).parent / "results/results-{}-{}.jsonl"
SPECIAL_TOKEN_REGEX = re.compile(r'<\|\S+\|>')
STOP_TOKEN_REGEX = re.compile(r'\n\S')

# Exponential backoff retry function
async def retry(sem, fn, max_retries=3):
    for i in range(max_retries):
        try:
            async with sem:
                return await fn()
        except Exception as e:
            wait_time = 0.5 * (2 ** i) + random.uniform(0, 0.1)
            print(f"Retry {i+1}/{max_retries} after {wait_time:.2f}s: {e}")
            await asyncio.sleep(wait_time)
    return await fn()  # Final attempt

# Get completion from LLM
async def get_completion(sem, prompt, num_tries=1, model='gpt-4', temperature=None):
    temperature = temperature or {1: 0.0, 10: 0.6, 100: 0.8}.get(num_tries, 0.7)
    
    async def fetch():
        if model in {'gpt-3.5-turbo', 'gpt-4'}:
            return await openai.ChatCompletion.acreate(messages=prompt, model=model, temperature=temperature, max_tokens=1000, n=num_tries)
        elif model == 'azure-gpt-3.5-turbo':
            return await azure_complete(prompt)
        elif 'claude' in model:
            return await claude_complete(prompt, model)
        else:
            return await openai.Completion.acreate(prompt=prompt, model=model, temperature=temperature, max_tokens=1000, n=num_tries)
    
    completion = await retry(sem, fetch)
    choices = completion['choices']
    return [choice.get('message', {}).get('content', choice.get('text', '')) for choice in choices]

# Read dataset
def iter_hval():
    with open(HUMAN_EVAL, "r") as f:
        return [json.loads(line) for line in f]

# Process prompts and get results
async def get_results(num_tries=10, model='gpt-4'):
    out_file = OUT_FILE.format(model, num_tries)
    sem = asyncio.Semaphore(10)
    results = []

    async def process_task(prompt, task_id):
        if model == 'gpt-3.5-turbo':
            full_prompt = gpt_3_5_prompt(prompt)
        elif model == 'gpt-4':
            full_prompt = gpt4_prompt(prompt)
        elif model == 'azure-gpt-3.5-turbo':
            full_prompt = azure_prompt(prompt)
        elif 'claude' in model:
            full_prompt = claude_prompt(prompt)
        else:
            full_prompt = prompt
        
        completions = await get_completion(sem, full_prompt, num_tries, model)
        return [{'task_id': task_id, 'completion': c} for c in completions]
    
    tasks = [process_task(item['prompt'], item['task_id']) for item in iter_hval()]
    for future in asyncio.as_completed(tasks):
        results.extend(await future)
    
    with open(out_file, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)
    
    remove_bloat(out_file)

# Cleanup function to remove unwanted tokens
def remove_bloat(in_jsonl):
    cleaned_results = []
    with open(in_jsonl, "r") as f:
        for line in f:
            out = json.loads(line)
            out['completion'] = SPECIAL_TOKEN_REGEX.split(out['completion'])[0]
            out['completion'] = STOP_TOKEN_REGEX.split(out['completion'])[0]
            cleaned_results.append(out)
    
    with open(in_jsonl, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in cleaned_results)

if __name__ == "__main__":
    num_tries = 1
    model = 'gpt-4'  # Change model as needed
    asyncio.run(get_results(num_tries, model))
    print(OUT_FILE.format(model, num_tries))
