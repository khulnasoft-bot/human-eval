import aiohttp
import asyncio
from aiohttp import ClientSession
import os
from dotenv import load_dotenv

load_dotenv()


async def azure_complete(prompt: str):
    # Load environment variables
    azure_openai_base = os.environ.get('azureOpenAiBase')
    azure_openai_deployment = os.environ.get('azureOpenAiDeployment')
    azure_openai_key = os.environ.get('azureOpenAiKey')

    # Check for missing environment variables
    if not all([azure_openai_base, azure_openai_deployment, azure_openai_key]):
        raise ValueError("Missing one or more environment variables: 'azureOpenAiBase', 'azureOpenAiDeployment', 'azureOpenAiKey'")

    api_version = "2023-03-15-preview"
    model = "GPT_3_5_TURBO"
    
    url = f"{azure_openai_base}openai/deployments/{azure_openai_deployment}/completions?api-version={api_version}"
    
    headers = {
        "api-key": azure_openai_key,
    }

    json_data = {
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 1200,
        "stream": False,
        "model": model,
        "stop": '```'
    }

    async with ClientSession() as session:
        try:
            async with session.post(url, json=json_data, headers=headers, timeout=10.0) as response:
                # Check if the response status is OK
                if response.status != 200:
                    raise Exception(f"Request failed with status {response.status}: {await response.text()}")
                
                # Parse the JSON response and handle potential errors
                try:
                    return await response.json()
                except aiohttp.ContentTypeError:
                    raise Exception("Failed to parse response as JSON.")
                
        except aiohttp.ClientError as e:
            # Log and re-raise client errors
            raise Exception(f"HTTP Client error occurred: {str(e)}")
        except asyncio.TimeoutError:
            raise Exception("Request timed out.")
        except Exception as e:
            # Catch all other exceptions
            raise Exception(f"An unexpected error occurred: {str(e)}")

