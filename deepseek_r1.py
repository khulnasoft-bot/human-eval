import aiohttp
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deepseek-r1')

# You should replace this with your desired API key and endpoint
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_API_URL = os.environ.get('DEEPSEEK_API_URL', 'https://api.deepseek.com/v1')

if not DEEPSEEK_API_KEY:
    logger.error("DEEPSEEK_API_KEY environment variable is not set!")
    raise ValueError("DEEPSEEK_API_KEY is required.")

if not DEEPSEEK_API_URL:
    logger.warning("DEEPSEEK_API_URL not set. Using default: https://api.deepseek.com/v1")

# Headers for the API requests
DEEPSEEK_HEADERS = {
    "X-API-Key": DEEPSEEK_API_KEY,
    "Content-Type": "application/json",
}

async def deepseek_complete(prompt: str, model: str, temperature=0.0):
    """
    Makes a request to the DeepSeek API to complete a given prompt.
    """
    async with aiohttp.ClientSession() as session:
        try:
            logger.info('Sending request to DeepSeek API')
            response = await session.post(f'{DEEPSEEK_API_URL}/complete',
                                          headers=DEEPSEEK_HEADERS,
                                          json={
                                              "prompt": prompt,
                                              "model": model,
                                              "max_tokens": 500,
                                              "temperature": temperature,
                                              "stream": False,
                                              "stop": "```"
                                          })

            # Check if the response status is OK
            response.raise_for_status()

            data = await response.json()

            # Extract completion from the response
            completion = data.get('completion', '')
            if completion.endswith('```'):
                completion = completion[:-3]  # Clean up the ending if needed

            logger.info(f"Prompt:\n{prompt}")
            logger.info("_____")
            logger.info(f"Completion:\n{completion}")
            logger.info("\n\n\n")

            return [completion]

        except aiohttp.ClientError as e:
            logger.error(f"Request error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []

if __name__ == "__main__":
    # Example of usage
    import asyncio
    prompt = "What is the capital of France?"
    model = "deepseek-model-v1"
    asyncio.run(deepseek_complete(prompt, model))
