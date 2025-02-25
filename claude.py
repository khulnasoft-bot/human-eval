import aiohttp
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANTHROPIC_HEADERS = {
    "X-API-Key": os.environ.get('ANTHROPIC_API_KEY'),
    "Content-Type": "application/json",
}

if not ANTHROPIC_HEADERS["X-API-Key"]:
    logger.error("ANTHROPIC_API_KEY environment variable is not set!")
    raise ValueError("ANTHROPIC_API_KEY is required.")

async def claude_complete(prompt: str, model: str, temperature=0.0):
    async with aiohttp.ClientSession() as session:
        try:
            logger.info('Running Claude')
            response = await session.post('https://api.anthropic.com/v1/complete', 
                                          headers=ANTHROPIC_HEADERS, 
                                          json={
                "prompt": prompt,
                "model": model,
                "max_tokens_to_sample": 500,
                "temperature": temperature,
                "stream": False,
                "stop": "```"
            })

            # Check if the response status is OK
            response.raise_for_status()

            data = await response.json()

            completion = data.get('completion', '')
            if completion.endswith('```'):
                completion = completion[:-3]

            logger.info(f"Prompt:\n{prompt}")
            logger.info("_____")
            logger.info(f"Completion:\n{completion}")
            logger.info("\n\n\n")

            return [completion]

        except aiohttp.ClientError as e:
            logger.error(f"An error occurred while making the request: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return []
