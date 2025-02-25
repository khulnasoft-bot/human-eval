import re

def parse_prompt(prompt: str):
    """
    Parse the prompt into a list of message dictionaries, where each dictionary contains 
    a 'role' and 'content' from the blocks between <|im_start|> and <|im_end|> tokens.
    """
    # Grab all blocks between the <|im_start|> and <|im_end|> tokens
    im_blocks = re.findall(r"<\|im_start\|>(.+?)<\|im_end\|>", prompt, re.DOTALL)

    messages = []
    for block in im_blocks:
        # Separate by <|im_sep|> token
        role, content = block.split("<|im_sep|>", 1)
        messages.append({"role": role.strip(), "content": content.strip()})

    return messages

def to_prompt(messages):
    """
    Convert a list of message dictionaries into a formatted prompt string.
    """
    return "".join([f"<|im_start|>{message['role']}<|im_sep|>{message['content']}<|im_end|>" for message in messages])
