import re

def extract_last_number(text: str):
    """
    Extract the last number from a model-generated text.
    Handles commas, dollar signs, decimals, and negative numbers.
    Returns None if no number is found.
    """

    if not isinstance(text, str) or not text.strip():
        return None
        
    # Clean common symbols that may interfere with extraction
    cleaned_text = text.replace(",", "").replace("$", "")
    
    # Regular expression to match numbers, including:
    # -?     : Optional negative sign
    # \d+    : One or more digits
    # (?:\.\d+)? : Optional decimal part
    numbers = re.findall(r"-?\d+(?:\.\d+)?", cleaned_text)
    
    if not numbers:
        return None
        
    return float(numbers[-1])

def gold_answer(answer_field: str):
    """
    Extract the correct number from the reference solution of GSM8K.
    Answers in GSM8K always end with '#### <number>'.
    """
    parts = answer_field.split("####")
    
    # If for some reason it doesn't have the standard format, we search the entire text
    if len(parts) < 2:
        return extract_last_number(answer_field)
        
    # Isolate the final part after '####' and extract the number
    final_part = parts[-1]
    return extract_last_number(final_part)