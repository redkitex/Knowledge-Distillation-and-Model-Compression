import pytest
from kd.answer_extraction import extract_last_number, gold_answer
from kd.data import load_split, find_ungradable_solutions

def test_extract_last_number_basics():
    # Simple numbers and selection of the last one
    assert extract_last_number("The answer is 42") == 42.0
    assert extract_last_number("First I had 2 apples, now I have 5") == 5.0

def test_extract_last_number_formatting():
    # Commas, dollar signs, decimals, and negatives
    assert extract_last_number("Total cost: $1,234.56") == 1234.56
    assert extract_last_number("The temperature dropped to -15.5") == -15.5
    assert extract_last_number("A cool $1,000,000") == 1000000.0
    # Stuck to the final punctuation
    assert extract_last_number("The result is 100.") == 100.0

def test_extract_last_number_no_number():
    # Error cases or unformatted responses
    assert extract_last_number("There is no number in this response.") is None
    assert extract_last_number("") is None
    assert extract_last_number(None) is None

def test_gold_answer():
    # Standard GSM8K format
    assert gold_answer("Let's calculate... \n#### 456") == 456.0
    
    # Reasoning with numbers (should ignore them and take only the one after ####)
    assert gold_answer("Step 1 gives 50. Step 2 gives 20.\n#### 70") == 70.0
    
    # Format with symbols in the official answer
    assert gold_answer("The final price is\n#### $2,000") == 2000.0
    
    # Fallback if for some reason the dataset doesn't have '####'
    assert gold_answer("Just the number 42") == 42.0

def test_all_human_solutions_gradable():
    #Load the training split for validation
    train_examples = load_split("train")
    
    # Creates the wrapper that data.py needs
    def grader(human_response: str, gold: str) -> bool:
        pred = extract_last_number(human_response)
        if pred is None:
            return False
        return abs(pred - float(gold)) < 1e-6

    # We searcg for any errors
    ungradable = find_ungradable_solutions(train_examples, grader)
    
    # Needs to be empty
    assert len(ungradable) == 0, f"{len(ungradable)} problems were found that were not graded correctly: {ungradable[:5]}"