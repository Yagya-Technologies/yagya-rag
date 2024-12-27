
text = """
Electromotive force (EMF) and potential difference (PD) are two closely related but distinct concepts in the context of electric circuits.\n\n1. **Source vs Load**: \n   - EMF is the difference in potential between two terminals of an electric source like a cell, generator, or solar cell when the circuit is open, i.e., it's not connected to a load. \n   - PD, on the other hand, is the difference in potential between two points in a circuit, such as between the two ends of a load or between two terminals of a source when the circuit is connected.\n\n2. \"Presence of a Load:\" \n   - EMF is measured in an open circuit, i.e., when the electric source is not connected to a load, whereas \n   - PD is always measured in a closed circuit.\n\n3. Definition: \n   - EMF is defined as the electromotive force or the potential difference that is created by a power source to drive electric current through a circuit. \n   - PD is the potential difference between two points in a circuit and is measured in terms of the work needed to move a unit charge from one point to another.\n\n4. Connection: \n   - EMF and PD are related by Ohm's Law, i.e., V = E (when circuit is open) and V = I R (when the circuit is connected), where 'V' refers to the electromotive force when the circuit is open and the potential difference"""

def text_formatter(raw_text: str, output_file: str):
    """
    Format the raw text by replacing newline characters and save to a file.

    Args:
        raw_text (str): The raw text containing newline characters.
        output_file (str): The file path to save the formatted text.
    """
    formatted_text = raw_text.replace('\\n\\n', ' ').replace('\\n', ' ')
    with open(output_file, 'w') as f:
        f.write(formatted_text)
    return formatted_text

text_formatter(text, './ss.txt')