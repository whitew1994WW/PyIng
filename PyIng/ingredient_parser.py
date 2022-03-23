import PyIng
import tflite_runtime.interpreter as tf_lite
import numpy as np
import re
import json

MODEL_PATH = "../models/output_model.tflite"


def parse_ingredients(ingredients):
    """
    Parses an ingredient or list of ingredients and returns a json indicating the
    name, quantity and unit of the ingredient.

    E.g.
    Input: "1 1/2 ounces of cheese"
    Output:
    {
        "ingredient": "cheese",
        "unit": "ounces",
        "quantity": 1.5
    }

    :param ingredients: String of ingredients or list of strings of ingredients
    :return: Dictionary containing the ingredient, unit and quantity within the input
    """
    pass


def _build_tf_interpreter():
    """
    Constructs a TF Lite model trained and saved from "train_model.ipynb".
    The TF lite model will then be able to take preprocessed input and convert it to a numeric output.
    :return:
    """
    interpreter = tf_lite.Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def _run_tf_interpreter(interpreter, input_data):
    """
    Runs inference on the tensorflow model trained in the ipynb in this project.

    :param interpreter: Tensorflow interpreter created with _build_tf_interpreter
    :param input_data: Input model data, already vectorized, length 60 array of integers
    :return: Unit_name_output - [60, 2] float array, one row for unit and one for name.
    Probabilities of word being a unit  or name respectively.
    """
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    qty_output = interpreter.get_tensor(output_details[0]['index'])
    unit_name_output = interpreter.get_tensor(output_details[1]['index'])

    return unit_name_output, qty_output


def _preprocess_input_ingredient_strings(ingredients: list):
    processed_ings = []
    for ing in ingredients:
        processed_ings.append(_preprocess_input_ingredient_string(ing))
    return processed_ings


def _preprocess_input_ingredient_string(ingredient: str):
    string = re.sub(r"([,./])", r" \g<1> ", ingredient)
    string = re.sub(r'[^\w\s.,/]', r" ", string)
    string = re.sub(r"\s+", r" ", string)
    string = re.sub("(\s$)|(^\s)", "", string)

    string = string.lower()
    # Remove trailing white space
    if string[-1] == " ":
        string = string[:-1]
    if string[0] == " ":
        string = string[1:]
    return string


def _model_output_to_dict(model_name_unit_output, model_qty_output, ingredient: str):
    """
    Converts the tensorflow model output into the useful information
    :param model_name_unit_output:
    Numpy array of shape (2, 60) givin ght eprobabilities of each word being a unit or name
    :param model_qty_output:
    Float indicating the quantity outputted by the model
    :param ingredient:
    Processed ingredient string e.g. "1 ounce plain flour"
    :return:
    Dictionary of parts of input - {"name": "plain flour", "unit": "ounce", "qty", 1}
    - Note the quantity is rounded to two decimal places
    """
    ingredient_list = np.array(ingredient.split(" "))
    # convert the model output to a boolean mask for selecting the words from the input
    name_mask = model_name_unit_output[0, :len(ingredient_list)].astype(np.bool)
    unit_mask = model_name_unit_output[1, :len(ingredient_list)].astype(np.bool)
    name = " ".join(ingredient_list[name_mask])
    unit = " ".join(ingredient_list[unit_mask])
    qty = round(model_qty_output, 2)
    output_dict = {"name": name, "unit": unit, "qty": qty}
    return output_dict


def _model_output_to_list_of_dicts(model_name_unit_output, model_qty_output, ingredients):
    return

def _vectorize_input_strings(input_ingredients):
    """
    Converts parsed input list of ingredients to a list of index lists. Where each word is converted into a number
    ["1 red apple", "1 red apple"] -> [[1, 120, 3], [1, 120, 3]]
    :param input_ingredients: List of ingredients ["1 red apple", "1 red apple"]
    :return: List of words numeric representations [[1, 120, 3], [1, 120, 3]]
    """
    word_index = _load_word_index()
    output_ings = []
    for ingredient in input_ingredients:
        output_ings.append(_vectorize_input_string(ingredient, word_index))
    return output_ings


def _vectorize_input_string(input_ingredient: str, word_index: dict):
    """
    Converts parsed input ingredient to a list of indexs.
    Where each word is converted into a number

    :param input_ingredients: Ingredient String e.g. "1 red apple"
    :param word_index:
    :return: List of words numeric representations [1, 120, 3]
    """
    # Convert to list of words
    word_list = input_ingredient.split(" ")
    idx_list = []

    # Convert list to indexes - replace with OOV token if not in word list
    for word in word_list:
        if word in word_index:
            idx_list.append(word_index[word])
        else:
            idx_list.append(word_index['<OOV>'])
    return idx_list


def _load_word_index():
    """
    Loads in the pre saved word index
    :return: {"word": idx ..... }
    """
    with open("../word_index.json", "r") as f:
        word_index = json.load(f)
    return word_index
