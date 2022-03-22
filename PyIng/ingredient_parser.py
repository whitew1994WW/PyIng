import PyIng
import tflite_runtime.interpreter as tf_lite
import numpy as np
import re


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


def _model_output_to_json(model_output, ingredients):
    return


def _pad_sequences(input_ingredients):
    return None


def _vectorize_input_strings(input_ingredients):

    return None