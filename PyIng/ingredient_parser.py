import PyIng
import tflite_runtime.interpreter as tf_lite
import numpy as np
import re
import pickle
from pkg_resources import resource_filename

WORD_INDEX_PATH = resource_filename('PyIng', "data/word_index.pckl")
MODEL_PATH = resource_filename('PyIng', "data/output_model.tflite")


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
    if isinstance(ingredients, str):
        ingredients = [ingredients]

    elif isinstance(ingredients, list):
        for ing in ingredients:
            if not isinstance(ing, str):
                raise ValueError("Input ingredients must all be strings!")
    else:
        raise ValueError("Ingredients is not the correct format. " +
                         "It must be either an ingredient string or a list of ingredient strings")

    # Preprocess ingredients
    processed_ingredients = _preprocess_input_ingredient_strings(ingredients)
    vectorized_ingredients = _vectorize_input_strings(processed_ingredients)
    padded_ing_vectors = _pad_post_array(vectorized_ingredients, 60)

    # Convert to float for input to interpreter
    interpreter_input = np.asarray(padded_ing_vectors).astype(np.float32)
    # Build the interpreter and run the input through
    interpreter = _build_tf_interpreter()

    unit_name_output, qty_output, qty_decimal_output = _run_tf_interpreter_multiple_input(interpreter, interpreter_input)
    qty_output = _round_qty(qty_output, qty_decimal_output)

    # Build the output
    output = _model_output_to_list_of_dicts(unit_name_output, qty_output, processed_ingredients)

    if len(output) == 1:
        output = output[0]
    return output


def _build_tf_interpreter():
    """
    Constructs a TF Lite model trained and saved from "train_model.ipynb".
    The TF lite model will then be able to take preprocessed input and convert it to a numeric output.
    :return:
    """
    interpreter = tf_lite.Interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


def _run_tf_interpreter_single_input(interpreter, input_data):
    """
    Runs inference on the tensorflow model trained in the ipynb in this project.

    :param interpreter: Tensorflow interpreter created with _build_tf_interpreter
    :param input_data: Input model data, already vectorized, length (1, 60) array of integers
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
    qty_decimal_output = interpreter.get_tensor(output_details[2]['index'])
    qty_output = interpreter.get_tensor(output_details[1]['index'])
    unit_name_output = interpreter.get_tensor(output_details[0]['index'])

    return unit_name_output, qty_output, qty_decimal_output


def _run_tf_interpreter_multiple_input(interpreter, input_data):
    num_ings = input_data.shape[0]
    output_unit_name = np.zeros((num_ings, 2, 60))
    output_qty = np.zeros(num_ings)
    output_qty_decimal = np.zeros((num_ings, 3))
    for i in range(num_ings):
        inp_array = np.reshape(input_data[i, :], (1, 60))
        unit_name, qty, qty_decimal = _run_tf_interpreter_single_input(interpreter, inp_array)
        output_unit_name[i, :, :] = unit_name
        output_qty[i] = qty
        output_qty_decimal[i, :] = qty_decimal
    return output_unit_name, output_qty, output_qty_decimal


def _preprocess_input_ingredient_strings(ingredients: list):
    """
    See _preprocess_input_ingredient_string.
    :param ingredients: List of Ingredient strings
    :return: processed list of ingredient strings
    """
    processed_ings = []
    for ing in ingredients:
        processed_ings.append(_preprocess_input_ingredient_string(ing))
    return processed_ings


def _preprocess_input_ingredient_string(ingredient: str):
    """
    Processes input ingredients to remove punctuation, de-capitalise amd remove trainling white space
    :param ingredient: ingredient string e.g. "3 Tablespoons Salt (fine)
    :return: processed string "3 tablespoons salt fine
    """
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
    rounded_name_output = np.round(model_name_unit_output[0, :len(ingredient_list)])
    rounded_unit_output = np.round(model_name_unit_output[1, :len(ingredient_list)])

    name_mask = rounded_name_output.astype(bool)
    unit_mask = rounded_unit_output.astype(bool)
    name = " ".join(ingredient_list[name_mask])
    unit = " ".join(ingredient_list[unit_mask])
    qty = round(model_qty_output, 2)
    output_dict = {"name": name, "unit": unit, "qty": qty}
    return output_dict


def _model_output_to_list_of_dicts(model_name_unit_output, model_qty_output, ingredients):
    """
    See _model_output_to_dict. Deals with multiple model outputs for multiple input ingredients.
    :param model_name_unit_output: Numpy array of shape (None, 2, 60)
    :param model_qty_output: List of quantity outputs from the model
    :param ingredients: List of ingredients
    :return: List of dictionarys containing parsed information
    """
    output_parsed_list = []
    for i in range(len(ingredients)):
        parsed_ing = _model_output_to_dict(model_name_unit_output[i, :, :], model_qty_output[i], ingredients[i])
        output_parsed_list.append(parsed_ing)
    return output_parsed_list


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


def _pad_post_array(input_array, length):
    """
    Pads the input 2D list with 0's at the end so that is is of shape (None, length)
    :param input_array: 2D list
    :param length: length to pad to
    :return: (None, length) padded array
    """
    output_array = []
    for arr in input_array:
        padded = arr + (length - len(arr))*[0]
        output_array.append(padded)
    return output_array


def _load_word_index():
    """
    Loads in the pre saved word index
    :return: {"word": idx ..... }
    """
    with open(WORD_INDEX_PATH, "rb") as f:
        word_index = pickle.load(f)
    return word_index


def _round_qty(qty_output, qty_decimal_output):
    """
    Rounds each element of qty_output to 0, 1 or 2 decimal places based on the corresponding one hot array in
    qty_decimal_output
    :param qty_output: Qty Predictions from _run_tf_interpreter_multiple_inputs array size (None, 1)
    :param qty_decimal_output: Qty Decimal predictions from same function as above, size (None, 3)
    :return: Returns an array the same shape as qty_output
    """
    for i in range(qty_output.shape[0]):
        idx_oh = np.argmax(qty_decimal_output[i, :])
        qty_output[i] = round(qty_output[i], idx_oh)
    return qty_output


