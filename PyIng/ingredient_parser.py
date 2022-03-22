
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


def _build_tf_model():
    """
    Constructs a TF Lite model trained and saved from "train_model.ipynb".
    The TF lite model will then be able to take preprocessed input and convert it to a numeric output.
    :return:
    """
    interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    my_signature = interpreter.get_signature_runner()

    # my_signature is callable with input as arguments.
    output = my_signature(x=tf.constant([1.0], shape=(1,10), dtype=tf.float32))
    # 'output' is dictionary with all outputs from the inference.
    # In this case we have single output 'result'.
    print(output['result'])


def _preprocess_input_ingredients(ingredients):
    return


def _model_output_to_json(model_output, ingredients):
    return

