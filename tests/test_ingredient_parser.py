import unittest
import PyIng
import numpy as np
import unittest.mock as mock


class TestIngredientParser(unittest.TestCase):
    def test_parse_ingredients(self):
        self.assertEqual(True, True)

    def test_build_tf_interreter(self):
        interpreter = PyIng.ingredient_parser._build_tf_interpreter()
        self.assertTrue(interpreter)

    def test_run_tf_interpreter(self):
        interpreter = PyIng.ingredient_parser._build_tf_interpreter()
        input_details = interpreter.get_input_details()
        # Test the model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.randint(1, 1000, input_shape), dtype=np.float32)
        output_name, output_qty = PyIng.ingredient_parser._run_tf_interpreter_single_input(interpreter, input_data)

        self.assertEqual(output_name.shape[1:], (2, 60))
        self.assertEqual(output_qty.shape[1], 1)

    def test_run_tf_interpreter_multiple_input(self):
        interpreter = PyIng.ingredient_parser._build_tf_interpreter()
        input_details = interpreter.get_input_details()
        # Test the model on random input data.
        input_shape = (3, 1, 60)
        input_data = np.array(np.random.randint(1, 1000, input_shape), dtype=np.float32)

        output_name, output_qty = PyIng.ingredient_parser._run_tf_interpreter_multiple_input(interpreter, input_data)
        self.assertEqual(output_name.shape, (3, 2, 60))


    def test_preprocess_input_ingredient_string(self):
        input_ingredient = "3 1/2 Teaspoons of salt, (finely diced) "
        expected_output = "3 1 / 2 teaspoons of salt , finely diced"
        output = PyIng.ingredient_parser._preprocess_input_ingredient_string(input_ingredient)
        self.assertEqual(output, expected_output)

    def test_preprocess_input_ingredient_strings(self):
        input_ingredient = ["3 1/2 Teaspoons of salt, (finely diced)"]*4
        expected_output = ["3 1 / 2 teaspoons of salt , finely diced"]*4
        output = PyIng.ingredient_parser._preprocess_input_ingredient_strings(input_ingredient)
        self.assertListEqual(output, expected_output)

    @mock.patch('PyIng.ingredient_parser._load_word_index')
    def test_vectorize_strings(self, mock_word_index):
        mock_word_index.return_value = {"<OOV>": 5, "3": 1, "of": 2, ",": 3}
        input_ingredients = ["3 1 / 2 teaspoons of salt , finely diced"]*4
        output = PyIng.ingredient_parser._vectorize_input_strings(input_ingredients)
        print(output)
        self.assertListEqual(output, [[1, 5, 5, 5, 5, 2, 5, 3, 5, 5]]*4)

    def test_vectorize_string(self):
        input_ingredients = "3 1 / 2 teaspoons of salt , finely diced"
        word_index = {"<OOV>": 5, "3": 1, "of": 2, ",": 3}
        output = PyIng.ingredient_parser._vectorize_input_string(input_ingredients, word_index)
        print(output)
        self.assertEqual(output, [1, 5, 5, 5, 5, 2, 5, 3, 5, 5])

    def test_load_word_index(self):
        word_index = PyIng.ingredient_parser._load_word_index()
        self.assertIsInstance(word_index, dict)

    def test_model_output_to_dict(self):
        model_name_unit_output = np.zeros((2, 60), dtype=np.float64)
        model_name_unit_output[0][2:4] = 0.6
        model_name_unit_output[1][1] = 0.6
        qty_output = 12.3452
        input_parsed_ing = "1 ounce plain flour"
        output_dict = PyIng.ingredient_parser._model_output_to_dict(model_name_unit_output, qty_output, input_parsed_ing)
        expected_output = {
            "name": "plain flour",
            "unit": "ounce",
            "qty": 12.35
        }
        self.assertEqual(output_dict, expected_output)

    def test_model_output_to_list_of_dicts(self):
        model_name_unit_outputs = np.zeros((3, 2, 60), dtype=np.float64)
        for i in range(3):
            model_name_unit_outputs[i][0][2:4] = 0.6
            model_name_unit_outputs[i][1][1] = 0.6
        qty_outputs = [12.3452]*3
        input_parsed_ings = ["1 ounce plain flour"]*3
        output_dict = PyIng.ingredient_parser._model_output_to_list_of_dicts(model_name_unit_outputs, qty_outputs, input_parsed_ings)
        expected_output = 3 *[{
            "name": "plain flour",
            "unit": "ounce",
            "qty": 12.35
        }]
        self.assertEqual(output_dict, expected_output)

    def test_pad_post_array(self):
        input_array = [[1, 2, 3], [1, 2, 3, 4], [1, 2]]
        length = 60
        padded_array = PyIng.ingredient_parser._pad_post_array(input_array, length)
        self.assertEqual(len(padded_array), 3)
        self.assertEqual(len(padded_array[0]), 60)
        self.assertEqual(padded_array[0][2], 3)

    def test_parse_ingredient(self):
        input_ingredients = ["1 teaspoon flour", "3 cylinders of tomatoes", "13.5 cm of ginger", "1 1/2 cups plus 1 teaspoon plain flour"]
        expected_output = [{"name": "flour", "unit": "teaspoon",  "qty": 1.00}, {"name": "tomatoes", "unit": "cylinders",  "qty": 3.00}, {"name": "ginger", "unit": "cm",  "qty": 13.50}, {}]
        output = PyIng.ingredient_parser.parse_ingredients(input_ingredients)
        self.assertListEqual(output, expected_output)

if __name__ == '__main__':
    unittest.main()
