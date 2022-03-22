import unittest
import PyIng
import numpy as np


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
        output_name, output_qty = PyIng.ingredient_parser._run_tf_interpreter(interpreter, input_data)

        self.assertEqual(output_name.shape[1:], (2, 60))
        self.assertEqual(output_qty.shape[1], 1)

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

    def test_vectorize_strings(self):
        input_ingredients = ["3 1 / 2 teaspoons of salt , finely diced"]*4
        output = PyIng.ingredient_parser._vectorize_input_strings(input_ingredients)
        self.assertEqual(output.shape, (4, 10))

    def test_pad_sequences(self):
        input_ingredients = ["3 1 / 2 teaspoons of salt , finely diced"]*4
        output = PyIng.ingredient_parser._pad_sequences(input_ingredients)
        self.assertEqual(output.shape, (4, 60))
if __name__ == '__main__':
    unittest.main()
