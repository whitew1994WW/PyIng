# PyIng - Ingredient parser
This is a python package for parsing ingredient strings. There is only one function `parse_ingredients`. It takes in a 
single string or list of strings and returns a dictionary containing the name, quantity and unit in that recipe string.
## Get Started

First install the package using pip:

```commandline
pip install pying
```

Then you can use it like follows:

```python
from PyIng import parse_ingredients

ingredients = ["3 large melons", "5 1/2 cups water", "2 cups flour"]

parsed_ingredients = parse_ingredients(ingredients)
```

the output `parsed_ingredients` should look like so:

```python
parsed_ingredients = [
    {
        "name": "melons",
        "unit": None,
        "qty": 3.0
    },
    {
        "name": "water",
        "unit": "cups",
        "qty": 5.5
    },
    {
        "name": "flour",
        "unit": "cups",
        "qty": 2
    }
]
```

## Training the model

The ingredient parser uses a LSTM model written in tensorflow. It is trained on a publically available dataset produced 
by the new york times:

https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjowpyb9uj2AhWNiFwKHcFQBC8QFnoECAgQAQ&url=https%3A%2F%2Fgithub.com%2Fnytimes%2Fingredient-phrase-tagger&usg=AOvVaw1AHIgZ0BfSe8ddG7E8alYt 

The entire model is trained using the `train_model.ipynb` and the model is saved in tflite format. Please try and improve on my model, it is far from optimal, I wanted to get something that works without spending too long.

THere are more details on training the model in the jupyter notebook.
