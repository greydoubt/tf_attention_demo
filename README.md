# tf_attention_demo
basic demonstration of "attention"

# Attention and Memory Demonstration

This project demonstrates the concepts of attention and memory using TensorFlow. It performs a series of steps, including preparing inputs, initializing weights, deriving key, query, and value, calculating attention scores, applying softmax, multiplying scores with values, and summing weighted values to obtain the output.

## Code Explanation

The code consists of two main files:

1. `main.py`: This file contains the main code that executes the steps and demonstrates the attention and memory concepts.
2. `attention.py`: This file contains utility functions used in the main code, including initializing weights, deriving key, query, and value, calculating attention scores, applying softmax, multiplying scores with values, and summing weighted values.

## Usage

To run the code, follow these steps:

1. Install the required dependencies by running `pip install tensorflow matplotlib`.
2. Run the `main.py` file using `python main.py`.

## Output

The code generates the following output:

- Text Output: The code prints the output of each step to the terminal.
- Text Files: The code saves the output of each step to separate text files with timestamps in the `output` folder.
- Graph Images: The code generates and saves graphs of the attention scores, attention weights, and weighted values as PNG images with timestamps in the `output` folder.

## Function Decorators

Function decorators are used in this code to decorate the step functions. Decorators provide a way to modify the behavior of functions without changing their source code. In this project, the `step_decorator` function is defined, which wraps each step function. The `step_decorator` function handles the output of the step result, prints it to the terminal, and saves the text and graph files.

## Assertion Call

The code uses the `assert_raises` context manager to catch and handle specific exceptions. The `assert_raises` context manager is used to assert that a specific error is raised during the execution of the main code. If the expected error is not raised, an `AssertionError` is raised. The traceback of any caught exception is printed to the terminal.

To understand the behavior of the code and ensure its correctness, the `assert_raises` context manager is used in the `__main__` block to catch and handle any `AssertionError` that might occur during the execution of the main code.

---

Feel free to modify the code and explore different inputs or variations of attention and memory concepts!
