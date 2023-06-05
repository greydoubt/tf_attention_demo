import tensorflow as tf
from utils.attention import (
    initialize_weights,
    derive_key_query_value,
    calculate_attention_scores,
    apply_softmax,
    multiply_scores_with_values,
    sum_weighted_values,
)
import traceback
import contextlib
import datetime
import matplotlib.pyplot as plt
import os

@contextlib.contextmanager
def assert_raises(error_class):
    try:
        yield
    except error_class as e:
        traceback.print_exc()
        pass
    else:
        raise AssertionError(f"Expected {error_class} to be raised, but no exception was raised.")

def create_output_folder():
    # Create an "output" folder if it doesn't exist
    if not os.path.exists("output"):
        os.makedirs("output")

def output_text(text):
    print(text)

    # Save text output to file with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"output/output_{timestamp}.txt"
    with open(file_name, "w") as file:
        file.write(text)

def output_graph(data, title):
    plt.plot(data)
    plt.title(title)

    # Save graph image to file with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"output/graph_{timestamp}.png"
    plt.savefig(file_name)
    plt.close()

def step_decorator(func):
    def wrapper(*args, **kwargs):
        output_text(f"Step: {func.__name__}")
        result = func(*args, **kwargs)
        output_text(str(result.numpy()))
        output_graph(result.numpy(), func.__name__)
        return result
    return wrapper

@tf.function
@step_decorator
def prepare_inputs():
    # Step 1: Prepare inputs
    x = tf.constant(
        [
            [1, 0, 1, 0],  # Input 1
            [0, 2, 0, 2],  # Input 2
            [1, 1, 1, 1],  # Input 3
        ],
        dtype=tf.float32,
    )
    return x

@tf.function
@step_decorator
def initialize_weights_step():
    # Step 2: Initialize weights
    weights = initialize_weights()
    return weights

@tf.function
@step_decorator
def derive_key_query_value_step(x, weights):
    # Step 3: Derive key, query, and value
    key, query, value = derive_key_query_value(x, weights)
    return key, query, value

@tf.function
@step_decorator
def calculate_attention_scores_step(query, key):
    # Step 4: Calculate attention scores for Input 1
    attention_scores = calculate_attention_scores(query, key)
    return attention_scores

@tf.function
@step_decorator
def apply_softmax_step(attention_scores):
    # Step 5: Apply softmax
    attention_weights = apply_softmax(attention_scores)
    return attention_weights

@tf.function
@step_decorator
def multiply_scores_with_values_step(attention_weights, value):
    # Step 6: Multiply scores with values
    weighted_values = multiply_scores_with_values(attention_weights, value)
    return weighted_values

@tf.function
@step_decorator
def sum_weighted_values_step(weighted_values):
    # Step 7: Sum weighted values to get Output 1
    output = sum_weighted_values(weighted_values)
    return output

def main():
    create_output_folder()

    # Step 1: Prepare inputs
    x = prepare_inputs()

    # Step 2: Initialize weights
    weights = initialize_weights_step()

    # Step 3: Derive key, query, and value
    key, query, value = derive_key_query_value_step(x, weights)

    # Step 4: Calculate attention scores for Input 1
    attention_scores = calculate_attention_scores_step(query, key)

    # Step 5: Apply softmax
    attention_weights = apply_softmax_step(attention_scores)

    # Step 6: Multiply scores with values
    weighted_values = multiply_scores_with_values_step(attention_weights, value)

    # Step 7: Sum weighted values to get Output 1
    output = sum_weighted_values_step(weighted_values)

    # Save the final output as a separate text file
    output_text("Final Output:")
    output_text(str(output.numpy()))

if __name__ == "__main__":
    with assert_raises(AssertionError):
        main()
