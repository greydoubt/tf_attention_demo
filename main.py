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

def output_text(text):
    print(text)

    # Save text output to file with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"output_{timestamp}.txt"
    with open(file_name, "w") as file:
        file.write(text)

def output_graph(data, title):
    plt.plot(data)
    plt.title(title)

    # Save graph image to file with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"graph_{timestamp}.png"
    plt.savefig(file_name)
    plt.close()

@tf.function
def main():
    # Step 1: Prepare inputs
    x = tf.constant(
        [
            [1, 0, 1, 0],  # Input 1
            [0, 2, 0, 2],  # Input 2
            [1, 1, 1, 1],  # Input 3
        ],
        dtype=tf.float32,
    )
    output_text("Input:")
    output_text(str(x.numpy()))

    # Step 2: Initialize weights
    weights = initialize_weights()
    output_text("Weights:")
    output_text(str(weights.numpy()))

    # Step 3: Derive key, query, and value
    key, query, value = derive_key_query_value(x, weights)
    output_text("Key:")
    output_text(str(key.numpy()))
    output_text("Query:")
    output_text(str(query.numpy()))
    output_text("Value:")
    output_text(str(value.numpy()))

    # Step 4: Calculate attention scores for Input 1
    attention_scores = calculate_attention_scores(query, key)
    output_text("Attention Scores:")
    output_text(str(attention_scores.numpy()))
    output_graph(attention_scores.numpy(), "Attention Scores")

    # Step 5: Apply softmax
    attention_weights = apply_softmax(attention_scores)
    output_text("Attention Weights:")
    output_text(str(attention_weights.numpy()))
    output_graph(attention_weights.numpy(), "Attention Weights")

    # Step 6: Multiply scores with values
    weighted_values = multiply_scores_with_values(attention_weights, value)
    output_text("Weighted Values:")
    output_text(str(weighted_values.numpy()))
    output_graph(weighted_values.numpy(), "Weighted Values")

    # Step 7: Sum weighted values to get Output 1
    output = sum_weighted_values(weighted_values)
    output_text("Output:")
    output_text(str(output.numpy()))

    # Save the final output as a separate text file
    output_text("Final Output:")
    output_text(str(output.numpy()))

if __name__ == "__main__":
    with assert_raises(AssertionError):
        main()
