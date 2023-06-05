import tensorflow as tf
from utils.attention import (
    initialize_weights,
    derive_key_query_value,
    calculate_attention_scores,
    apply_softmax,
    multiply_scores_with_values,
    sum_weighted_values,
)


# https://www.tensorflow.org/guide/function

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

    # Step 2: Initialize weights
    weights = initialize_weights()

    # Step 3: Derive key, query, and value
    key, query, value = derive_key_query_value(x, weights)

    # Step 4: Calculate attention scores for Input 1
    attention_scores = calculate_attention_scores(query, key)

    # Step 5: Apply softmax
    attention_weights = apply_softmax(attention_scores)

    # Step 6: Multiply scores with values
    weighted_values = multiply_scores_with_values(attention_weights, value)

    # Step 7: Sum weighted values to get Output 1
    output = sum_weighted_values(weighted_values)

    # Print the output
    print("Output 1:")
    print(output.numpy())

if __name__ == "__main__":
    main()
