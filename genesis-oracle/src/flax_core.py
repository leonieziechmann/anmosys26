import jax
import jax.numpy as jnp
import flax.linen as nn

# 1. Define the Multi-Layer Perceptron (MLP) module using Flax Linen
class MultiLayerPerceptron(nn.Module):
    """
    A simple stateless Multi-Layer Perceptron (MLP) Module in Flax.
    This class serves solely as a computational blueprint (the architecture).
    Unlike Keras, it does NOT store or manage any weights/biases in its instance variables.
    """
    latent_dim: int = 8
    output_dim: int = 50

    @nn.compact
    def __call__(self, x):
        # First Dense Layer with ReLU activation
        x = nn.Dense(features=self.latent_dim, name="dense_compression")(x)
        x = nn.relu(x)
        # Second Dense Layer (linear output reconstruction)
        x = nn.Dense(features=self.output_dim, name="dense_expansion")(x)
        return x

if __name__ == "__main__":
    print("=============================================================")
    print("      FLAX STATELESS ARCHITECTURE DEMONSTRATION")
    print("=============================================================\n")
    
    # JAX random number generation is strictly stateless and reproducible!
    # A single seed is split explicitly to generate deterministic randomness.
    key = jax.random.PRNGKey(42)
    key_init, key_input = jax.random.split(key)
    
    # Create dummy input: Batch size of 4, window size of 50
    # JAX requires a sample input during initialization to infer input shape shapes.
    dummy_input = jax.random.normal(key_input, (4, 50))
    print(f"[*] Input Data Shape: {dummy_input.shape}")
    
    # 2. Model Blueprint Instantiation
    # Instantiating the module creates a stateless architecture wrapper.
    # No weights or biases are allocated, and 'model' itself has no trainable state.
    model = MultiLayerPerceptron(latent_dim=8, output_dim=50)
    print("[*] Instantiated MultiLayerPerceptron. Architecture is fully stateless.")
    
    # 3. Explicit Model Weight Initialization (model.init)
    # The actual parameters are generated and returned as a nested frozen dictionary (PyTree).
    # The 'model' object does NOT retain these parameters. They are stored in 'variables'.
    variables = model.init(key_init, dummy_input)
    
    print("\n[*] Model Variables (Parameters) initialized externally:")
    print(f"    Variables Type: {type(variables)}")
    
    # Inspect the layers, weights, and biases stored in our immutable parameters dict
    for layer_name, params in variables["params"].items():
        print(f"\n    Layer '{layer_name}':")
        for param_name, param_tensor in params.items():
            print(f"      - '{param_name}': shape = {param_tensor.shape}, dtype = {param_tensor.dtype}")
            
    # 4. Explicit Stateless Forward Pass (model.apply)
    # To run a forward pass, we call model.apply and must explicitly provide the
    # variables dict alongside the inputs. The neural network computes the outputs
    # as a pure function of these two parameters without reading or writing internal state.
    print("\n[*] Running Forward Pass via model.apply...")
    outputs = model.apply(variables, dummy_input)
    print(f"    Forward Pass Output Shape: {outputs.shape}")
    
    # 5. Demonstrating Functional Purity: Modifying Weights on the Fly
    # Since parameters are just a standard JAX PyTree, we can map over them or swap them instantly.
    # Let's zero out all weights in our parameter dictionary:
    print("\n[*] Demonstrating Functional Purity: zeroing out parameters...")
    zero_variables = jax.tree_util.tree_map(lambda param: param * 0.0, variables)
    
    # Run the forward pass with the zeroed variables
    zero_outputs = model.apply(zero_variables, dummy_input)
    print(f"    Mean of outputs with zeroed weights: {jnp.mean(zero_outputs):.6f} (Should be exactly 0)")
    print("\n=============================================================")
