"""Unit tests for modules."""
import unittest
import jax.numpy as jnp
from flax import nnx
from modules import SinusoidalEmbedding_, TimeEmbedding_, Attn, Block, ResNetBlock, UNet


class TestSinusoidalEmbedding(unittest.TestCase):
    """
    Unit tests for the SinusoidalEmbedding class.
    """

    def test_call(self):
        """
        Test the call method of the SinusoidalEmbedding class.
        """
        dim = 32
        embedding = SinusoidalEmbedding_(dim=dim)
        inputs = jnp.ones((10, 20))
        output = embedding(inputs)
        self.assertEqual(output.shape, (10, 20, dim))

class TestTimeEmbedding(unittest.TestCase):
    """
    Unit tests for the TimeEmbedding_ class.
    """

    def test_call(self):
        """
        Test the call method of the TimeEmbedding_ class.

        This method tests the functionality of the call method in the TimeEmbedding_ class.
        It creates an instance of the TimeEmbedding_ class, passes some inputs to it, and
        checks if the output has the expected shape.

        Returns:
            None
        """
        dim = 32
        rngs = nnx.Rngs(0)
        time_embedding = TimeEmbedding_(rngs=rngs, dim=dim)
        inputs = jnp.ones((10, 20))
        output = time_embedding(inputs)
        self.assertEqual(output.shape, (10, 20, dim * 4))

class TestAttn(unittest.TestCase):
    """
    Unit tests for the Attn class.
    """

    def test_call(self):
        """
        Test the `call` method of the `Attn` class.

        This method creates an instance of the `Attn` 
        class and calls its `call` method with some inputs.
        It then asserts that the output shape is as expected.

        Returns:
            None
        """
        in_features = 32
        dim = 32
        num_heads = 4
        rngs = nnx.Rngs(0)
        attn = Attn(in_features=in_features, dim=dim, num_heads=num_heads, rngs=rngs)
        inputs = jnp.ones((10, 20, in_features))
        output = attn(inputs)
        self.assertEqual(output.shape, (10, 20, in_features))

class TestBlock(unittest.TestCase):
    """
    Unit tests for the `Block` class.
    """

    def test_call(self):
        """
        Test the `call` method of the `Block` class.

        This method creates an instance of the `Block` class and tests its 
        `call` method by passing in a tensor of ones as input.
        It then asserts that the output shape matches the expected shape.

        Returns:
            None
        """
        in_features = 32
        out_features = 64
        rngs = nnx.Rngs(0)
        block = Block(in_features=in_features, out_features=out_features, rngs=rngs)
        inputs = jnp.ones((10, 20, in_features))
        output = block(inputs)
        self.assertEqual(output.shape, (10, 20, out_features))

class TestResNetBlock(unittest.TestCase):
    """
    Unit tests for the ResNetBlock class.
    """

    def test_call(self):
        """
        Test the `call` method of the ResNetBlock class.

        This method creates an instance of the ResNetBlock class and tests its 
        `call` method by passing in some inputs and a time embedding.
        It then asserts that the shape of the output matches the expected shape.

        Returns:
            None
        """
        in_features = 32
        out_features = 64
        rngs = nnx.Rngs(0)
        resnet_block = ResNetBlock(in_features=in_features, out_features=out_features, time_emb_dim=out_features * 4, rngs=rngs)
        inputs = jnp.ones((10, 20, 30, in_features))
        time_embed = jnp.ones((10, out_features * 4))
        output = resnet_block(inputs, time_embed)
        self.assertEqual(output.shape, (10, 20, 30, out_features))

class TestUNet(unittest.TestCase):
    """
    Unit tests for the UNet class.
    """

    def test_call(self):
        """
        Test the `call` method of the UNet class.

        This method creates an instance of the UNet class and calls its `call` method with some inputs.
        It then asserts that the output shape is as expected.

        Returns:
            None
        """
        rngs = nnx.Rngs(0)
        num_channels = 3
        out_features = 64
        num_groups = 8
        num_heads = 4
        dim_scale_factor = (1, 2, 4, 8)
        unet = UNet(
            rngs=rngs, 
            num_channels=num_channels, 
            out_features=out_features, 
            num_groups=num_groups, 
            num_heads=num_heads, 
            dim_scale_factor=dim_scale_factor
            )
        inputs = jnp.ones((10, 16, 16, num_channels))
        time = jnp.ones((10,))
        output = unet((inputs, time))
        self.assertEqual(output.shape, (10, 16, 16, num_channels))

if __name__ == '__main__':
    unittest.main()