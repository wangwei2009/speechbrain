"""
This file contains a very simple PyTorch module to use for enhancement.

To replace this model, change the `!new:` tag in the hyperparameter file
to refer to a built-in SpeechBrain model or another file containing
a custom PyTorch module.

Authors
 * Peter Plantinga 2021
"""
import torch


class CustomModel(torch.nn.Module):
    """Basic RNN model with projection layers between RNN layers.

    Arguments
    ---------
    input_size : int
        Size of the expected input in the 3rd dimension.
    rnn_size : int
        Number of neurons to use in rnn (for each direction -> and <-).
    projection : int
        Number of neurons in projection layer.
    layers : int
        Number of RNN layers to use.
    """

    def __init__(self, input_size, contex=0, rnn_size=128, projection=64, layers=2):
        super().__init__()
        self.layers = torch.nn.ModuleList()

        contex_size = input_size * (2 * contex + 1)

        # Alternate RNN and projection layers.
        for i in range(layers):
            self.layers.append(
                torch.nn.GRU(
                    input_size=contex_size if i == 0 else projection,
                    hidden_size=rnn_size,
                    bidirectional=True,
                )
            )

            # Projection layer reduces size, except last layer, which
            # goes back to input size to create the mask
            linear_size = input_size if i == layers - 1 else projection
            self.layers.append(
                torch.nn.Linear(
                    in_features=rnn_size * 2, out_features=linear_size,
                )
            )

        # Use ReLU to make sure outputs aren't negative (unhelpful for masking)
        self.layers.append(torch.nn.ReLU())

    def forward(self, x):
        """Shift to time-first, pass layers, then back to batch-first."""
        x = x.transpose(0, 1)
        for layer in self.layers:
            x = layer(x)
            if isinstance(x, tuple):
                x = x[0]
        x = x.transpose(0, 1)
        # x = torch.clamp(x, min=0, max=10)
        return x

if __name__ == "__main__":
    input_size = 257
    contex = 3
    model = CustomModel(input_size, contex=contex)
    # input_data = torch.rand(100, 20, input_size)
    from torchsummary import summary
    summary(model, (20, input_size * (2 * contex + 1)))
