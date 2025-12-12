"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description="Run HCMEIS.")

    parser.add_argument("--expert",
                        type=int,
                        default=3,
                        help="class of the expert experiment. Default is 0.")

    parser.add_argument("--epochs",
                        type=int,
                        default=500,
                        help="Number of training epochs. Default is 500.")

    parser.add_argument("--filters-1",
                        type=int,
                        default=128,
                        help="Filters (neurons) in 1st convolution. Default is 128.")

    parser.add_argument("--filters-2",
                        type=int,
                        default=64,
                        help="Filters (neurons) in 2nd convolution. Default is 64.")

    parser.add_argument("--filters-3",
                        type=int,
                        default=16,
                        help="Filters (neurons) in 3rd convolution. Default is 16.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
                        help="Neurons in tensor network layer. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0,
                        help="Dropout probability. Default is 0.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
                        help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=  10 ** -5,
                        help="Adam weight decay. Default is 10^-5.")

    return parser.parse_args()
