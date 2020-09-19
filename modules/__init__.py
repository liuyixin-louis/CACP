
from .eltwise import *
from .grouping import *
from .matmul import *
from .rnn import *
from .aggregate import *
from .topology import *

__all__ = ['EltwiseAdd', 'EltwiseSub', 'EltwiseMult', 'EltwiseDiv', 'Matmul', 'BatchMatmul',
           'Concat', 'Chunk', 'Split', 'Stack',
           'LSTMCell', 'LSTM', 'convert_model_to_lstm',
           'Norm', 'Mean', 'BranchPoint', 'Print']


class Print(nn.Module):
    """Utility module to temporarily replace modules to assess activation shape.

    This is useful, e.g., when creating a new model and you want to know the size
    of the input to nn.Linear and you want to avoid calculating the shape by hand.
    """
    def forward(self, x):
        print(x.size())
        return x