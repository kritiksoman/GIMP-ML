import torch
import torch.nn as nn
from functools import reduce
from torch.autograd import Variable


class shave_block(nn.Module):
    def __init__(self, s):
        super(shave_block, self).__init__()
        self.s=s
    def forward(self,x):
        return x[:,:,self.s:-self.s,self.s:-self.s]

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

def generator():
    G = nn.Sequential( # Sequential,
        nn.ReflectionPad2d((40, 40, 40, 40)),
        nn.Conv2d(1,32,(9, 9),(1, 1),(4, 4)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    ),
                shave_block(2),
                ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    ),
                shave_block(2),
                ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    ),
                shave_block(2),
                ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    ),
                shave_block(2),
                ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.Sequential( # Sequential,
            LambdaMap(lambda x: x, # ConcatTable,
                nn.Sequential( # Sequential,
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.Conv2d(128,128,(3, 3)),
                    nn.BatchNorm2d(128),
                    ),
                shave_block(2),
                ),
            LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
        nn.ConvTranspose2d(128,64,(3, 3),(2, 2),(1, 1),(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.ConvTranspose2d(64,32,(3, 3),(2, 2),(1, 1),(1, 1)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32,2,(9, 9),(1, 1),(4, 4)),
        nn.Tanh(),
    )
    return G