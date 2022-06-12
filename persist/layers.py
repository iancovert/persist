import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp_probs(probs):
    '''Clamp probabilities to ensure stable logs.'''
    eps = torch.finfo(probs.dtype).eps
    return torch.clamp(probs, min=eps, max=1-eps)


def concrete_sample(logits, temperature, shape=torch.Size([])):
    '''
    Sampling for Concrete distribution (see eq. 10 of Maddison et al., 2017).

    Args:
      logits: Concrete logits parameters.
      temperature: Concrete temperature parameter.
      shape: sample shape.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    gumbels = - torch.log(- torch.log(u))
    scores = (logits + gumbels) / temperature
    return scores.softmax(dim=-1)


def bernoulli_concrete_sample(logits, temperature, shape=torch.Size([])):
    '''
    Sampling for BinConcrete distribution (see PyTorch source code, differs
    slightly from eq. 16 of Maddison et al., 2017).

    Args:
       logits: tensor of BinConcrete logits parameters.
       temperature: BinConcrete temperature parameter.
       shape: sample shape.
    '''
    uniform_shape = torch.Size(shape) + logits.shape
    u = clamp_probs(torch.rand(uniform_shape, dtype=torch.float32,
                               device=logits.device))
    return torch.sigmoid((F.logsigmoid(logits) - F.logsigmoid(-logits)
                          + torch.log(u) - torch.log(1 - u)) / temperature)


class BinaryMask(nn.Module):
    '''
    Input layer that selects features by learning a k-hot mask.

    Args:
      input_size: number of inputs.
      num_selections: number of features to select.
      temperature: temperature for Concrete samples.
      gamma: used to map learned parameters to logits (helps convergence speed).
    '''

    def __init__(self,
                 input_size,
                 num_selections,
                 temperature=10.0,
                 gamma=1/3):
        super().__init__()
        self._logits = nn.Parameter(
            torch.zeros(num_selections, input_size, dtype=torch.float32,
                        requires_grad=True))
        self.input_size = input_size
        self.num_selections = num_selections
        self.output_size = input_size
        self.temperature = temperature
        self.gamma = gamma

    @property
    def logits(self):
        return self._logits / self.gamma

    @property
    def probs(self):
        return (self.logits).softmax(dim=1)

    def sample(self, n_samples):
        '''Sample approximate k-hot vectors.'''
        samples = concrete_sample(
            self.logits, self.temperature, torch.Size([n_samples]))
        return torch.max(samples, dim=-2).values

    def forward(self, x):
        '''Sample and apply mask.'''
        m = self.sample(len(x))
        x = x * m
        return x, m

    def get_inds(self):
        '''Get selected indices.'''
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return (f'input_size={self.input_size}, temperature={self.temperature},'
                f' num_selections={self.num_selections}')


class BinaryGates(nn.Module):
    '''
    Input layer that selects features by learning binary gates for each feature,
    similar to [1].

    [1] Dropout Feature Ranking for Deep Learning Models (Chang et al., 2017)

    Args:
      input_size: number of inputs.
      temperature: temperature for BinConcrete samples.
      init: initial value for each gate's probability of being 1.
      gamma: used to map learned parameters to logits (helps convergence speed).
    '''

    def __init__(self,
                 input_size,
                 temperature=0.1,
                 init=0.99,
                 gamma=1/2):
        super().__init__()
        init_logit = - torch.log(1 / torch.tensor(init) - 1) * gamma
        self._logits = nn.Parameter(torch.full(
            (input_size,), init_logit, dtype=torch.float32, requires_grad=True))
        self.input_size = input_size
        self.output_size = input_size
        self.temperature = temperature
        self.gamma = gamma

    @property
    def logits(self):
        return self._logits / self.gamma
    
    @property
    def probs(self):
        return torch.sigmoid(self.logits)

    def sample(self, n_samples):
        '''Sample approximate binary masks.'''
        return bernoulli_concrete_sample(
            self.logits, self.temperature, torch.Size([n_samples]))

    def forward(self, x):
        '''Sample and apply mask.'''
        m = self.sample(len(x))
        x = x * m
        return x, m

    def get_inds(self, num_features=None, threshold=None):
        '''
        Get selected indices.

        Args:
          num_features: number of top features to return.
          threshold: probability threshold for determining selected features.
        '''
        if (num_features is None) == (threshold is None):
            raise ValueError('exactly one of num_features and threshold must be'
                             ' specified')

        if num_features:
            inds = torch.argsort(self.probs)[-num_features:]
        elif threshold:
            inds = (self.probs > threshold).nonzero()[:, 0]
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return f'input_size={self.input_size}, temperature={self.temperature}'


class ConcreteSelector(nn.Module):
    '''
    Input layer that selects features by learning a binary matrix, based on [2].
    
    [2] Concrete Autoencoders for Differentiable Feature Selection and
    Reconstruction (Balin et al., 2019)

    Args:
      input_size: number of inputs.
      num_selections: number of features to select.
      temperature: temperature for Concrete samples.
      gamma: used to map learned parameters to logits (helps convergence speed).
    '''

    def __init__(self,
                 input_size,
                 num_selections,
                 temperature=10.0,
                 gamma=1/3):
        super().__init__()
        self._logits = nn.Parameter(
            torch.zeros(num_selections, input_size, dtype=torch.float32,
                        requires_grad=True))
        self.input_size = input_size
        self.num_selections = num_selections
        self.output_size = num_selections
        self.temperature = temperature
        self.gamma = gamma

    @property
    def logits(self):
        return self._logits / self.gamma

    @property
    def probs(self):
        return self.logits.softmax(dim=1)

    def sample(self, n_samples):
        '''Sample approximate binary matrices.'''
        return concrete_sample(
            self.logits, self.temperature, torch.Size([n_samples]))

    def forward(self, x):
        '''Sample and apply selector matrix.'''
        M = self.sample(len(x))
        x = torch.matmul(M, x.unsqueeze(2)).squeeze(2)
        return x, M

    def get_inds(self):
        '''Get selected indices.'''
        inds = torch.argmax(self.logits, dim=1)
        return torch.sort(inds)[0].cpu().data.numpy()

    def extra_repr(self):
        return (f'input_size={self.input_size}, temperature={self.temperature},'
                f' num_selections={self.num_selections}')
