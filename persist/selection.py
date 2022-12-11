import torch
import numpy as np
import torch.nn as nn
from persist import models, utils


class PERSIST:
    '''
    Class for using the predictive and robust gene selection for spatial
    transcriptomics (PERSIST) method.

    Args:
      train_dataset: dataset of training examples (ExpressionDataset).
      val_dataset: dataset of validation examples.
      loss_fn: loss function, such as HurdleLoss(), nn.MSELoss() or
        nn.CrossEntropyLoss().
      device: torch device, such as torch.device('cuda', 0).
      preselected_inds: list of indices that must be selected.
      hidden: number of hidden units per layer (list of ints).
      activation: activation function between layers.
    '''
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 loss_fn,
                 device,
                 preselected_inds=[],
                 hidden=[128, 128],
                 activation=nn.ReLU()):
        # TODO add verification for dataset type.
        self.train = train_dataset
        self.val = val_dataset
        self.loss_fn = loss_fn

        # Architecture parameters.
        self.hidden = hidden
        self.activation = activation

        # Set device.
        assert isinstance(device, torch.device)
        self.device = device

        # Set preselected genes.
        self.preselected = np.sort(preselected_inds).astype(int)

        # Initialize candidate genes.
        self.set_genes()

    def get_genes(self):
        '''Get currently selected genes, not including preselected genes.'''
        return self.candidates

    def set_genes(self, candidates=None):
        '''Restrict the subset of genes.'''
        if candidates is None:
            # All genes but pre-selected ones.
            candidates = np.array(
                [i for i in range(self.train.max_input_size)
                 if i not in self.preselected])
        
        else:
            # Ensure that candidates do not overlap with pre-selected genes.
            assert len(np.intersect1d(candidates, self.preselected)) == 0
        self.candidates = candidates

        # Set genes in datasets.
        included = np.sort(np.concatenate([candidates, self.preselected]))
        self.train.set_inds(included)
        self.val.set_inds(included)

        # Set relative indices for pre-selected genes.
        self.preselected_relative = np.array(
            [np.where(included == ind)[0][0] for ind in self.preselected])

    def eliminate(self,
                  target,
                  lam_init=None,
                  mbsize=64,
                  max_nepochs=250,
                  lr=1e-3,
                  tol=0.2,
                  start_temperature=10.0,
                  end_temperature=0.01,
                  optimizer='Adam',
                  lookback=10,
                  max_trials=10,
                  bar=True,
                  verbose=False):
        '''
        Narrow the set of candidate genes: train a model with the BinaryGates
        layer and annealed penalty to eliminate a large portion of the inputs.

        Args:
          target: target number of genes to select (in addition to pre-selected
            genes).
          lam_init: initial lambda value.
          mbsize: minibatch size.
          max_nepochs: maximum number of epochs.
          lr: learning rate.
          tol: tolerance around gene target number.
          start_temperature: starting temperature for BinConcrete samples.
          end_temperature: final temperature value.
          optimizer: optimization algorithm.
          lookback: number of epochs to wait for improvement before stopping.
          max_trials: maximum number of training rounds before returning an
            error.
          bar: whether to display tqdm progress bar for each round of training.
          verbose: verbosity.
        '''
        # Reset candidate genes.
        all_inds = np.arange(self.train.max_input_size)
        all_candidates = np.array_equal(
            self.candidates, np.setdiff1d(all_inds, self.preselected))
        all_train_inds = np.array_equal(self.train.inds, all_inds)
        all_val_inds = np.array_equal(self.val.inds, all_inds)
        if not (all_candidates and all_train_inds and all_val_inds):
            print('resetting candidate genes')
            self.set_genes()

        # Initialize architecture.
        if isinstance(self.loss_fn, utils.HurdleLoss):
            output_size = 2 * self.train.output_size
        elif isinstance(self.loss_fn, (nn.CrossEntropyLoss, nn.MSELoss)):
            output_size = self.train.output_size
        else:
            output_size = self.train.output_size
            print(f'Unknown loss function, assuming {self.loss_fn} requires '
                  f'{self.train.output_size} outputs')

        model = models.SelectorMLP(input_layer='binary_gates',
                                   input_size=self.train.input_size,
                                   output_size=output_size,
                                   hidden=self.hidden,
                                   activation=self.activation,
                                   preselected_inds=self.preselected_relative)
        model = model.to(self.device)

        # Determine lam_init, if necessary.
        if lam_init is None:
            if isinstance(self.loss_fn, utils.HurdleLoss):
                print('using HurdleLoss, starting with lam = 0.01')
                lam_init = 0.01
            elif isinstance(self.loss_fn, nn.MSELoss):
                print('using MSELoss, starting with lam = 0.01')
                lam_init = 0.01
            elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
                print('using CrossEntropyLoss, starting with lam = 0.0001')
                lam_init = 0.0001
            else:
                print('unknown loss function, starting with lam = 0.0001')
                lam_init = 0.0001
        else:
            print(f'trying lam = {lam_init:.6f}')

        # Prepare for training and lambda search.
        assert 0 < target < self.train.input_size
        assert 0.1 <= tol < 0.5
        assert lam_init > 0
        lam_list = [0]
        num_remaining = self.train.input_size
        num_remaining_list = [num_remaining]
        lam = lam_init
        trials = 0

        # Iterate until num_remaining is near the target value.
        while np.abs(num_remaining - target) > target * tol:
            # Ensure not done.
            if trials == max_trials:
                raise ValueError(
                    'reached maximum number of trials without selecting the '
                    'desired number of genes! The results may have large '
                    'variance due to small dataset size, or the initial lam '
                    'value may be bad')
            trials += 1

            # Train.
            model.fit(self.train,
                      self.val,
                      lr,
                      mbsize,
                      max_nepochs,
                      start_temperature=start_temperature,
                      end_temperature=end_temperature,
                      loss_fn=self.loss_fn,
                      lam=lam,
                      optimizer=optimizer,
                      lookback=lookback,
                      bar=bar,
                      verbose=verbose)

            # Extract inds.
            inds = model.input_layer.get_inds(threshold=0.5)
            num_remaining = len(inds)
            print(f'lam = {lam:.6f} yielded {num_remaining} genes')

            if np.abs(num_remaining - target) <= target * tol:
                print(f'done, lam = {lam:.6f} yielded {num_remaining} genes')

            else:
                # Guess next lam value.
                next_lam = modified_secant_method(
                    lam, 1 / (1 + num_remaining), 1 / (1 + target),
                    np.array(lam_list), 1 / (1 + np.array(num_remaining_list)))

                # Clip lam value for stability.
                next_lam = np.clip(next_lam, a_min=0.1 * lam, a_max=10 * lam)

                # Possibly reinitialize model.
                if num_remaining < target * (1 - tol):
                    # BinaryGates layer is not great at allowing features
                    # back in after inducing too much sparsity.
                    print('Reinitializing model for next iteration')
                    model = models.SelectorMLP(
                        input_layer='binary_gates',
                        input_size=self.train.input_size,
                        output_size=output_size,
                        hidden=self.hidden,
                        activation=self.activation,
                        preselected_inds=self.preselected_relative)
                    model = model.to(self.device)
                else:
                    print('Warm starting model for next iteration')

                # Prepare for next iteration.
                lam_list.append(lam)
                num_remaining_list.append(num_remaining)
                lam = next_lam
                print(f'next attempt is lam = {lam:.6f}')

        # Set eligible genes.
        true_inds = self.candidates[inds]
        self.set_genes(true_inds)
        return true_inds, model

    def select(self,
               num_genes,
               mbsize=64,
               max_nepochs=250,
               lr=1e-3,
               start_temperature=10.0,
               end_temperature=0.01,
               optimizer='Adam',
               bar=True,
               verbose=False):
        '''
        Select genetic probes: train a model with BinaryMask layer to select
        a precise number of model inputs.

        Args:
          num_genes: number of genes to select (in addition to pre-selected
            genes).
          mbsize: minibatch size.
          max_nepochs: maximum number of epochs.
          lr: learning rate.
          start_temperature: starting temperature value for Concrete samples.
          end_temperature: final temperature value.
          optimizer: optimization algorithm.
          bar: whether to display tqdm progress bar.
          verbose: verbosity.
        '''
        # Possibly reset candidate genes.
        included_inds = np.sort(
            np.concatenate([self.candidates, self.preselected]))
        candidate_train_inds = np.array_equal(self.train.inds, included_inds)
        candidate_val_inds = np.array_equal(self.val.inds, included_inds)
        if not (candidate_train_inds and candidate_val_inds):
            print('setting candidate genes in datasets')
            self.set_genes(self.candidates)

        # Initialize architecture.
        if isinstance(self.loss_fn, utils.HurdleLoss):
            output_size = 2 * self.train.output_size
        elif isinstance(self.loss_fn, (nn.CrossEntropyLoss, nn.MSELoss)):
            output_size = self.train.output_size
        else:
            output_size = self.train.output_size
            print(f'assuming loss function {self.loss_fn} requires '
                  f'{self.train.output_size} outputs')

        input_size = len(self.candidates) + len(self.preselected)
        model = models.SelectorMLP(input_layer='binary_mask',
                                   input_size=input_size,
                                   output_size=output_size,
                                   hidden=self.hidden,
                                   activation=self.activation,
                                   preselected_inds=self.preselected_relative,
                                   num_selections=num_genes).to(self.device)

        # Train.
        model.fit(self.train,
                  self.val,
                  lr,
                  mbsize,
                  max_nepochs,
                  start_temperature,
                  end_temperature,
                  loss_fn=self.loss_fn,
                  optimizer=optimizer,
                  bar=bar,
                  verbose=verbose)

        # Return genes.
        inds = model.input_layer.get_inds()
        true_inds = self.candidates[inds]
        print(f'done, selected {len(inds)} genes')
        return true_inds, model


def modified_secant_method(x0, y0, y1, x, y):
    '''
    A modified version of secant method, used here to determine the correct lam
    value. Note that we use x = lam and y = 1 / (1 + num_remaining) rather than
    y = num_remaining, because this gives better results.

    The standard secant method uses the two previous points to calculate a
    finite difference rather than an exact derivative (as in Newton's method).
    Here, we used a robustified derivative estimator: we find the curve,
    which passes through the most recent point (x0, y0), that minimizes a
    weighted least squares loss for all previous points (x, y). This improves
    robustness to nearby guesses (small |x - x'|) and noisy evaluations.

    Args:
      x0: most recent x.
      y0: most recent y.
      y1: target y value.
      x: all previous xs.
      y: all previous ys.
    '''
    # Get robust slope estimate.
    weights = 1 / np.abs(x - x0)
    slope = (
        np.sum(weights * (x - x0) * (y - y0)) /
        np.sum(weights * (x - x0) ** 2))

    # Clip slope to minimum value.
    slope = np.clip(slope, a_min=1e-6, a_max=None)

    # Guess x1.
    x1 = x0 + (y1 - y0) / slope
    return x1
