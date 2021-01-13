#!/usr/bin/env python 

import numpy as np


class Chimera:

    def __init__(self, relatives, absolutes, softness=1e-3):
        """
        Hierarchy-based scalarizing function for multi-objective optimization. The user can obtain a single
        scalarizing function from a hierarchy of objectives and their associated relative or absolute thresholds.

        Parameters
        ----------
        relatives : list
            list of relative thresholds, within [0,1], for each objective.
        absolutes : list
            list of absolute thresholds for each objective.
        softness : float, optional
            Smoothing parameter. Default is 0.001.
        """
        self.relatives = relatives
        self.absolutes = absolutes
        self.softness  = softness

    def _soft_step(self, value):
        arg = - value / self.softness
        return np.exp(- np.logaddexp(0, arg))

    def _hard_step(self, value):
        result = np.where(value > 0., 1., 0.)
        return result

    def _step(self, value):
        if self.softness < 1e-5:
            return self._hard_step(value)
        else:
            return self._soft_step(value)

    def _rescale(self, objs):
        """ rescales objectives and absolute threshols such that all
            observed objectives are within [0, 1]
        """
        _objectives = np.empty(objs.shape)
        _absolutes  = np.empty(self.absolutes.shape)
        for idx in range(objs.shape[1]):
            min_obj, max_obj = np.amin(objs[:, idx]), np.amax(objs[:, idx])
            if min_obj < max_obj:
                _objectives[:, idx] = (objs[:, idx] - min_obj) / (max_obj - min_obj)
                _absolutes[idx]  = (self.absolutes[idx] - min_obj) / (max_obj - min_obj)
            else:
                _objectives[:, idx] = objs[:, idx] - min_obj
                _absolutes[idx]  = self.absolutes[idx] - min_obj
        return _objectives, _absolutes

    def _shift(self, _objectives, _absolutes):
        """ shift rescaled objectives based on identified regions of
            interest
        """
        _transposed_objs = _objectives.transpose()
        shapes           = _transposed_objs.shape
        _shifted_objs    = np.empty((shapes[0] + 1, shapes[1]))

        mins, maxs = [], []
        thresholds = []
        domain     = np.arange(shapes[1])
        shift      = 0

        for idx, obj in enumerate(_transposed_objs):

            # get absolute thresholds
            minimum = np.amin(obj[domain])
            maximum = np.amax(obj[domain])

            mins.append(minimum)
            maxs.append(maximum)

            if np.isnan(self.relatives[idx]):
                threshold = _absolutes[idx]
            else:
                threshold = minimum + self.relatives[idx] * (maximum - minimum)

            # adjust to region of interest
            interest = np.where(obj[domain] < threshold)[0]
            if len(interest) > 0:
                domain = domain[interest]

            # apply shift
            thresholds.append(threshold + shift)
            _shifted_objs[idx] = _transposed_objs[idx] + shift

            # compute new shift
            if idx < len(_transposed_objs) - 1:
                shift -= np.amax(_transposed_objs[idx + 1][domain]) - threshold
            else:
                shift -= np.amax(_transposed_objs[0][domain]) - threshold
                _shifted_objs[idx + 1] = _transposed_objs[0] + shift
        return _shifted_objs, thresholds

    def _scalarize(self, _shifted_objs, thresholds):
        _merits = _shifted_objs[-1].copy()
        for idx in range(0, len(_shifted_objs) - 1)[::-1]:
            _merits *= self._step( - _shifted_objs[idx] + thresholds[idx])
            _merits += self._step(   _shifted_objs[idx] - thresholds[idx]) * _shifted_objs[idx]
        return _merits.transpose()

    def scalarize(self, objs):
        """Scalarize the objectives.

        Parameters
        ----------
        objs : array
            Array of ...

        Returns
        -------
        merits : array
            blablabla
        """
        _objectives, _absolutes   = self._rescale(objs)
        _shifted_objs, thresholds = self._shift(_objectives, _absolutes)
        merits = self._scalarize(_shifted_objs, thresholds)
        if np.amax(merits) > 0.:
            merits = (merits - np.amin(merits)) / (np.amax(merits) - np.amin(merits))
        return merits

