#!/usr/bin/env python 

import numpy as np


class Chimera:

    def __init__(self, tolerances, absolutes=None, percentiles=None, goals=None, softness=1e-3):
        """
        Hierarchy-based scalarizing function for multi-objective optimization. The user can obtain a single
        scalarizing function from a hierarchy of objectives and their associated relative or absolute thresholds.

        Parameters
        ----------
        tolerances : list
            list of tolerances for each objective. The order of these tolerances should reflect
            the hierarchy of the objectives, such that the first element of the list should be the tolerance for the
            first objective, the second element the tolerance for the second objective, and so on. By default, relative
            tolerance (within [0,1]) are expected. If you would like to provide absolute tolerances, you need to
            set pass ``True`` to the ``absolute`` argument.
        percentiles : list
            list indicating whether the corresponding thresholds are expected to be treated as percentiles rather than 
            fractions. If ``True``, the corresponding relative threshold is not set as a fraction of the co-domain, but
            as the nth percentile of the observed objectives. Default is ``False`` for all tolerances.
        absolutes : list
            list indicating whether the corresponding thresholds are absolute as opposed to relative ones. Default
            is ``False`` for all tolerances.
        goals : list
            list of optimization goals. By default, it is assumed that all objectives are being minimized
            ('min'). If some objectives are to be maximized ('max'), you need to specify it with this argument.
            E.g. if we have a hierarchy of 2 objectives, where the maximize the first and minimize the second,
            you should pass ['max', 'min'] to this argument.
        softness : float
            Smoothing parameter. Default is 0.001.
        """

        # check input
        # -----------
        if absolutes is not None:
            # check same length as tolerances
            if len(absolutes) != len(tolerances):
                raise ValueError('`tolerances` and `absolutes` should be lists of the same length')
            # check only True/False in absolute
            for b in absolutes:
                if not isinstance(b, bool):
                    raise ValueError("`absolutes` should be a list of True/False values")

        if percentiles is not None:
            # check same length as tolerances
            if len(percentiles) != len(tolerances):
                raise ValueError('`tolerances` and `percentiles` should be lists of the same length')
            # check only True/False in absolute
            for b in percentiles:
                if not isinstance(b, bool):
                    raise ValueError("`percentiles` should be a list of True/False values")

        if goals is not None:
            if len(goals) != len(tolerances):
                raise ValueError('`tolerances` and `goals` should be lists of the same length')
            for goal in goals:
                if goal not in ['min', 'max']:
                    raise ValueError("`goals` can only contain 'min' or 'max'")

        # attributes
        # ----------
        self.tolerances = np.array(tolerances)
        self.softness = softness

        if goals is None:
            self.goals = ['min'] * len(self.tolerances)
        else:
            self.goals = goals

        if absolutes is None:
            self.absolutes = [False] * len(self.tolerances)
        else:
            self.absolutes = absolutes

        if percentiles is None:
            self.percentiles = [False] * len(self.tolerances)
        else:
            self.percentiles = percentiles

        # check that we do not have both absolutes and percentiles defined
        for b1, b2 in zip(self.absolutes, self.percentiles):
            if b1 is True and b2 is True:
                raise ValueError('a tolerance value cannot be defined as both an absolute and a percentile tolerance\n',
                                 'make sure that `absolutes` and `percentiles` do not have any same element set to `True`')

        # check that all relative tolerances are in [0,1]
        for b, t in zip(self.absolutes, self.tolerances):
            if b is False:
                if t > 1. or t < 0.:
                    raise ValueError('relative tolerances need to be between 0 and 1. If you would like to use '
                                     'absolute tolerances, you should set the corresponding element of '
                                     '"absolutes" to True')

        # internal attrs (useful for inspection/debugging)
        # ------------------------------------------------
        self._objs = None
        self._scaled_objs = None
        self._shifted_objs = None
        self._thresholds = None
        self._scaled_thresholds = None
        self._shifted_thresholds = None

    def _soft_step(self, value):
        arg = - value / self.softness
        return np.exp(- np.logaddexp(0, arg))

    @staticmethod
    def _hard_step(value):
        result = np.where(value >= 0., 1., 0.)
        return result

    def _step(self, value):
        if self.softness < 1e-5:
            return self._hard_step(value)
        else:
            return self._soft_step(value)

    @staticmethod
    def _invert_binary(a):
        return np.abs(np.array(a) - 1.)

    def _adjust_objectives(self, objs):
        """adjust objectives based on optimization goal"""
        adjusted_objs = np.empty(objs.shape)
        adjusted_thre = np.empty(self.tolerances.shape)

        for i, obj_goal in enumerate(self.goals):
            if obj_goal == 'min':
                adjusted_objs[:, i] = objs[:, i]
                adjusted_thre[i] = self.tolerances[i]
            elif obj_goal == 'max':
                adjusted_objs[:, i] = - objs[:, i]
                # if absolute tolerance and max, we invert threshold
                if self.absolutes[i] is True:
                    adjusted_thre[i] = - self.tolerances[i]
                else:
                    adjusted_thre[i] = self.tolerances[i]

        return adjusted_objs, adjusted_thre

    def _rescale_objs_and_thres(self, objs, thres):
        """ rescales objectives and absolute threshols such that all
            observed objectives are within [0, 1]
        """
        _objectives = np.empty(objs.shape)
        _thresholds = np.empty(thres.shape)

        # iterate over objectives
        for idx in range(objs.shape[1]):
            min_obj, max_obj = np.amin(objs[:, idx]), np.amax(objs[:, idx])
            # to avoid division by zero, check max-min > 0
            if min_obj < max_obj:
                _objectives[:, idx] = (objs[:, idx] - min_obj) / (max_obj - min_obj)
                # scale _thresholds only if absolute, otherwise skip
                if self.absolutes[idx] is True:
                    _thresholds[idx] = (thres[idx] - min_obj) / (max_obj - min_obj)
                else:
                    _thresholds[idx] = thres[idx]
            # if all objs values are the same, simply shift values to zero (and absolute threshold accordingly)
            else:
                _objectives[:, idx] = objs[:, idx] - min_obj
                # scale _thresholds only if absolute, otherwise skip
                if self.absolutes[idx] is True:
                    _thresholds[idx] = thres[idx] - min_obj
                else:
                    _thresholds[idx] = thres[idx]

        return _objectives, _thresholds

    def _shift(self, objectives, thresholds):
        """ shift rescaled objectives based on identified regions of
            interest
        """
        transposed_objs = objectives.transpose()  # the objectives
        domain = np.arange(transposed_objs.shape[1])
        
        # we return the shifted objectives, with additional obj that repeats the first obj
        # and the shifted thresholds that go with those objectives
        shifted_objs = []
        shifted_thresholds = []  # the shifted thresholds
        
        # the first objective is not shifted
        shift = 0.            
        shifted_objs.append(transposed_objs[0])

        for idx, obj in enumerate(transposed_objs):

            # adapt relative thresholds/tolerances according to domain of interest
            if self.absolutes[idx] is False:
                if self.percentiles[idx] is False:
                    domain_min = np.min(obj[domain])
                    domain_max = np.max(obj[domain])
                    _threshold = domain_min + thresholds[idx] * (domain_max - domain_min)
                # if relative percentile tolerance, take percentile rather than fraction
                else:
                    _threshold = np.percentile(obj[domain], thresholds[idx] * 100., interpolation='linear')
            else:
                _threshold = thresholds[idx]
            
            # compute and append shifted thresholds
            shifted_threshold = _threshold - shift
            shifted_thresholds.append(shifted_threshold)

            # adjust to region of interest
            interest = np.where(obj[domain] < thresholds[idx])[0]
            if len(interest) > 0:
                domain = domain[interest]
            
            # compute new shift
            next_idx = (idx + 1) % transposed_objs.shape[0]  # i.e. loop back to idx == 0
            shift = np.amax(transposed_objs[next_idx][domain]) - np.min(shifted_thresholds)
            
            # apply shift and append to shifted objective
            shifted_obj = transposed_objs[next_idx] - shift
            shifted_objs.append(shifted_obj)
            
        return np.array(shifted_objs), np.array(shifted_thresholds)

    def _scalarize(self, shifted_objs, shifted_thres):
        # iterate from last objective to first
        merits = shifted_objs[-1].copy()
        for idx in range(0, len(shifted_objs) - 1)[::-1]:
            # first, make lower objective zero where there shuold be the higher objective
            merits *= self._invert_binary(self._step(shifted_objs[idx] - shifted_thres[idx]))
            # then, add the higher objective to the domain region that we just made equal to zero
            merits += self._step(shifted_objs[idx] - shifted_thres[idx]) * shifted_objs[idx]
        return merits.transpose()

    def scalarize(self, objs):
        """Scalarize the objectives.

        Parameters
        ----------
        objs : array
            Two-dimensional array containing the objective values for all samples collected. Each row should contain
            a different sample, and each column a different objective. The order of the columns should reflect the
            desired hierarchy of the objectives. Hence, ``objs[:, 0]`` should contain all values for the first
            objective, ``objs[:, 1]`` all values for the second objective, and so on.

        Returns
        -------
        merits : array
            One-dimensional array with the scalarized objective.
        """

        # adjust objectives, i.e. invert if maximizing
        self._objs, self._thresholds = self._adjust_objectives(np.array(objs))
        # normalize objectives (and absolute thresholds if present)
        self._scaled_objs, self._scaled_thresholds = self._rescale_objs_and_thres(self._objs, self._thresholds)
        # sort objectives and thresholds to minimize correctly
        self._shifted_objs, self._shifted_thresholds = self._shift(self._scaled_objs, self._scaled_thresholds)
        # scalarize objectives based on shifted objectives and thresholds
        merits = self._scalarize(self._shifted_objs, self._shifted_thresholds)
        # normalize chimera objective function
        if np.amax(merits) > 0.:
            merits = (merits - np.amin(merits)) / (np.amax(merits) - np.amin(merits))

        return merits

