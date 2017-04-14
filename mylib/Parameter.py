# -*- coding=UTF-8 -*-

"""

File name : Parameter.py

Creation date : 06-02-2016

Last modified :

Created by :

Purpose :

    Parameter class for Latin Hypercube Sample. Consists of two variables:
    lower and high, for sampling intervals. Includes sampling method.

Usage :

    alpha = Parameter(low=0.1, high=2)
    alpha.sample(10)
    
    Draws 10 samples from bounds.

Observations :

    Divides interval into 10 pieces, drawing from each piece at a time to
    ensure uniform coverage.

"""

import numpy as np

class Parameter():

    def __init__(self, name, low, high):

        self.low = low
        self.high = high
        self.name = name

    def sample(self, n):

        # n is the number of samples to be drawn

        # Arrays with low/high bounds
        low_ = np.zeros(10)
        high_ = np.zeros(10)
        samples = np.zeros(n)

        # Populating low_ and high_ and drawing samples
        for i in range(10):
            low_[i] = self.low + i*(self.high - self.low)/10
            high_[i] = self.low + (i+1)*(self.high - self.low)/10

            samples[n/10*i:n/10*(i+1)] = (high_[i] -\
                    low_[i])*np.random.random(n//10) + low_[i] 

        #return (self.high - self.low)*np.random.random(n) + self.low
        np.random.shuffle(samples)
        return samples

