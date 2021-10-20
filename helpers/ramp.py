import numpy as np
from ba3l.ingredients.ingredient import Ingredient


# credit: https://github.com/iBelieveCJM/Tricks-of-Semi-supervisedDeepLeanring-Pytorch/blob/master/utils/ramps.py


def pseudo_rampup(T1, T2):
    def warpper(epoch):
        if epoch > T1:
            alpha = (epoch - T1) / (T2 - T1)
            if epoch > T2:
                alpha = 1.0
        else:
            alpha = 0.0
        return alpha

    return warpper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper


def linear_rampup(rampup_length):
    """Linear rampup"""

    def warpper(epoch):
        if epoch < rampup_length:
            return epoch / rampup_length
        else:
            return 1.0

    return warpper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    """Linear rampup -(start)- (rampdown_length) \ _(for the rest)  """
    def warpper(epoch):
        if epoch <= start:
            return 1.
        elif epoch - start < rampdown_length:
            return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value
    return warpper


def exp_rampdown(rampdown_length, num_epochs):
    """Exponential rampdown from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5 * (epoch - (num_epochs - rampdown_length))
            return float(np.exp(-(ep * ep) / rampdown_length))
        else:
            return 1.0

    return warpper


def cosine_rampdown(rampdown_length, num_epochs):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""

    def warpper(epoch):
        if epoch >= (num_epochs - rampdown_length):
            ep = .5 * (epoch - (num_epochs - rampdown_length))
            return float(.5 * (np.cos(np.pi * ep / rampdown_length) + 1))
        else:
            return 1.0

    return warpper


def exp_warmup(rampup_length, rampdown_length, num_epochs):
    rampup = exp_rampup(rampup_length)
    rampdown = exp_rampdown(rampdown_length, num_epochs)

    def warpper(epoch):
        return rampup(epoch) * rampdown(epoch)

    return warpper


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)
    def warpper(epoch):
        return rampup(epoch) * rampdown(epoch)
    return warpper


def test_warmup():
    warmup = exp_warmup(20, 100, 150)
    for ep in range(500):
        print(warmup(ep))


def test_warmupl():
    warmup = exp_warmup_linear_down(20, 100, 50, 0.001)
    for ep in range(500):
        print(warmup(ep))


def cosine_cycle(cycle_len=20,ramp_down_start=100,last_lr_value=0.01):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    ramp_down_start = cycle_len+ (ramp_down_start-1)//cycle_len*(cycle_len)
    print("adjusted ramp_down_start:",ramp_down_start)
    def warpper(epoch):
        ep =  (epoch+cycle_len//2.)/(1.*cycle_len)
        if epoch>ramp_down_start:
            return last_lr_value
        return float(last_lr_value + (1.-last_lr_value)* .5 * (np.cos(2.*np.pi * ep) + 1))
    return warpper


if __name__ == '__main__':
    test= exp_warmup_linear_down(20, 100, 50, 150)
    for i in range(250):
        print(test(i))