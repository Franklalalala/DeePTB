"""Real FFT gradient/divergence in the selected charge plane-wave subspace."""
def derivative(values, gvec, mask, axis):
    import numpy as np
    if type(values).__module__.startswith('torch'):
        import torch
        coeff=torch.fft.fftn(values)
        return torch.fft.ifftn(1j*gvec[...,axis]*torch.where(mask,coeff,0.)).real
    coeff=np.fft.fftn(values)
    return np.fft.ifftn(1j*gvec[...,axis]*np.where(mask,coeff,0.)).real
