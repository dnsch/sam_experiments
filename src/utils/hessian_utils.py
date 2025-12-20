from pathlib import Path
import sys
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parents[2]))

from lib.utils.pyhessian.pyhessian import hessian


# Calculate max Eigenvalue of Hessian (= sharpness of the loss landscape)
# using PyHessian
# https://github.com/amirgholami/PyHessian/tree/master
def compute_top_eigenvalue_and_eigenvector(model, criterion, data_loader):
    model.eval()
    hessian_comp = hessian(
        model, criterion, dataloader=data_loader, cuda=torch.cuda.is_available()
    )

    # Compute top eigenvalue
    top_eigenvalue, top_eigenvector = hessian_comp.eigenvalues(top_n=1)

    return top_eigenvalue, top_eigenvector
