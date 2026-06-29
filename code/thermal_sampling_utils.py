from typing import Tuple
import numpy as np

def box_muller_transform(mu: float = 0.0, sigma: float = 1.0) -> Tuple[float, float]:
    """
    Generate a single pair of independent normal random variables with a specified 
    mean (mu) and standard deviation (sigma).
    This version is modified to closely match the behavior of the TDEP Fortran 
    'boxmuller' subroutine, which generates one pair at a time and includes scaling.

    Args:
        mu: The desired mean of the normal random variables. Defaults to 0.0.
        sigma: The desired standard deviation of the normal random variables. 
               Must be non-negative. Defaults to 1.0.
        
    Returns:
        A tuple containing two float values (x1, x2) drawn from a normal 
        distribution with the specified mean and standard deviation.
        x1 is generated using the sine component and x2 using the cosine component,
        to align with the Fortran TDEP boxmuller subroutine's assignment of z1 (sin) and z2 (cos).
    """
    if sigma < 0:
        raise ValueError("Standard deviation (sigma) cannot be negative.")

    # Generate uniform random numbers
    u1 = 0.0
    while u1 == 0.0:
        u1 = np.random.random()
    u2 = np.random.random()
    
    # Standard Box-Muller transformation components
    # r_val corresponds to sqrt(-2 * np.log(u1))
    r_val = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    
    # Generate standard normal variates (mean 0, std 1)
    std_normal_variate1 = r_val * np.sin(theta)
    std_normal_variate2 = r_val * np.cos(theta) 
    
    # Scale to the desired mean (mu) and standard deviation (sigma)
    x1 = mu + std_normal_variate1 * sigma
    x2 = mu + std_normal_variate2 * sigma
    
    return x1, x2