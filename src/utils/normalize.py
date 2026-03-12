import numpy as np

def normalize(x, min_val, max_val):
        """
        Normalize x from [min_val, max_val] to [-1, 1].
        
        x:          Value to normalize
        min_val:    Minimum value of range
        max_val:    Maximum value of range
        
        Returns:
            Normalized value in [-1, 1]
        """
        return np.clip(2 * (x - min_val) / (max_val - min_val) - 1, -1, 1)