import numpy as np
import pypolarshift.mapping as ppmap



def get_shift_mat(dim: int , k: int =1, complex: bool =False)->  np.ndarray:
    """
    Generate a shift matrix of specified dimension for circular shifting.

    Parameters:
        dim (int): The dimension of the square shift matrix to be created.
        k (int, optional): The number of positions to shift. Positive values result in rightward shifts, while negative values result in leftward shifts. Default is 1.
        complex (bool, optional): 
            If True, generates a complex exponential shift matrix. 
            If False, generates a standard circular shift matrix. Default is False.

    Returns:
        np.ndarray: A shift matrix of shape (dim, dim), either as a complex matrix or a standard circular shift matrix.
    """
    if complex:
        return  np.linalg.matrix_power(np.identity(dim) * np.array([np.exp( k*(2*np.pi*1j)/dim ) for k in range(dim)]), k)
    else:
        return np.linalg.matrix_power(np.roll(np.identity(dim), 1, 0), k)



def apply_shift(im: np.ndarray, k: int = 1, transform: str = 'vertical') -> tuple:
    """
    Apply a specified transformation to a 2D image, either by shifting, scaling, or rotating.

    Parameters:
        im (np.ndarray): Input 2D image to be transformed.
        k (int, optional): The number of positions to shift. Positive values result in rightward shifts, while negative values result in leftward shifts. Default is 1.
        transform (str): 
            Type of transformation to apply to the image. Must be one of the following:
            - 'vertical': Applies a vertical shift to the image.
            - 'horizontal': Applies a horizontal shift to the image.
            - 'scale': Applies a scaling transformation using polar coordinates.
            - 'rotate': Applies a rotational transformation using polar coordinates.

    Returns:
        tuple: 
            - Transformed image as a NumPy array.
            - If `transform` is 'scale' or 'rotate', returns a tuple containing:
                - The cartesian representation of the shifted image.
                - The polar representation of the shifted image.
              Else, returns cartesian representation of the shifted image.

    """

    n = im.shape[0]

    if transform == 'vertical':
        shift_mat = get_shift_mat(n, k)
        return np.einsum('ij, jk -> ik', shift_mat, im) 
    
    elif transform == 'horizental':
        shift_mat = get_shift_mat(n, k)
        return np.einsum('ij, jk -> ik', shift_mat, im.T).T 

    elif transform == 'scale':
        im_pol = ppmap.cartesian_to_polar(im, scaling='log')
        shift_mat = get_shift_mat(n, k)
        im_pol_shift = np.einsum('ij, jk -> ik', shift_mat, im_pol.T).T 
        return ppmap.polar_to_cartesian(im_pol_shift, scaling='log') , im_pol_shift

    elif transform == 'rotate':
        im_pol = ppmap.cartesian_to_polar(im, scaling='linear')
        alpha = int(im_pol.shape[0]/im_pol.shape[1])
        shift_mat = get_shift_mat(alpha*n, alpha*k)
        im_pol_shift = np.einsum('ij, jk -> ik', shift_mat, im_pol) 
        return ppmap.polar_to_cartesian(im_pol_shift, scaling='linear') , im_pol_shift
    
    else:
        raise ValueError("Transofrmation must be in {'vertical', 'horizental', 'scale', 'rotate'}")