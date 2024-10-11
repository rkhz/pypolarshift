import numpy as np
import math
import scipy

def cartesian_to_polar(image: np.ndarray, scaling: str ='linear', pad: int =3, eps: float =1e-12) -> np.ndarray:
    """
    Convert a 2D image from cartesian coordinates to polar coordinates.

    Parameters:
        image (np.ndarray): Input 2D image to be transformed.
        scaling (str): Type of scaling for the radius ('linear' or 'log').
                       - 'linear': Uses linear scaling for radius.
                       - 'log': Uses logarithmic scaling for radius.
        pad (int): Number of pixels to pad the image on each side. 
        eps (float): A small value to avoid log(0) in log scaling.

    Returns:
        np.ndarray: The polar mapped image.
    
    Equations:
        From Cartesian coordinates (x, y):
            Map to Polar coordinates (rho, theta, scaling == 'linear'):
                - x = rho * cose(theta)
                - y = rho * sin(theta)
        
            Map to Log-Polar coordinates (log_rho, theta, scaling == 'log'):
                - rho = exp(log log_rho)
                - x = rho * cose(theta)
                - y = rho * sin(theta)
    """

    center = np.array(image.shape)//2    
    radius = math.isqrt(np.sum(center**2))

    num_rho = np.max(image.shape)    
    if num_rho > 500:
        num_theta = 2*num_rho
    else:
        num_theta = 4*num_rho

    if scaling == 'linear':
        rho = np.linspace(0, radius, num_rho, endpoint=False)
        theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)

        rho_mesh, theta_mesh = np.meshgrid(rho, theta)

        y = rho_mesh * np.sin(theta_mesh) + center[0] + pad
        x = rho_mesh * np.cos(theta_mesh) + center[1] + pad
    elif scaling == 'log':
        log_rho = np.linspace(0, np.log(radius + eps), num_rho, endpoint=False)
        theta = np.linspace(0, 2*np.pi, num_theta, endpoint=False)

        log_rho_mesh, theta_mesh = np.meshgrid(log_rho, theta)

        y = np.exp(log_rho_mesh) * np.sin(theta_mesh) + center[0] + pad
        x = np.exp(log_rho_mesh) * np.cos(theta_mesh) + center[1] + pad
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")
        
    polar_map = np.zeros((num_theta, num_rho), dtype=image.dtype)
    image_padded = np.pad(image, ((pad, pad), (pad, pad)), 'edge')
    scipy.ndimage.map_coordinates(image_padded, (y,x), order=3, output=polar_map, mode='constant') 
    return polar_map



def polar_to_cartesian(image: np.ndarray, scaling: str ='linear', output_shape: tuple =None, pad: int =3, eps: float =1e-12) -> np.ndarray:
    """
    Convert a polar mapped image back to cartesian coordinates.

    Parameters:
        image (np.ndarray): Input polar image to be transformed.
        scaling (str): Type of scaling used in the polar mapping ('linear' or 'log').
                       - 'linear': Assumes linear scaling for radius.
                       - 'log': Assumes logarithmic scaling for radius.
        output_shape (tuple): Shape of the output Cartesian image. If None, defaults to square shape.
        pad (int): Number of pixels to pad the image on each side.
        eps (float): A small value to avoid issues with log calculations.

    Returns:
        np.ndarray: The Cartesian mapped image.
        
    Equations:
        From Polar Mapping (scaling == 'linear'):
            Map to Cartesian coordinates (x, y):
                - rho = sqrt(x**2 + y**2)
                - theta = arctan(y/x)
        
        From Log-Polar Mapping  (scaling == 'log'):
            Map to Cartesian coordinates (x, y):
                - log_rho = log(sqrt(x**2 + y**2) + eps)
                - theta = arctan(y/x)
        
        Note, for adjusting negative angles, we do:
                      |- theta+2*pi   (if theta < 0)
            - theta = |
                      |- theta         (otherwise)
    """
    if output_shape is None:
        output_shape = (image.shape[1], image.shape[1])
    elif not isinstance(output_shape, tuple):
        raise ValueError("output_shape must be instance of tuple ")

    center = np.array(output_shape)//2 # (y,x)
    radius = math.isqrt(np.sum(center**2))

    scale = {
        'radius': image.shape[1] / radius,
        'radius_log': image.shape[1] / np.log(radius+eps),
        'theta' : image.shape[0] / (2*np.pi)
    }

    y, x = np.meshgrid( np.arange(output_shape[0]),  np.arange(output_shape[1]), indexing='ij') - center[:,None,None]

    if scaling == 'linear':
        rho = scale['radius'] * np.sqrt(x**2 + y**2)
        theta = np.arctan2(y,x) 
        theta = scale['theta'] * np.where(theta < 0, theta + 2*np.pi, theta) 
    elif scaling == 'log':
        rho = scale['radius_log'] * np.log( np.sqrt((x+eps)**2 + y**2))
        theta = np.arctan2(y,x+eps) 
        theta = scale['theta'] * np.where(theta < 0, theta + 2*np.pi, theta) 
    else:
        raise ValueError("Scaling value must be in {'linear', 'log'}")
    
    cartesian_map = np.zeros(output_shape, dtype=image.dtype)
    image_padded = np.pad(image, ((pad, pad), (pad, pad)), 'edge')
    scipy.ndimage.map_coordinates(image_padded, (theta + pad, rho + pad), order=3, output=cartesian_map,  mode='constant')
    return cartesian_map 