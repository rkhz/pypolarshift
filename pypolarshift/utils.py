import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_rot_transform(polar_map, polar_map_shifted, image, title=''):
    fig = plt.figure()
    plt.subplot(311)
    plt.imshow(polar_map.T)
    plt.title('{}Polar Map'.format(title))
    plt.axis('off')

    plt.subplot(312)
    plt.imshow(polar_map_shifted.T)
    plt.title('{}Polar Map (shifted)'.format(title))
    plt.axis('off')

    plt.subplot(313)
    plt.imshow(image)
    plt.title('{}Rotated Image'.format(title))
    plt.axis('off')

    plt.tight_layout(h_pad=0.5,w_pad=0.0)
    plt.show()
    
    
    
def plot_polar_cart(polar_map, image, polar_map_shifted, image_rot):
    col, row  = polar_map.shape
    if col > row:
        polar = polar_map.T
    else:
        col, row  = polar_map.T.shape
    fig = plt.figure(figsize=(10, 5)) 
    gs = GridSpec(2, 2, width_ratios=[col//row, 1])
    
    
    ax1 = fig.add_subplot(gs[0])
    ax1.set_aspect('equal') 
    im1 = ax1.imshow(polar_map.T)
    ax1.set_title('Log-Polar Map')
    ax1.set_ylabel('log(radius) (log_rho)')
    ax1.set_xlabel('angle (theta)')
    # Second subplot (NxN)
    
    ax2 = fig.add_subplot(gs[1])
    ax2.set_aspect('equal')  
    im2 = ax2.imshow(image)  
    ax2.set_title('Image')
    ax2.set_ylabel('y')
    ax2.set_xlabel('x')
    plt.tight_layout()
    
    ax3 = fig.add_subplot(gs[2])
    ax3.set_aspect('equal') 
    im3 = ax3.imshow(polar_map_shifted.T) 
    ax3.set_title('Shifthed Log-Polar Map')
    ax3.set_ylabel('log(radius) (log_rho)')
    ax3.set_xlabel('angle (theta)')
    
    ax4 = fig.add_subplot(gs[3])
    ax4.set_aspect('equal')
    im4 = ax4.imshow(image_rot)  
    ax4.set_title('Rotated Image')
    ax4.set_ylabel('y')
    ax4.set_xlabel('x')
    plt.tight_layout()
    plt.show()