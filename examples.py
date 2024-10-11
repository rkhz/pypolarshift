# %%
import torchvision
import pypolarshift as pp 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)) #standardization
    ])

    data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform= torchvision.transforms.ToTensor())

    image = data[0][0].squeeze().numpy()
    
    # get log-polar mapping    
    polar_map = pp.mapping.cartesian_to_polar(image, scaling='log')

    # apply a rotation transforn to the image
    image_rot, polar_map_rot = pp.shift.apply_shift(image, k=4, transform='rotate')
    
    pp.utils.plot_polar_cart(polar_map, image, polar_map_rot, image_rot)
    
# %%
