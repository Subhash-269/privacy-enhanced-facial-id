import torch
import tenseal as ts
from torchvision import transforms
from PIL import Image
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load and transform an image
image = Image.open('test_images\licensed-image.jpeg')
image_tensor = transform(image)

# Encrypt a tensor
def encrypt_tensor(tensor):
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=16384, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    encrypted_tensor = ts.ckks_vector(context, tensor.flatten().tolist())
    return encrypted_tensor, context

# Encrypt the image tensor
encrypted_image, context = encrypt_tensor(image_tensor)

# # Decrypt the tensor
# def decrypt_tensor(encrypted_tensor, context):
#     decrypted_data = encrypted_tensor.decrypt(context)
#     return torch.tensor(decrypted_data).reshape(3, 224, 224)

# # Decrypt and view the image
# decrypted_image_tensor = decrypt_tensor(encrypted_image, context)
# decrypted_image = transforms.ToPILImage()(decrypted_image_tensor)
# decrypted_image.show()