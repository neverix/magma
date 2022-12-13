from magma import Magma
from magma.image_input import ImageInput
from magma.image_encoders import get_image_encoder

model = get_image_encoder("openclip")

inputs =[
    ## supports urls and path/to/image
    ImageInput('https://www.art-prints-on-demand.com/kunst/thomas_cole/woods_hi.jpg'),
    'Describe the painting:'
]

# print(output[0]) ##  A cabin on a lake
