import PIL
from PIL import Image

#baseheight = 200
img = Image.open('austin.jpg')
#hpercent = (baseheight / float(img.size[1]))
#wsize = int((float(img.size[0]) * float(hpercent)))
img = img.resize((300, 400), PIL.Image.ANTIALIAS)
img.save('siple_resized_image.jpg')