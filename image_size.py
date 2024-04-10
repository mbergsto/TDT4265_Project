from PIL import Image

# Erstatt 'path_to_your_image.jpg' med den faktiske stien til bildet ditt
image_path = 'rbk/1_train-val_1min_aalesund_from_start/img1/000003.jpg'
with Image.open(image_path) as img:
    width, height = img.size

print(f"Bildestørrelsen er: {width}x{height} piksler")
