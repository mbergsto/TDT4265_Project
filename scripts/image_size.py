from PIL import Image

# Erstatt 'path_to_your_image.jpg' med den faktiske stien til bildet ditt
image_path = 'pitch.png'
with Image.open(image_path) as img:
    width, height = img.size

print(f"Bildestørrelsen er: {width}x{height} piksler")
