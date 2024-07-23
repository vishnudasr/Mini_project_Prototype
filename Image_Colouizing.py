import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import io

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the colorization model
class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load the trained model
model = ColorizationNet().to(device)
model.load_state_dict(torch.load('colorization_model.pth', map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

def colorize_image(image, model, device):
    original_size = image.size
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    output = output.squeeze().permute(1, 2, 0).cpu().numpy()
    output = (output * 255).astype(np.uint8)
    output_image = Image.fromarray(output)
    output_image = output_image.resize(original_size, Image.BILINEAR)
    return output_image

# Streamlit interface
st.title("Image Colorization using Deep Learning")

uploaded_file = st.file_uploader("Choose a grayscale image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption='Uploaded Grayscale Image', use_column_width=True)
    
    st.write("")
    st.write("Colorizing...")
    
    colorized_image = colorize_image(image, model, device)
    
    st.image(colorized_image, caption='Colorized Image', use_column_width=True)
    
    # Option to download the colorized image
    img_byte_arr = io.BytesIO()
    colorized_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    st.download_button(
        label="Download Colorized Image",
        data=img_byte_arr,
        file_name="colorized_image.png",
        mime="image/png"
    )
