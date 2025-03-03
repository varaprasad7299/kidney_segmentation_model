import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import uvicorn
# Initialize FastAPI
app = FastAPI()

# Define model architecture
class EfficientNetB5_FPN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b5')
        self.channels = [40, 64, 176, 512]
        self.fpn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, 256, kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for in_ch in reversed(self.channels)
        ])
        self.smooth_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for _ in range(len(self.channels))
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x):
        endpoints = self.backbone.extract_endpoints(x)
        features = [
            endpoints['reduction_2'],
            endpoints['reduction_3'],
            endpoints['reduction_4'],
            endpoints['reduction_5']
        ]

        fpn_features = []
        prev_feature = None
        for feature, fpn_layer, smooth_layer in zip(
            reversed(features), self.fpn_layers, self.smooth_layers):

            if prev_feature is None:
                prev_feature = fpn_layer(feature)
            else:
                prev_feature = F.interpolate(
                    prev_feature, size=feature.shape[-2:], mode='nearest')
                prev_feature = prev_feature + fpn_layer(feature)

            fpn_features.insert(0, smooth_layer(prev_feature))

        output = self.final_conv(fpn_features[0])
        output = F.interpolate(output, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return output

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientNetB5_FPN(num_classes=3)
model.load_state_dict(torch.load("efficientnet_b5_fpn_best.pth", map_location=device))
model.to(device)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    output_mask = (output_mask / output_mask.max()) * 255
    output_mask = output_mask.astype(np.uint8)

    mask_image = Image.fromarray(output_mask)
    img_io = io.BytesIO()
    mask_image.save(img_io, format="PNG")
    img_io.seek(0)

    return Response(content=img_io.getvalue(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)