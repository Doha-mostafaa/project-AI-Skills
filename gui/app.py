import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Art Style Classifier", layout="wide", page_icon="🎨")

st.markdown("""
<style>
    .main-title {font-size: 3rem; color: #4B4BFF; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎨 Art Style Recognition & Comparison</p>', unsafe_allow_html=True)
st.markdown("### Compare: ResNet50 vs EfficientNet-B0 vs VGG16")

SELECTED_CLASSES = ["Impressionism", "Realism", "Expressionism", "Romanticism", "Post_Impressionism"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2.models
def build_resnet():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 5)
    )
    return model

def build_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    return model

def build_vgg():
    model = models.vgg16(weights=None)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 1024),
        nn.ReLU(True),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(True),
        nn.Dropout(0.4),
        nn.Linear(512, 5)
    )
    return model

# 3. weights
@st.cache_resource
def load_all_models():
    models_dict = {}
    
    # 1. ResNet50
    try:
        res = build_resnet()
        ckpt = torch.load("resnet_50.pt", map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            res.load_state_dict(ckpt["model_state_dict"])
        else:
            res.load_state_dict(ckpt)
        res.to(device).eval()
        models_dict["ResNet50"] = res
    except Exception as e:
        print(f"ResNet Error: {e}")

    # 2. EfficientNet-B0
    try:
        eff = build_efficientnet()
        eff.load_state_dict(torch.load("efficientnet_b0_best.pth", map_location=device))
        eff.to(device).eval()
        models_dict["EfficientNet-B0"] = eff
    except Exception as e:
        print(f"EffNet Error: {e}")

    # 3. VGG16
    try:
        vgg = build_vgg()
        vgg.load_state_dict(torch.load("best_vgg16.pth", map_location=device))
        vgg.to(device).eval()
        models_dict["VGG16"] = vgg
    except Exception as e:
        print(f"VGG Error: {e}")
        
    return models_dict

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].clone()

        h1 = self.target_layer.register_forward_hook(forward_hook)
        h2 = self.target_layer.register_full_backward_hook(backward_hook)
        self.hook_handles.extend([h1, h2])

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[:, class_idx]
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().detach().cpu().numpy()
        
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() + 1e-8)
        else:
            cam = cam 
            
        return cam

    def close(self):
        for h in self.hook_handles:
            h.remove()

# 5. (Preprocessing)
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


models_loaded = load_all_models()
if not models_loaded:
    st.error("⚠️ The model file was not found, check the names and location.")

tab1, tab2 = st.tabs(["🖼️ Live Demo & Comparison", "📊 Evaluation Metrics"])

with tab1:
    st.sidebar.header("Input Options")
    source = st.sidebar.radio("Choose Input:", ["Upload Image", "Camera"])
    
    img_input = None
    if source == "Upload Image":
        uploaded = st.sidebar.file_uploader("Upload Art Image", type=["jpg", "png", "jpeg"])
        if uploaded:
            img_input = Image.open(uploaded).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            img_input = Image.open(cam).convert("RGB")

    if img_input:
        col_orig, _ = st.columns([1, 2])
        with col_orig:
            st.image(img_input, caption="Original Image", width=300)
        
        if st.button("Analyze & Compare"):
            input_tensor = val_tfms(img_input).unsqueeze(0).to(device)
            
            cols = st.columns(len(models_loaded))
            
            for idx, (name, model) in enumerate(models_loaded.items()):
                with cols[idx]:
                    st.info(f"🤖 {name}")
                    
                    try:
                        # 1. Prediction
                        with torch.no_grad():
                            logits = model(input_tensor)
                            probs = torch.softmax(logits, dim=1)
                            top3_prob, top3_idx = torch.topk(probs, 3)
                        
                        st.write("**Top-3 Predictions:**")
                        pred_class_idx = top3_idx[0, 0].item()
                        
                        for i in range(3):
                            c_name = SELECTED_CLASSES[top3_idx[0, i].item()]
                            c_prob = top3_prob[0, i].item()
                            st.progress(c_prob)
                            st.caption(f"{c_name}: {c_prob*100:.1f}%")

                        # 2. Grad-CAM
                        st.write("**Attention Map:**")
                        
                        target_layer = None
                        if "ResNet" in name:
                            target_layer = model.layer4[-1]
                        elif "EfficientNet" in name:
                            target_layer = model.features[-1] 
                        elif "VGG" in name:
                            target_layer = model.features[-1]
                        
                        if target_layer:
                            gcam = GradCAM(model, target_layer)
                            cam_mask = gcam.generate(input_tensor, pred_class_idx)
                            gcam.close()
                            
                            cam_mask = cv2.resize(cam_mask, (224, 224))
                            heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
                            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
                            
                            orig_np = np.array(img_input.resize((224, 224))) / 255.0
                            overlay = 0.6 * orig_np + 0.4 * heatmap
                            overlay = np.clip(overlay, 0, 1)
                            
                            st.image(overlay, caption=f"Focus Area ({name})", use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Error in {name}: {e}")

with tab2:
    st.header("Model Evaluation Results")
    st.markdown("Based on the Test Set (WikiArt Subset)")
    
    import pandas as pd
    summary_data = {
        "Model": ["EfficientNet-B0", "ResNet50", "VGG16"],
        "Accuracy": ["82.45%", "69.70%", "67.82%"],
        "F1-Score": ["0.8265", "0.6952", "0.6790"],
        "Best For": ["High Accuracy & Balance", "General Use", "Academic Baseline"]
    }
    df = pd.DataFrame(summary_data)
    st.table(df)

    st.markdown("---") 
    st.markdown("### Detailed Analysis & Confusion Matrix")
    
    
    model_choice = st.selectbox("Select Model:", ["EfficientNet-B0", "ResNet50", "VGG16"])
    
    col1, col2 = st.columns([1, 1])
    
    if model_choice == "EfficientNet-B0":
        with col1:
            st.metric("Accuracy", "82.45%")
            st.metric("F1-Score", "0.8265")
            st.metric("Precision", "0.8210")
            st.metric("Recall", "0.8340")
        with col2:
            st.markdown("**Confusion Matrix:**")
            st.image("cm_efficientnet.png", use_container_width=True)
            
    elif model_choice == "ResNet50":
        with col1:
            st.metric("Accuracy", "69.70%")
            st.metric("F1-Score", "0.6952")
            st.metric("Precision", "0.6930")
            st.metric("Recall", "0.7032")
        with col2:
            st.markdown("**Confusion Matrix:**")
            st.image("cm_resnet.png", use_container_width=True)
            
    elif model_choice == "VGG16":
        with col1:
            st.metric("Accuracy", "67.82%")
            st.metric("F1-Score", "0.6790")
            st.metric("Precision", "0.6722")
            st.metric("Recall", "0.6995")
        with col2:
            st.markdown("**Confusion Matrix:**")
            st.image("cm_vgg.png", use_container_width=True)
