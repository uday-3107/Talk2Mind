import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.facial.preprocess_facial import CLASSES

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = "evaluation/results/gradcam_examples"
os.makedirs(RESULTS_DIR, exist_ok=True)

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

model_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices for key expressive regions
# ---------------------------------------------------------------------------
_LEFT_EYE   = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_BROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_MOUTH      = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
               375, 321, 405, 314, 17, 84, 181, 91, 146]
_NOSE       = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

_LANDMARK_GROUPS = [_LEFT_EYE, _RIGHT_EYE, _LEFT_BROW, _RIGHT_BROW, _MOUTH, _NOSE]


def _build_landmark_mask(pil_img_224):
    """
    Strategy 1 — MediaPipe landmark-guided spatial mask.
    Returns float32 (224,224) mask with 1.0 inside eyes/brows/mouth/nose,
    0.0 elsewhere, Gaussian-smoothed at edges.
    Returns None if MediaPipe unavailable or no face detected.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None

    mp_face_mesh = mp.solutions.face_mesh
    img_np = np.array(pil_img_224, dtype=np.uint8)
    h, w   = img_np.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.3,
    ) as face_mesh:
        results = face_mesh.process(img_np)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in landmarks])

    mask = np.zeros((h, w), dtype=np.float32)
    dilation = max(5, int(h * 0.03))

    for group in _LANDMARK_GROUPS:
        group_pts = pts[group]
        hull = cv2.convexHull(group_pts)
        cx = int(np.mean(group_pts[:, 0]))
        cy = int(np.mean(group_pts[:, 1]))
        dilated = hull.copy().astype(np.float32)
        for i in range(len(hull)):
            dx = hull[i][0][0] - cx
            dy = hull[i][0][1] - cy
            dist = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
            scale = 1.0 + dilation / dist
            dilated[i][0][0] = cx + dx * scale
            dilated[i][0][1] = cy + dy * scale
        dilated = np.clip(dilated, 0, [w - 1, h - 1]).astype(np.int32)
        cv2.fillPoly(mask, [dilated], 1.0)

    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    if mask.max() > 0:
        mask = mask / mask.max()
    return mask


def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, len(CLASSES)),
    )
    model.load_state_dict(torch.load("models/facial_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def generate_gradcam(image_path_or_pil, model=None, save_path=None):
    """
    Landmark-guided GradCAM (Strategy 1).

    Pipeline:
      1. Run GradCAM on the full 224x224 face (classification unchanged).
      2. Build MediaPipe landmark mask over eyes, eyebrows, mouth, nose.
      3. Multiply raw activation map by mask — heat stays in expressive regions.
      4. Re-normalise and overlay coloured heatmap on original image.

    Falls back to unmasked full-face GradCAM when MediaPipe is unavailable
    or detects no landmarks.

    Returns: (overlay_PIL_image, class_name, confidence, raw_cam_array)
    NOTE: Dashboard ignores class_name/confidence from here and uses the
    facial_probs already obtained from get_facial_embedding() so the
    polar chart and the GradCAM caption always agree.
    """
    if model is None:
        model = load_model()

    if isinstance(image_path_or_pil, str):
        pil_img = Image.open(image_path_or_pil).convert("RGB")
    else:
        pil_img = image_path_or_pil.convert("RGB")

    pil_224      = pil_img.resize((224, 224), Image.LANCZOS)
    input_tensor = model_transform(pil_224).unsqueeze(0).to(DEVICE)

    # Step 1 — Raw GradCAM (full face, target = top prediction)
    cam           = GradCAM(model=model, target_layers=[model.layer4[1].conv2])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]   # (224, 224)

    # Steps 2 & 3 — Landmark mask x activation
    landmark_mask = _build_landmark_mask(pil_224)
    if landmark_mask is not None:
        masked_cam = grayscale_cam * landmark_mask
        cam_max    = masked_cam.max()
        if cam_max > 1e-6:
            masked_cam = masked_cam / cam_max
        display_cam = masked_cam
    else:
        display_cam = grayscale_cam     # fallback

    # Step 4 — Colour overlay
    rgb_img = np.array(pil_224) / 255.0
    overlay = show_cam_on_image(rgb_img, display_cam, use_rgb=True)
    result  = Image.fromarray(overlay)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(save_path)

    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)
    pred_idx   = probs.argmax(dim=1).item()
    confidence = probs[0, pred_idx].item()

    return result, CLASSES[pred_idx], confidence, grayscale_cam


def run_demo():
    model = load_model()
    print(f"Model loaded on {DEVICE}")
    print(f"Classes: {CLASSES}\n")

    test_dir = "data/raw/facial/test"
    summary  = []

    for cls in CLASSES:
        class_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        images = sorted(os.listdir(class_dir))
        if not images:
            continue
        img_path  = os.path.join(class_dir, images[0])
        base_name = os.path.splitext(images[0])[0]
        save_path = os.path.join(RESULTS_DIR, f"{cls}_{base_name}_gradcam.jpg")

        result, pred, conf, _ = generate_gradcam(img_path, model, save_path)

        correct = "Yes" if pred == cls else "No"
        summary.append((cls, pred, conf, correct))
        print(f"  {cls:>10} -> predicted {pred:<10} ({conf:.1%}) {correct}")

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("Summary:")
    print(f"  {'True':>10} {'Pred':<10} {'Conf':<8} Match")
    print(f"  {'-'*35}")
    for true_label, pred, conf, correct in summary:
        print(f"  {true_label:>10} {pred:<10} {conf:.1%}  {correct}")


if __name__ == "__main__":
    run_demo()
