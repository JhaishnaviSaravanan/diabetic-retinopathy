import streamlit as st
from PIL import Image
import os
import shutil
import tempfile
from pathlib import Path
from ultralytics import YOLO
import base64
from io import BytesIO
import pandas as pd

# Load model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train/weights/best.pt")

# Page configuration
st.set_page_config(
    page_title="DR Prediction System",
    page_icon="👁️",
    layout="wide"
)

model = load_model()

# Enhanced stage information with severity levels
stage_info = {
    "Mild NPDR": {
        "🧬 Primary Change": "Microaneurysms (tiny bulges in blood vessels)",
        "👁️ Vision Impact": "None or minimal",
        "😐 Symptoms": "Often asymptomatic",
        "🩺 Management": "Regular monitoring; blood sugar control",
        "⚠️ Severity": "Low",
        "🏥 Urgency": "Routine follow-up"
    },
    "Moderate NPDR": {
        "🧬 Primary Change": "Blood vessels become blocked",
        "👁️ Vision Impact": "Possible mild changes",
        "😐 Symptoms": "May still be asymptomatic",
        "🩺 Management": "Closer monitoring; tighter glucose and BP control",
        "⚠️ Severity": "Moderate",
        "🏥 Urgency": "Regular monitoring"
    },
    "Severe NPDR": {
        "🧬 Primary Change": "More vessels blocked → areas of retina starved of blood",
        "👁️ Vision Impact": "Increasing risk of vision loss",
        "😐 Symptoms": "Blurred vision possible",
        "🩺 Management": "Immediate referral to ophthalmologist; possible laser treatment",
        "⚠️ Severity": "High",
        "🏥 Urgency": "Prompt medical attention"
    },
    "PDR": {
        "🧬 Primary Change": "New abnormal blood vessels grow (neovascularization)",
        "👁️ Vision Impact": "Severe → risk of blindness",
        "😐 Symptoms": "Floaters, vision loss, dark areas",
        "🩺 Management": "Urgent treatment (laser therapy, injections, surgery)",
        "⚠️ Severity": "Critical",
        "🏥 Urgency": "URGENT - Immediate treatment required"
    }
}

def get_image_base64(img_path):
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_severity_color(stage):
    colors = {
        "Mild NPDR": "🟢",
        "Moderate NPDR": "🟡", 
        "Severe NPDR": "🟠",
        "PDR": "🔴"
    }
    return colors.get(stage, "⚪")

# Main header
st.markdown(
    "<h1 style='text-align: center;'>👁️ Diabetic Retinopathy Detection AI</h1>",
    unsafe_allow_html=True
)

# File upload section
st.markdown("#### 📤 Image Upload")
uploaded_files = st.file_uploader(
    "Select fundus images for analysis",
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True,
    help="Upload clear fundus photography images. Multiple files supported."
)

if not uploaded_files:
    st.divider()

else:
    # Progress tracking
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Results summary
    results_summary = {
        "files_processed": 0,
        "stages_detected": set(),
        "high_risk_count": 0
    }
    
    import warnings
    import logging
    import cv2
    warnings.filterwarnings("ignore")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    logging.getLogger("torch").setLevel(logging.ERROR)

    for idx, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (idx + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing {uploaded_file.name}... ({idx + 1}/{total_files})")

        # Use a temp file in the root workspace folder for image upload
        temp_image_path = os.path.join(os.getcwd(), f"_temp_{idx}{os.path.splitext(uploaded_file.name)[1]}")
        with open(temp_image_path, "wb") as temp_img:
            temp_img.write(uploaded_file.read())
        image_path = temp_image_path

        # Run prediction
        results = model.predict(image_path, save=False, conf=0.25)

        # Convert images to base64
        base64_orig = get_image_base64(image_path)
        # Get prediction image in memory and fix color shift
        pred_img = results[0].plot()  # numpy array with boxes drawn (BGR)
        pred_img_rgb = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        pred_pil = Image.fromarray(pred_img_rgb)
        buffered = BytesIO()
        pred_pil.save(buffered, format="PNG")
        base64_pred = base64.b64encode(buffered.getvalue()).decode()

        # Remove temp image after use
        try:
            os.remove(temp_image_path)
        except Exception:
            pass

        # Create expandable section for each image
        with st.expander(f"📊 Analysis Results: {uploaded_file.name}", expanded=True):
            # Image comparison
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.markdown("#### 🖼️ Original Image")
                st.markdown(
                    f"<div style='display: flex; justify-content: center;'>"
                    f"<img src='data:image/png;base64,{base64_orig}' style='max-height:350px; width:500px;'>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with img_col2:
                st.markdown("#### 🖼️ AI Detection")
                st.markdown(
                    f"<div style='display: flex; justify-content: center;'>"
                    f"<img src='data:image/png;base64,{base64_pred}' style='max-height:350px; width:500px;'>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Detection results
            detected_classes = set()
            confidence_scores = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    class_name = model.names[cls_id]
                    confidence = float(box.conf[0].item())
                    detected_classes.add(class_name)
                    confidence_scores.append(confidence)

            if detected_classes:
                st.divider()
                for class_name in sorted(detected_classes):
                    severity_indicator = get_severity_color(class_name)
                    st.markdown(f"### {severity_indicator} Detected Stage: **{class_name}**")
                    info = stage_info.get(class_name)
                    if info:
                        st.markdown("#### 📋 Clinical Information")
                        info_col1, info_col2 = st.columns(2)
                        items = list(info.items())
                        mid_point = len(items) // 2
                        with info_col1:
                            df1 = pd.DataFrame(items[:mid_point], columns=["Parameter", "Details"])
                            st.dataframe(df1, hide_index=True, use_container_width=True)
                        with info_col2:
                            df2 = pd.DataFrame(items[mid_point:], columns=["Parameter", "Details"])
                            st.dataframe(df2, hide_index=True, use_container_width=True)
                        if class_name in ["Severe NPDR", "PDR"]:
                            st.error(f"⚠️ **ATTENTION REQUIRED**: {class_name} detected. Please consult an ophthalmologist promptly.")
                            results_summary["high_risk_count"] += 1
                        elif class_name in ["Moderate NPDR"]:
                            st.warning(f"⚠️ **MONITORING NEEDED**: {class_name} detected. Regular follow-up recommended.")
                        else:
                            st.success(f"✅ **EARLY STAGE**: {class_name} detected. Continue regular monitoring.")
                    results_summary["stages_detected"].add(class_name)
            else:
                st.success("✅ **No DR stages detected** in this image")
                st.info("Continue regular eye examinations as recommended by your healthcare provider")

        results_summary["files_processed"] += 1
    
    # Final progress update
    progress_bar.progress(1.0)
    status_text.text("✅ Analysis complete!")
    
    # Important disclaimer
    st.divider()
    st.warning("""
    ⚠️ **Medical Disclaimer**: This AI system is designed to assist in screening but should not replace professional medical diagnosis. 
    Always consult with qualified healthcare providers for proper diagnosis and treatment decisions.
    """)