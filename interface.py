import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Parkinson's Wave Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        height: 3.5em;
        border-radius: 12px;
        font-size: 18px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .prediction-box {
        padding: 30px;
        border-radius: 16px;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .healthy-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
    }
    
    .parkinson-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
    }
    
    .upload-section {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 16px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: white;
    }
    
    h1, h2, h3 {
        font-weight: 700;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .uploaded-image {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 3px solid #e0e0e0;
    }
    
    .step-indicator {
        display: flex;
        align-items: center;
        padding: 15px;
        background: white;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def build_vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

@st.cache_resource
def load_model(model_path):
    try:
        model = build_vgg16_model()
        model.load_weights(model_path)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"❌ Failed to load weights: {e}")
        return None

def preprocess_image(image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    return img_array

def predict_parkinsons(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)[0][0]
    
    if prediction > 0.5:
        label = "Parkinson's Disease"
        confidence = prediction * 100
        status = "parkinson"
    else:
        label = "Healthy"
        confidence = (1 - prediction) * 100
        status = "healthy"
    
    return label, confidence, status, prediction

def create_gauge_chart(confidence, status):
    """Create an interactive gauge chart using Plotly"""
    if status == "healthy":
        color = '#28a745'
    else:
        color = '#dc3545'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level", 'font': {'size': 24, 'weight': 'bold'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#f0f0f0'},
                {'range': [50, 75], 'color': '#e0e0e0'},
                {'range': [75, 100], 'color': '#d0d0d0'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Inter"},
        height=300
    )
    
    return fig

def main():
    # Header Banner
    st.markdown("""
    <div class='header-banner'>
        <h1 style='margin: 0; font-size: 42px;'>🏥 Parkinson's Disease Detection System</h1>
        <p style='margin: 10px 0 0 0; font-size: 18px; opacity: 0.9;'>
        AI-Powered Early Screening Through Wave Pattern Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with improved design
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        
        model_path = st.text_input(
            "Model File Path:",
            "parkinsons_vgg16_wave_final.h5",
            help="Enter the path to your trained model file"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 System Information")
        st.markdown("""
        <div class='info-card'>
            <strong>Model:</strong> VGG16 Transfer Learning<br>
            <strong>Accuracy:</strong> 83.33%<br>
            <strong>Dataset:</strong> Wave Patterns<br>
            <strong>Status:</strong> <span style='color: #28a745;'>● Active</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📝 How to Draw")
        steps = [
            ("1", "Use pen on paper/tablet"),
            ("2", "Draw continuous waves"),
            ("3", "Maintain consistent height"),
            ("4", "Keep regular spacing"),
            ("5", "Draw smoothly"),
            ("6", "Take clear photo")
        ]
        
        for num, text in steps:
            st.markdown(f"""
            <div class='step-indicator'>
                <div class='step-number'>{num}</div>
                <div>{text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚕️ Disclaimer")
        st.warning("""
        This tool is for **screening only**. 
        Always consult healthcare professionals 
        for accurate diagnosis.
        """)
    
    # Main content with two columns
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class='upload-section'>
            <h2 style='margin-top: 0;'>📤 Upload Wave Drawing</h2>
            <p style='color: #666;'>Upload a clear image of a hand-drawn wave pattern for AI analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.markdown("<div class='uploaded-image'>", unsafe_allow_html=True)
            st.image(image, caption="📸 Uploaded Wave Drawing")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Image metadata
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Width", f"{image.size[0]}px")
            with col_b:
                st.metric("Height", f"{image.size[1]}px")
        else:
            st.info("👆 Please upload a wave drawing to begin analysis")
            
            st.markdown("---")
            st.markdown("#### 💡 What Makes a Good Wave?")
            
            good_features = [
                ("✅", "Continuous smooth curves"),
                ("✅", "Consistent wave height"),
                ("✅", "Regular spacing"),
                ("✅", "No breaks or gaps")
            ]
            
            for icon, text in good_features:
                st.markdown(f"{icon} {text}")
            
            st.markdown("#### ⚠️ Signs of Concern")
            concern_features = [
                ("🔴", "Irregular amplitude"),
                ("🔴", "Visible tremors"),
                ("🔴", "Inconsistent spacing"),
                ("🔴", "Difficulty maintaining smoothness")
            ]
            
            for icon, text in concern_features:
                st.markdown(f"{icon} {text}")
    
    with col2:
        st.markdown("""
        <div class='upload-section'>
            <h2 style='margin-top: 0;'>🔍 Analysis Results</h2>
            <p style='color: #666;'>AI-powered screening results will appear here</p>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Load model with progress
            with st.spinner("🤖 Loading AI model..."):
                model = load_model(model_path)
            
            if model is not None:
                st.success("✅ Model loaded successfully!")
                
                # Analyze button
                if st.button("🔬 Analyze Wave Drawing", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Preprocessing image...")
                    progress_bar.progress(25)
                    
                    status_text.text("Running AI analysis...")
                    progress_bar.progress(50)
                    
                    # Make prediction
                    label, confidence, status, raw_prediction = predict_parkinsons(model, image)
                    
                    status_text.text("Generating results...")
                    progress_bar.progress(75)
                    
                    progress_bar.progress(100)
                    status_text.empty()
                    progress_bar.empty()
                    
                    # Display results
                    st.markdown("---")
                    
                    # Result box
                    if status == "healthy":
                        st.markdown(f"""
                        <div class='prediction-box healthy-box'>
                            <h2 style='color: #28a745; margin: 0;'>✅ {label}</h2>
                            <p style='font-size: 32px; margin: 15px 0; font-weight: bold;'>{confidence:.1f}%</p>
                            <p style='margin: 0; font-size: 16px; line-height: 1.6;'>
                            The wave patterns suggest healthy motor control with minimal tremors. 
                            Continue regular monitoring and maintain a healthy lifestyle.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='prediction-box parkinson-box'>
                            <h2 style='color: #dc3545; margin: 0;'>⚠️ {label} Indicators</h2>
                            <p style='font-size: 32px; margin: 15px 0; font-weight: bold;'>{confidence:.1f}%</p>
                            <p style='margin: 0; font-size: 16px; line-height: 1.6;'>
                            The wave patterns show potential motor control irregularities. 
                            Please consult a neurologist for comprehensive evaluation.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Interactive gauge
                    st.markdown("### 📊 Confidence Visualization")
                    fig = create_gauge_chart(confidence, status)
                    st.plotly_chart(fig)
                    
                    # Metrics in cards
                    st.markdown("### 📋 Detailed Metrics")
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color: #667eea; margin: 0;'>Raw Score</h4>
                            <p style='font-size: 28px; font-weight: bold; margin: 10px 0;'>{raw_prediction:.4f}</p>
                            <p style='font-size: 12px; color: #999; margin: 0;'>Model Output</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color: #667eea; margin: 0;'>Classification</h4>
                            <p style='font-size: 20px; font-weight: bold; margin: 10px 0;'>{label}</p>
                            <p style='font-size: 12px; color: #999; margin: 0;'>Binary Result</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <h4 style='color: #667eea; margin: 0;'>Confidence</h4>
                            <p style='font-size: 28px; font-weight: bold; margin: 10px 0;'>{confidence:.1f}%</p>
                            <p style='font-size: 12px; color: #999; margin: 0;'>Certainty Level</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Recommended Actions")
                    
                    if status == "parkinson":
                        st.error("""
                        **⚠️ Important Next Steps:**
                        
                        1. 🏥 **Consult a Neurologist** - Schedule comprehensive evaluation
                        2. 🔬 **Additional Testing** - DaTscan, physical exam, clinical assessment
                        3. 📊 **Monitor Symptoms** - Track motor control changes
                        4. ⏰ **Early Action** - Better outcomes with early intervention
                        5. 👨‍👩‍👧‍👦 **Family Support** - Inform and involve family members
                        """)
                    else:
                        st.success("""
                        **✅ Maintain Good Health:**
                        
                        1. 🔄 **Regular Monitoring** - Periodic wave drawing tests
                        2. 💪 **Stay Active** - Exercise and balanced diet
                        3. 📅 **Annual Check-ups** - Routine health examinations
                        4. 📚 **Stay Informed** - Know Parkinson's warning signs
                        5. 🩺 **Report Changes** - Consult if symptoms appear
                        """)
                    
                    # Expandable info
                    with st.expander("📚 Learn More About Parkinson's Disease"):
                        st.markdown("""
                        #### What is Parkinson's Disease?
                        
                        A progressive neurological disorder affecting movement control.
                        
                        **Common Symptoms:**
                        - 🤝 Tremors (resting tremor)
                        - 🐌 Slowed movement (bradykinesia)
                        - 💪 Muscle rigidity
                        - ⚖️ Balance problems
                        - 🗣️ Speech changes
                        - ✍️ Writing difficulties
                        
                        **Early Detection Benefits:**
                        - Better symptom management
                        - More treatment options
                        - Improved quality of life
                        - Slower progression
                        """)
            else:
                st.error("❌ Model loading failed. Please verify:")
                st.markdown("""
                - ✓ Model file exists
                - ✓ Correct file path
                - ✓ File not corrupted
                - ✓ Proper VGG16 format
                """)
        else:
            st.info("👈 Upload an image to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: white; border-radius: 12px; margin-top: 30px;'>
        <h3 style='color: #667eea; margin: 0;'>🏥 Parkinson's Disease Detection System</h3>
        <p style='color: #999; margin: 10px 0 0 0;'>
        Powered by VGG16 Deep Learning | For Screening Purposes Only
        </p>
        <p style='color: #ccc; font-size: 12px; margin: 5px 0 0 0;'>
        Always consult qualified healthcare professionals for medical diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()