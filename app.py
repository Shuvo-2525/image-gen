import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image
import io
import urllib.parse

# Set up page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered"
)

# Initialize Hugging Face Inference Client
def get_client():
    return InferenceClient(
        provider="replicate",
        api_key=st.secrets["HF_API_KEY"]
    )

# Social media sharing functions
def create_share_links(prompt, app_url):
    text = urllib.parse.quote(f"Check out this AI-generated image I created! 🎨\nPrompt: {prompt}\n")
    return {
        "twitter": f"https://twitter.com/intent/tweet?text={text}&url={app_url}",
        "facebook": f"https://www.facebook.com/sharer/sharer.php?u={app_url}",
        "linkedin": f"https://www.linkedin.com/sharing/share-offsite/?url={app_url}",
        "whatsapp": f"https://api.whatsapp.com/send?text={text} {app_url}"
    }

# Sidebar controls
with st.sidebar:
    st.header("🛠️ Generation Settings")
    num_images = st.slider("Number of images", 1, 10, 2)
    model_name = st.text_input("Model ID", "black-forest-labs/FLUX.1-dev")
    width = st.number_input("Width", 256, 1024, 512)
    height = st.number_input("Height", 256, 1024, 512)
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)
    num_inference_steps = st.slider("Inference Steps", 10, 150, 50)
    negative_prompt = st.text_area("Negative Prompt", "")
    seed = st.number_input("Seed (-1 for random)", -1, 1000000, -1)

# Main interface
st.title("🎨 AI Image Generator")
prompt = st.text_input("Enter your creative prompt:", "Astronaut riding a horse in photorealistic style")

# Generation button
if st.button("Generate Images 🚀"):
    if not prompt:
        st.error("Please enter a prompt!")
    else:
        with st.spinner(f"Generating {num_images} amazing images..."):
            client = get_client()
            cols = st.columns(num_images)
            
            for i in range(num_images):
                with cols[i]:
                    try:
                        # Generate image
                        image = client.text_to_image(
                            prompt,
                            model=model_name,
                            negative_prompt=negative_prompt,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps,
                            width=width,
                            height=height,
                            seed=seed if seed != -1 else None
                        )
                        
                        # Display image
                        st.image(image, use_column_width=True)
                        
                        # Prepare image bytes for download
                        img_bytes = io.BytesIO()
                        image.save(img_bytes, format="PNG")
                        
                        # Download button
                        st.download_button(
                            label="Download Image ⬇️",
                            data=img_bytes.getvalue(),
                            file_name=f"generated_image_{i+1}.png",
                            mime="image/png"
                        )
                        
                        # Social sharing buttons
                        app_url = "YOUR_APP_URL"  # Replace with your deployed app URL
                        share_links = create_share_links(prompt, app_url)
                        
                        st.markdown("**Share this creation:**")
                        st.markdown(f"""
                            [![Twitter](https://img.shields.io/badge/Share-Twitter-1DA1F2?style=for-the-badge)]({share_links['twitter']})
                            [![Facebook](https://img.shields.io/badge/Share-Facebook-1877F2?style=for-the-badge)]({share_links['facebook']})
                            [![LinkedIn](https://img.shields.io/badge/Share-LinkedIn-0A66C2?style=for-the-badge)]({share_links['linkedin']})
                            [![WhatsApp](https://img.shields.io/badge/Share-WhatsApp-25D366?style=for-the-badge)]({share_links['whatsapp']})
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error generating image {i+1}: {str(e)}")

# Add some instructional text
st.markdown("""
    ### How to use:
    1. Enter your creative prompt in the text box
    2. Adjust generation settings in the sidebar
    3. Click the "Generate Images" button
    4. Download or share your favorite creations!
    
    ### Pro tips:
    - Use descriptive language for best results
    - Experiment with different guidance scales
    - Use negative prompts to exclude unwanted elements
    """)
