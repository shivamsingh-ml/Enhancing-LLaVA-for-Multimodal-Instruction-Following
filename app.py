import streamlit as st
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from PIL import Image
import torch
import requests
from io import BytesIO

@st.cache_resource
def load_model():
    model_id = "llava-hf/llava-1.5-7b-hf"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )

    return processor, model

processor, model = load_model()

st.set_page_config(page_title="LLaVA Visual Assistant", page_icon="🖼️")

st.title("🖼️ LLaVA Visual Assistant")
st.write("Upload an image or provide a URL and enter a prompt to ask about the image.")

image_source = st.radio("Choose image source:", ["Upload", "URL"])

image = None

if image_source == "Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
elif image_source == "URL":
    url = st.text_input("Enter image URL")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("Failed to load image from URL.")

if image:
    st.image(image, caption="Input Image", use_container_width=True)

prompt = st.text_input("Prompt", "What is in this image?")

if st.button("Generate Answer") and image and prompt:
    with st.spinner("Generating response..."):
        prompt_with_image = "<image>\n" + prompt

        inputs = processor(text=prompt_with_image, images=image, return_tensors="pt").to(model.device)

        outputs = model.generate(**inputs, max_new_tokens=100)
        response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.success("LLaVA's Response:")
    st.write(response)
