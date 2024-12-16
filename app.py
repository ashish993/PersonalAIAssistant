import streamlit as st
from typing import Generator
from groq import Groq
import base64

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Personal Assistant")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


st.subheader("My Personal Assistant", divider="rainbow", anchor=False)

client = Groq(
    api_key=Groq(api_key=st.secrets["GROQ_API_KEY"])
)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details

models = "llama-3.3-70b-versatile"

# Sidebar for image upload
with st.sidebar:
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    upload_button = st.button("Upload")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def analyze_image(image_bytes):
    """Analyze the image in the uploaded image."""
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    response = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Identify the image and analyze it and give a detailed description or any specific information as asked by the user."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        stream=False,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stop=None,
    )

    return response.choices[0].message.content


def handle_uploaded_image(uploaded_image):
    """Handles the uploaded image by displaying it and adding a message to the chat."""
    st.session_state.messages.append({"role": "user", "content": "Uploaded an image."})
    with st.chat_message("user", avatar='👨‍💻'):
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Analyze the image in the uploaded image
    image_bytes = uploaded_image.read()
    contents = analyze_image(image_bytes)
    st.session_state.image_description = contents

    # Display the response immediately
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(contents)

    # Save the uploaded image in session state
    st.session_state.uploaded_image = uploaded_image


if upload_button and uploaded_image:
    handle_uploaded_image(uploaded_image)

if prompt := st.chat_input("Enter your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Check if there is an uploaded image description in session state
    if "image_description" in st.session_state:
        st.session_state.messages.append({"role": "assistant", "content": st.session_state.image_description})

    # Fetch response from Groq API for the prompt
    try:
        chat_completion = client.chat.completions.create(
            model=models,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})
