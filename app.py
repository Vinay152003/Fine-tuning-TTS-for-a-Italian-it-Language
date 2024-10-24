import gradio as gr
import torch
from datasets import load_dataset
from transformers import SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech

# Load the fine-tuned model and vocoder for Italian
model_id = "Sandiago21/speecht5_finetuned_voxpopuli_it"
model = SpeechT5ForTextToSpeech.from_pretrained(model_id)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Load speaker embeddings dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7440]["xvector"]).unsqueeze(0)

# Load processor for the Italian model
processor = SpeechT5Processor.from_pretrained(model_id)

# Optional: Text cleanup for Italian-specific characters
replacements = [
    ("à", "a"),
    ("è", "e"),
    ("é", "e"),
    ("ì", "i"),
    ("ò", "o"),
    ("ù", "u"),
]

# Text-to-speech synthesis function
def synthesize_speech(text):
    # Clean up text
    for src, dst in replacements:
        text = text.replace(src, dst)

    # Process input text
    inputs = processor(text=text, return_tensors="pt")

    # Generate speech using the model and vocoder
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Return the generated speech as (sample_rate, audio_array)
    return (16000, speech.cpu().numpy())

# Title and description for the Gradio interface
title = "Italian Text-to-Speech with SpeechT5"
description = """
This demo generates speech in Italian using the fine-tuned SpeechT5 model from Hugging Face.
The model is fine-tuned on the VoxPopuli Italian dataset.
"""

# Create Gradio interface
interface = gr.Interface(
    fn=synthesize_speech,
    inputs=gr.Textbox(label="Input Text", placeholder="Enter Italian text here..."),
    outputs=gr.Audio(label="Generated Speech"),
    title=title,
    description=description,
    examples=["Questa è una dimostrazione di sintesi vocale in italiano."]
)

# Launch the interface
interface.launch()
