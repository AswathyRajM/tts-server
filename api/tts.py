# apis.py

import sys
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

def generate_speech(text, person):
    # Initialize SpeechT5 components
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Process text using the processor
    inputs = processor(text=text, return_tensors="pt")

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

    # Set the speaker based on the provided person parameter
    if person == "male":
        speaker_index = 5004
    elif person == "female":
        speaker_index = 7306
    else:
        raise ValueError("Invalid value for 'person'. Use 'male' or 'female'.")

    # Generate speech using the selected speaker
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]["xvector"]).unsqueeze(0)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Save the generated speech as a WAV file
    # sf.write("speech.wav", speech.numpy(), samplerate=16000)

    # print(f"The speech was generated for {result_person}.")
    # Create an in-memory buffer to hold the speech data
    output_file = "output_file.wav"

    # Write the speech data to the buffer
    sf.write(output_file, speech.numpy(), samplerate=16000, format='wav', subtype='PCM_16')

    # Return the in-memory buffer
    return output_file

