<h3><a href="">Fine-tuning-TTS-for-a-Italian-it-Language Using SpeechT5</a></h3>
<a href="https://huggingface.co/spaces/Vinay15/Fine-tuning_TTS_for_a_Regional_Language"><img src="https://img.shields.io/badge/Huggingface-yellow"></a>
<a href="https://www.linkedin.com/in/vinay-hipparge/"><img src="https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/vinay-hipparge/"></a>
<a href="mailto:vinayhipparge15@gmail.com"><img src="https://img.shields.io/badge/Gmail--informational?style=social&logo=gmail"></a>
<a href="https://colab.research.google.com/drive/1xR8fcgUeEQ36fQSaWxRGxYodQ16215pO"><img src="https://img.shields.io/badge/Google-Colab-red"></a>

### Live-URL
You can access the Live Web Application at My Hugging Face Space: [https://huggingface.co/spaces/Vinay15/Fine-tuning_TTS_for_a_Regional_Language](https://huggingface.co/spaces/Vinay15/Fine-tuning_TTS_for_a_Regional_Language)

[Note: The Hugging Face space may be in a sleeping state due to inactivity. Please click the link to restart the space]

This repository contains an implementation of a fine-tuned Italian text-to-speech (TTS) model based on SpeechT5 from Hugging Face. The model is trained on the VoxPopuli dataset to generate high-quality Italian speech audio from text input, with integrated support for a Gradio interface for user-friendly interactions.

## My-Fine-tuning-Model:
https://huggingface.co/Vinay15/speecht5_finetuned_voxpopuli_it

![Screenshot 2024-10-27 114156](https://github.com/user-attachments/assets/f67e6a98-fb32-49a9-baf7-0cf56ffc9e3a)


## Gradio interface: 
https://huggingface.co/spaces/Vinay15/Fine-tuning_TTS_for_a_Regional_Language

![Screenshot 2024-10-28 161645](https://github.com/user-attachments/assets/d7fb172b-3a97-4787-8150-bbee51485ad6)


## Task-2-Report
-Task 2 Report Link: https://drive.google.com/file/d/1cvNPkuFlTZAu1iDaagCwVRGXFd6r6vqi/view?usp=sharing

## Final-Report
-Final-Report Link: https://drive.google.com/file/d/1wDt3EtxSrp_HFEUA5QCGWF9W9emqswWn/view?usp=sharing


## Audio-Sample:

-Task 2 Pre-trained Model Audio using Bark TTS: https://github.com/user-attachments/assets/fb589705-63ab-454a-aa77-6e6caa8757ea

-Task 2 Fine-tuned Model Audio Sample using SpeechT5: https://github.com/user-attachments/assets/3f0b9aa9-1dcf-4802-b660-4dba0502e1c7

## Google Colab Code Link:

-Pre-trained Model Italian Language using Bark TTS Colab link:
https://colab.research.google.com/drive/1STwdydwZLgP_SyZPI1ZNp8HpjaPL5VJu?usp=sharing

-Fine Tuned Model Italian Language using SpeechT5 Colab link:
https://colab.research.google.com/drive/1xR8fcgUeEQ36fQSaWxRGxYodQ16215pO

## Table of Contents
- [Introduction](#introduction)
- [Live-URL](#Live-URL)
- [Task-2-Report](#Task-2-Report)
- [Audio-Sample](#Audio-Sample)
- [Environment Setup](#environment-setup)
- [Install](#Install)
- [Dataset-Description](#Dataset-Description)
- [My-Fine-tuning-Model](#My-Fine-tuning-Model)
- [Model-Description](#Model-Description)
- [Training-and-Fine-Tuning](#Training-and-Fine-Tuning)
- [Performance-Evaluation](#Performance-Evaluation)
- [License](#license)

## Introduction

This project presents a fine-tuned Italian Text-to-Speech (TTS) model based on Microsoft's SpeechT5 framework. Leveraging the VoxPopuli dataset, this model produces high-quality Italian speech synthesis, making it suitable for applications in natural language processing, Italian language learning, and automated voice systems. The Gradio interface allows for seamless interaction: users can input Italian text and receive clear, natural speech output. The repository includes code for deploying the model with Gradio, enabling both researchers and developers to experiment and integrate this Italian TTS capability into various applications easily.

## Environment Setup

### Requirements

- Necessary Libraries:
  - `gradio`
  - `transformers`
  - `datasets`
  - `soundfile`
  - `sentencepiece`

## Install

You can run SpeechT5 TTS locally with the Transformers library.

1. First install the [Transformers library](https://github.com/huggingface/transformers), sentencepiece, soundfile and datasets(optional):

```
pip install --upgrade pip
pip install --upgrade transformers sentencepiece datasets[audio]
```

2. Run inference via the `Text-to-Speech` (TTS) pipeline. You can access the SpeechT5 model via the TTS pipeline in just a few lines of code!

```python
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("Hello, my dog is cooler than you!", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
```

3. Run inference via the Transformers modelling code - You can use the processor + generate code to convert text into a mono 16 kHz speech waveform for more fine-grained control.

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute.", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
```

## Dataset-Description

![Screenshot 2024-10-27 114647](https://github.com/user-attachments/assets/6098042d-e5ea-476e-a78e-18386358a498)

## Model-Description

![Screenshot 2024-10-27 120041](https://github.com/user-attachments/assets/33367047-9922-4e57-928e-fe3ad9982476)

## Training-and-Fine-Tuning 

![Screenshot 2024-10-27 120652](https://github.com/user-attachments/assets/ad14656d-cad2-4442-9570-07c205ae4ac7)

## Performance-Evaluation 

![Screenshot 2024-10-27 120846](https://github.com/user-attachments/assets/b3eba732-3b8e-4199-b857-62c135580406)

## License

This project is licensed under the MIT License. See the following for details:

```
MIT License

Copyright (c) 2024 Vinay15

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```"






