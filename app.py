import torch 
import re 
import gradio as gr
from PIL import Image

from transformers import AutoTokenizer, ViTFeatureExtractor, VisionEncoderDecoderModel 
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device='cpu'

model_id = "nttdataspain/vit-gpt2-stablediffusion2-lora"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

# Predict function
def predict(image):
    img = image.convert('RGB')
    model.eval()
    pixel_values = feature_extractor(images=[img], return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]

input = gr.inputs.Image(label="Upload any Image", type = 'pil', optional=True)
output = gr.outputs.Textbox(type="text",label="Captions")
examples_folder = os.path.join(os.path.dirname(__file__), "examples")
examples = [os.path.join(examples_folder, file) for file in os.listdir(examples_folder)]

with gr.Blocks() as demo:
    
    gr.HTML(
        """
        <div style="text-align: center; max-width: 1200px; margin: 20px auto;">
        <h2 style="font-weight: 900; font-size: 3rem; margin: 0rem">
            📸 Image-to-Text with Awais Nayyar 📝
        </h2>
        <br>   
        </div>
        """)
    
    with gr.Row():
            with gr.Column(scale=1):
                # img = gr.inputs.Image(label="Upload any Image", type = 'pil', optional=True)
                img = gr.Image(label="Upload any Image", type = 'pil', optional=True)

                # img = gr.inputs.Image(type="pil", label="Upload any Image", optional=True)

                button = gr.Button(value="Convert")
            with gr.Column(scale=1):
                # out = gr.outputs.Textbox(type="text",label="Captions")
                out = gr.Label(type="text", label="Captions")

                
    button.click(predict, inputs=[img], outputs=[out])
 
    gr.Examples(
        examples=examples,
        inputs=img,
        outputs=out,
        fn=predict,
        cache_examples=True,
    )
demo.launch(debug=True)
