from fastai.vision.all import *
import gradio as gr

def is_player(x): return x[0]
learn = load_learner('model_basketball.pkl')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return f'{pred} with a probability of {probs[idx]:.04f}'
    
image = gr.inputs.Image()
label = gr.outputs.Label()

intf = gr.Interface(fn = classify_image, inputs = image, outputs = label)
intf.launch(inline = False)