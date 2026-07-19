import base64, io, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from .config import settings

class Predictor:
    def __init__(self): self.interpreter=None; self.classes=[]; self.keras=None
    def load(self):
        if self.interpreter is not None:return
        if not Path(settings.model_path).exists(): raise RuntimeError("No trained TFLite model is installed. Run ml/train.py and copy its output into backend/models.")
        try:
            import tensorflow as tf
        except ModuleNotFoundError as error:
            raise RuntimeError("TensorFlow is not installed. Install backend dependencies before running image predictions.") from error
        self.interpreter=tf.lite.Interpreter(model_path=settings.model_path); self.interpreter.allocate_tensors(); self.classes=json.loads(Path(settings.model_info_path).read_text())
        if Path(settings.keras_model_path).exists(): self.keras=tf.keras.models.load_model(settings.keras_model_path,compile=False)
    def predict(self, raw:bytes):
        self.load(); image=Image.open(io.BytesIO(raw)).convert("RGB"); thumbnail=image.copy(); thumbnail.thumbnail((180,180)); buff=io.BytesIO(); thumbnail.save(buff,"JPEG",quality=80)
        # The integrated PlantVillage CNN was trained with torchvision's
        # ImageNet normalization, not MobileNetV2's [-1, 1] normalization.
        resized=image.resize((224,224)); array=np.asarray(resized,dtype=np.float32)/255.0
        normalized=np.expand_dims((array-np.array([0.485,0.456,0.406],dtype=np.float32))/np.array([0.229,0.224,0.225],dtype=np.float32),0)
        inp=self.interpreter.get_input_details()[0];out=self.interpreter.get_output_details()[0];self.interpreter.set_tensor(inp["index"],normalized);self.interpreter.invoke();probs=self.interpreter.get_tensor(out["index"])[0]
        index=int(np.argmax(probs)); label=self.classes[index]; split=label.replace("___","|").replace("_"," ").split("|",1); crop=split[0];disease=split[1] if len(split)>1 else label
        # This exported notebook model has no accompanying Keras graph, so a
        # Grad-CAM cannot be computed safely. The API keeps the field for UI compatibility.
        return {"className":label,"confidence":float(probs[index]),"crop":crop,"disease":disease,"thumbnailBase64":base64.b64encode(buff.getvalue()).decode(),"heatmapBase64":None}
    def gradcam(self,image,index):
        if self.keras is None:return None
        import tensorflow as tf
        last=next(layer for layer in reversed(self.keras.layers) if isinstance(layer,tf.keras.layers.Conv2D))
        grad_model=tf.keras.Model(self.keras.inputs,[last.output,self.keras.output])
        with tf.GradientTape() as tape:
            conv,pred=grad_model(image); loss=pred[:,index]
        weights=tf.reduce_mean(tape.gradient(loss,conv),axis=(0,1,2)); heat=tf.reduce_sum(conv[0]*weights,axis=-1);heat=tf.maximum(heat,0)/(tf.reduce_max(heat)+1e-8)
        colour=cv2.applyColorMap(np.uint8(255*cv2.resize(heat.numpy(),(224,224))),cv2.COLORMAP_JET); original=np.uint8((image[0]+1)*127.5); overlay=cv2.addWeighted(cv2.cvtColor(original,cv2.COLOR_RGB2BGR),.55,colour,.45,0);ok,png=cv2.imencode(".png",overlay)
        return base64.b64encode(png).decode() if ok else None
predictor=Predictor()
