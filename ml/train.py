"""Train MobileNetV2 on a 38-folder PlantVillage dataset and export TFLite.
Example: python train.py --dataset /data/PlantVillage --output ../backend/models
"""
import argparse, base64, io, json, random
from pathlib import Path
import cv2, numpy as np, tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE=(224,224); BATCH=32; SEED=42
# Requested field-photo augmentation parameters: ±25° rotation, 20% zoom,
# horizontal flip and brightness range equivalent to [0.7, 1.3].
GEOMETRIC_AUGMENT=tf.keras.Sequential([
 tf.keras.layers.RandomRotation(25/360),
 tf.keras.layers.RandomFlip("horizontal"),
 tf.keras.layers.RandomZoom(height_factor=(-.2,.2),width_factor=(-.2,.2)),
 tf.keras.layers.RandomBrightness(.3, value_range=(0,255))
])
def motion_blur(image: np.ndarray) -> np.ndarray:
    """Random linear blur approximates camera movement in field captures."""
    size=random.choice([3,5,7]); kernel=np.zeros((size,size)); kernel[size//2,:]=1/size
    return cv2.filter2D(image,-1,kernel) if random.random()<.35 else image
def synthetic_background(image: np.ndarray) -> np.ndarray:
    """Replace weakly-green/white PlantVillage background with natural texture noise."""
    if random.random()>.45:return image
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV); green=cv2.inRange(hsv,(25,25,20),(95,255,255))
    # Leaf pixels are usually green; preserve them, blend non-leaf regions into a soil/foliage texture.
    bg=np.random.normal([105,100,65],[35,35,25],image.shape).clip(0,255).astype(np.uint8)
    mask=cv2.GaussianBlur(green,(9,9),0)[...,None]/255.; return (image*mask+bg*(1-mask)).astype(np.uint8)
def decode_augment(path,label,training=True):
    data=tf.io.read_file(path); image=tf.image.decode_jpeg(data,channels=3); image=tf.image.resize(image,IMG_SIZE); image=tf.cast(image,tf.uint8)
    if training:
        image=tf.numpy_function(lambda x: synthetic_background(motion_blur(x)),[image],tf.uint8); image.set_shape((*IMG_SIZE,3))
        image=GEOMETRIC_AUGMENT(tf.cast(image,tf.float32),training=True)
    return tf.keras.applications.mobilenet_v2.preprocess_input(tf.cast(image,tf.float32)),label
def make_ds(paths,labels,training=False):
    ds=tf.data.Dataset.from_tensor_slices((paths,labels));
    if training: ds=ds.shuffle(len(paths),seed=SEED)
    return ds.map(lambda p,y:decode_augment(p,y,training),num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH).prefetch(tf.data.AUTOTUNE)
def build_model(n):
    base=tf.keras.applications.MobileNetV2(input_shape=(*IMG_SIZE,3),include_top=False,weights='imagenet'); base.trainable=False
    x=tf.keras.layers.GlobalAveragePooling2D()(base.output); x=tf.keras.layers.Dropout(.3)(x); x=tf.keras.layers.Dense(256,activation='relu')(x); out=tf.keras.layers.Dense(n,activation='softmax')(x)
    return tf.keras.Model(base.input,out),base
def grad_cam(model,image_batch,class_index=None,last_conv_name='out_relu'):
    """Returns base64 PNG overlay. Use the H5 Keras model (not TFLite) for gradients."""
    conv=model.get_layer(last_conv_name); grad_model=tf.keras.Model(model.inputs,[conv.output,model.output])
    with tf.GradientTape() as tape:
        conv_out,preds=grad_model(image_batch); idx=tf.argmax(preds[0]) if class_index is None else class_index; loss=preds[:,idx]
    grads=tape.gradient(loss,conv_out); weights=tf.reduce_mean(grads,axis=(0,1,2)); heat=tf.reduce_sum(conv_out[0]*weights,axis=-1); heat=tf.maximum(heat,0)/(tf.reduce_max(heat)+1e-8)
    h=cv2.resize(heat.numpy(),IMG_SIZE); coloured=cv2.applyColorMap(np.uint8(255*h),cv2.COLORMAP_JET)
    original=np.uint8((image_batch[0].numpy()+1)*127.5); overlay=cv2.addWeighted(cv2.cvtColor(original,cv2.COLOR_RGB2BGR),.55,coloured,.45,0)
    ok,encoded=cv2.imencode('.png',overlay); return base64.b64encode(encoded).decode() if ok else ''
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',default='../backend/models');args=p.parse_args(); tf.keras.utils.set_random_seed(SEED)
 root=Path(args.dataset); classes=sorted(d.name for d in root.iterdir() if d.is_dir()); assert len(classes)==38,f'Expected 38 class folders; found {len(classes)}'
 paths=[]; labels=[]
 for i,c in enumerate(classes):
  for ext in ('*.jpg','*.JPG','*.jpeg','*.png'): paths.extend(str(x) for x in (root/c).glob(ext)); labels.extend([i]*len(list((root/c).glob(ext))))
 paths=np.array(paths);labels=np.array(labels); assert len(paths)>0,'No images found'
 train_p,temp_p,train_y,temp_y=train_test_split(paths,labels,test_size=.30,stratify=labels,random_state=SEED)
 val_p,test_p,val_y,test_y=train_test_split(temp_p,temp_y,test_size=.50,stratify=temp_y,random_state=SEED)
 print(f'Split: train={len(train_p)}, validation={len(val_p)}, test={len(test_p)}')
 weights=dict(enumerate(compute_class_weight('balanced',classes=np.arange(len(classes)),y=train_y)))
 model,base=build_model(len(classes)); model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 callbacks=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(patience=2)]
 model.fit(make_ds(train_p,train_y,True),validation_data=make_ds(val_p,val_y),epochs=10,class_weight=weights,callbacks=callbacks)
 base.trainable=True
 for layer in base.layers[:-40]: layer.trainable=False
 model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 model.fit(make_ds(train_p,train_y,True),validation_data=make_ds(val_p,val_y),epochs=15,class_weight=weights,callbacks=callbacks)
 yhat=np.argmax(model.predict(make_ds(test_p,test_y),verbose=1),axis=1); report=classification_report(test_y,yhat,target_names=classes,output_dict=True,zero_division=0); print(classification_report(test_y,yhat,target_names=classes,zero_division=0))
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True); model.save(out/'model.h5'); converter=tf.lite.TFLiteConverter.from_keras_model(model); (out/'model.tflite').write_bytes(converter.convert()); (out/'class_names.json').write_text(json.dumps(classes)); (out/'classification_report.json').write_text(json.dumps(report,indent=2)); print(f'Saved model and honest per-class report to {out.resolve()}')
if __name__=='__main__': main()
