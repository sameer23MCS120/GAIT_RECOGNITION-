

import os
# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, applications, regularizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW, Adam
from sklearn.metrics import accuracy_score
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image as ExcelImage

# Settings
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 16
BUFFER_SIZE  = 1000
DATASET_DIR  = "/home/rishang/Documents/code/h/GEI_Image"
OUTPUT_DIR   = "/home/rishang/Documents/code/h/Results/GEI/Resnet50_Cosine_HPM_TTA_FewShot/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIEW_ANGLES  = ['000','018','036','054','072','090','108','126','144','162','180']
INITIAL_LR   = 1e-4
WEIGHT_DECAY = 1e-4

# TTA transforms
TTA_TRANSFORMS = [
    lambda x: x,
    lambda x: tf.image.flip_left_right(x),
    lambda x: tf.image.rot90(x, k=1),
    lambda x: tf.image.rot90(x, k=3),
]

# training‐only augmentation
augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal")], name="light_augmentation")

def channel_attention(input_feature, ratio=8):
    ch = input_feature.shape[-1]
    d1 = layers.Dense(ch//ratio, activation='relu', kernel_initializer='he_normal')
    d2 = layers.Dense(ch,           kernel_initializer='he_normal')
    avg = layers.GlobalAveragePooling2D()(input_feature)
    avg = layers.Reshape((1,1,ch))(avg)
    avg = d2(d1(avg))
    mx  = layers.GlobalMaxPooling2D()(input_feature)
    mx  = layers.Reshape((1,1,ch))(mx)
    mx  = d2(d1(mx))
    attn = layers.Activation('sigmoid')(layers.Add()([avg, mx]))
    return layers.Multiply()([input_feature, attn])

def spatial_attention(input_feature):
    avg = layers.Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_feature)
    mx  = layers.Lambda(lambda x: tf.reduce_max( x, axis=3, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=3)([avg, mx])
    attn   = layers.Conv2D(1, 7, padding='same', activation='sigmoid',
                           kernel_initializer='he_normal', use_bias=False)(concat)
    return layers.Multiply()([input_feature, attn])

def load_angle_dataset(subjects, condition, angle):
    paths, labels = [], []
    idx_map = {sid:i for i,sid in enumerate(subjects)}
    for sid in subjects:
        folder = os.path.join(DATASET_DIR, sid, condition, angle)
        if os.path.isdir(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith(('.png','.jpg')):
                    paths.append(os.path.join(folder,fname))
                    labels.append(idx_map[sid])
    def decode(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
        img.set_shape([None,None,1])
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.image.grayscale_to_rgb(img)
        img = tf.cast(img, tf.float32)/255.0
        lbl = tf.one_hot(label, depth=len(subjects))
        return img, lbl
    ds = tf.data.Dataset.from_tensor_slices((paths,labels))
    ds = ds.map(decode, tf.data.AUTOTUNE)
    ds = ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds, len(subjects)

def build_model(num_classes, num_parts=6, dropout_rate=0.5):
    inp = layers.Input((*IMAGE_SIZE,3))
    x   = augmentation(inp)
    base = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    for l in base.layers: l.trainable=False
    for l in base.layers:
        if l.name.startswith('conv5_') or isinstance(l, layers.BatchNormalization):
            l.trainable=True
    feat = channel_attention(base.output)
    feat = spatial_attention(feat)
    def hpm(f):
        feats=[]; h=tf.shape(f)[1]
        for i in range(num_parts):
            s=tf.cast(i*h//num_parts,tf.int32)
            e=tf.cast((i+1)*h//num_parts,tf.int32)
            p=f[:,s:e,:,:]
            feats.append(tf.reduce_mean(p,axis=[1,2]))
            feats.append(tf.reduce_max(  p,axis=[1,2]))
        return tf.concat(feats,axis=-1)
    y = layers.Lambda(hpm,name="hpm")(feat)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(512,activation='relu',kernel_regularizer=regularizers.l2(5e-4))(y)
    y = layers.Dropout(dropout_rate)(y)
    out = layers.Dense(num_classes, activation='softmax')(y)
    # optimizer + schedule
    total_steps=1000
    lr_sched = CosineDecayRestarts(INITIAL_LR, first_decay_steps=total_steps, t_mul=2., m_mul=1., alpha=0.)
    opt = AdamW(learning_rate=lr_sched, weight_decay=WEIGHT_DECAY)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=opt,
                  loss=CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])
    return model

def train_model(model, ds, epochs=25):
    n = tf.data.experimental.cardinality(ds).numpy()
    v = int(0.2*n)
    val_ds = ds.take(v)
    tr_ds  = ds.skip(v).map(lambda i,l:(tf.image.random_flip_left_right(i),l), tf.data.AUTOTUNE)
    cbs=[EarlyStopping('val_loss',patience=8,restore_best_weights=True)]
    model.fit(tr_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)
    return model

def adapt_batchnorm(model, ds, steps=10):
    for imgs, _ in ds.take(steps):
        _ = model(imgs, training=True)

def predict_with_tta(model, images):
    preds=[]
    for tfm in TTA_TRANSFORMS:
        aug = tfm(images)
        preds.append(model.predict(aug,verbose=0))
    return np.mean(preds,axis=0)

def evaluate(model, test_sets, subjects, angles, out_dir):
    for name, views in test_sets.items():
        # BN adapt on full target domain
        full_ds=None
        for v in views:
            ds,_ = load_angle_dataset(subjects, v, angles[0])
            full_ds = ds if full_ds is None else full_ds.concatenate(ds)
        adapt_batchnorm(model, full_ds, steps=10)

        records=[]
        for ta in angles:
            row={'Train_Angle':ta}
            for te in angles:
                # per-angle test ds
                tds=None
                for v in views:
                    ds,_=load_angle_dataset(subjects,v,te)
                    tds=ds if tds is None else tds.concatenate(ds)
                yt,yp=[],[]
                for imgs,lbls in tds:
                    preds = predict_with_tta(model,imgs.numpy())
                    yt.extend(np.argmax(lbls.numpy(),axis=1))
                    yp.extend(np.argmax(preds,axis=1))
                row[te]=accuracy_score(yt,yp)
            records.append(row)
        df=pd.DataFrame(records).set_index('Train_Angle').T
        pf=os.path.join(out_dir,f"acc_{name}.png")
        df.plot(figsize=(10,6),marker='o'); plt.title(f"TTA+BN Adapt+FewShot ({name})")
        plt.grid(True); plt.savefig(pf); plt.close()
        wb=Workbook(); ws=wb.active; ws.title=name
        for r in dataframe_to_rows(df.reset_index(),index=False,header=True): ws.append(r)
        img=ExcelImage(pf); img.anchor='H2'; ws.add_image(img)
        wb.save(os.path.join(out_dir,f"results_{name}.xlsx"))

# MAIN SCRIPT
# Stage 1
subjects1=[f"{i:03d}" for i in range(1,75)]
views1   =['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02']
ds1=None
for ang in VIEW_ANGLES:
    for v in views1:
        d,nc=load_angle_dataset(subjects1,v,ang)
        ds1=d if ds1 is None else ds1.concatenate(d)
model1=build_model(nc)
model1=train_model(model1, ds1, epochs=25)

# Stage 2
subjects2=[f"{i:03d}" for i in range(75,125)]
views2   =['nm-01','nm-02','nm-03','nm-04']
ds2=None
for ang in VIEW_ANGLES:
    for v in views2:
        d,nc2=load_angle_dataset(subjects2,v,ang)
        ds2=d if ds2 is None else ds2.concatenate(d)
model2=build_model(nc2)
for l1,l2 in zip(model1.layers,model2.layers):
    if l1.name==l2.name and l1.get_weights() and l2.get_weights():
        l2.set_weights(l1.get_weights())
model2=train_model(model2, ds2, epochs=25)

# Few‐shot fine‐tuning on bg/cl
views_fs=['bg-01','bg-02','cl-01','cl-02']
ds_fs=None
for ang in VIEW_ANGLES:
    for v in views_fs:
        d,_=load_angle_dataset(subjects2,v,ang)
        ds_fs=d if ds_fs is None else ds_fs.concatenate(d)
# freeze backbone, unfreeze head
for layer in model2.layers: layer.trainable=False
# last 4 layers are BN, Dense512, Dropout, Dense_out
for layer in model2.layers[-4:]: layer.trainable=True
model2.compile(optimizer=Adam(5e-6),
               loss=CategoricalCrossentropy(label_smoothing=0.1),
               metrics=['accuracy'])
model2.fit(ds_fs, epochs=10)

# Final Evaluation
test_sets = {
    'nm_05_06': ['nm-05','nm-06'],
    'bg_01_02': ['bg-01','bg-02'],
    'cl_01_02': ['cl-01','cl-02']
}
evaluate(model2, test_sets, subjects2, VIEW_ANGLES, OUTPUT_DIR)

print("✅ Done! (TTA + BN adaptation + few‐shot fine‐tuning)")

