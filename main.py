import flask, os, io, h5py
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pandas as pd
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from datetime import datetime
import time

# print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))

def DenoiseWavelet(data, type='BayesShrink'):
    def BayesShrink():
        im_bayes = denoise_wavelet(np.array(data), convert2ycbcr=True, multichannel=True,
                                  method='BayesShrink', mode='soft', 
                                  rescale_sigma=True, wavelet_levels=4)
        return im_bayes
    
    def VisuShrink():
        sigma_est = estimate_sigma(np.array(data), multichannel=True, average_sigmas=True)
        im_visu = denoise_wavelet(np.array(img), convert2ycbcr=True, multichannel=True,
                                  method='VisuShrink', mode='soft', wavelet_levels=4,
                                  sigma=sigma_est, rescale_sigma=True)
        
        return im_visu
    
    if type=='BayesShrink':
        return BayesShrink()
    elif type=='VisuShrink':
        return VisuShrink()

def create_feature(data_feature):
    rs = []
    data_denoise = DenoiseWavelet(data_feature, type='BayesShrink')
    dt = tf.keras.utils.timeseries_dataset_from_array(data=data_denoise, targets=None,
                                                          sequence_length=100, sequence_stride=20,
                                                          shuffle=False)
    for i in dt:
        rs.append(i)
        
    rs = tf.stack(rs)
    feature_rs = tf.data.Dataset.from_tensor_slices(rs)
    feature_rs = feature_rs.cache().batch(56).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return feature_rs

def get_predict_p(arr, threshold=.5, types='Highest'):
    arr = arr.reshape(-1)
    
    if types=='Highest':
        if np.max(arr)>=threshold:
            return np.argmax(arr)*20+275
        else:
            return -1
    elif types=='Early':
        for i, p_prob in enumerate(arr):
            if p_prob>=threshold:
                return i*20+275
        else:
            return -1
    elif types=='Late':
        i_thres = None
        for i, p_prob in enumerate(arr):
            if p_prob>=threshold:
                i_thres = i
        if i_thres!=None:
            return i_thres*20+275
        else:
            return -1

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_set, batch_size=196):
        self.x = x_set
        self.y = np.random.random(size=(x_set.shape))
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

# === LOAD SAVEDMODEL ===

class F1_Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign(2 *((p*r)/(p + r + 1e-6)))
        
    def result(self):
        return self.f1

    def reset_state(self):
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)

PATH = '/Models/P-Model'
p_model = tf.keras.models.load_model(PATH, custom_objects={'F1_Score':F1_Score,
                                                              'loss':tfa.losses.SigmoidFocalCrossEntropy()})

# === LOAD SAVEDMODEL ===

PATH = '/Models/S-Model'
s_model = tf.keras.models.load_model(PATH)

X = np.load('/stead_indonesia_single_wavelength.npz')['arr_0']

start_ = 35075
end_ = start_+1200
epochs = 352
epochs_loop = 0

is_p_detected = False
is_s_detected = False

is_p_time_detected = False
is_s_time_detected = False

temp_s, temp_mag = 0, 0

p_s_mag_prediction = {}

while True:
    try:
        sub_p_s_mag_prediction = {}

        mult = 100
        data_pred = create_feature(X[start_:end_, :])
        y_pred = p_model.predict(data_pred)

        high_ = get_predict_p(y_pred, types='Highest')
        early_ = get_predict_p(y_pred, types='Early')
        late_ = get_predict_p(y_pred, types='Late')

        if (high_ and early_ and late_) != -1:
            if is_p_time_detected==False:
                p_time = datetime.now()
                p_year = p_time.year
                p_month = p_time.month
                p_day = p_time.day
                p_date = f'{p_day}/{p_month}/{p_year}'

                p_hour = p_time.hour
                p_minute = p_time.minute
                p_second = p_time.second
                p_timestamp = p_hour*3600+p_minute*60+p_second
                is_p_time_detected = True

            is_p_detected = True

        if is_p_detected and ((high_ and early_ and late_) == -1):
            data_pred_np = DenoiseWavelet(X[start_-6000:start_, :], type='BayesShrink')
            data_pred_np = np.expand_dims(data_pred_np, axis=0)
                        
            data_pred = DataGenerator(data_pred_np, batch_size=196)
            y_pred = s_model.predict(data_pred.x).flatten()

            pred_s, pred_mag = y_pred[0], y_pred[1]
            pred_s = np.exp(pred_s)
            pred_mag = np.exp(pred_mag)-1
                        
            is_s_detected = True
            
            s_time = datetime.now()
            s_year = s_time.year
            s_month = s_time.month
            s_day = s_time.day
            s_date = f'{s_day}/{s_month}/{s_year}'

            s_hour = s_time.hour
            s_minute = s_time.minute
            s_second = s_time.second
            s_timestamp = s_hour*3600+s_minute*60+s_second
                                    
        if is_p_detected==False and is_s_detected==False:
            sub_p_s_mag_prediction.update({
                'P-Wave Date':-1,
                'P-Wave TimeStamp':-1,
                'S-Wave Time':-1,
                'S-Wave TimeStamp':-1,
                'Radius':-1,
                'Latitude':-8.4702,
                'Longitude':114.1521,
            })
            p_s_mag_prediction.update({epochs_loop+1:sub_p_s_mag_prediction})

        elif is_p_detected==True and is_s_detected==False:
            sub_p_s_mag_prediction.update({
                'P-Wave Date':p_date,
                'P-Wave TimeStamp':p_timestamp,
                'S-Wave Date':-1,
                'S-Wave TimeStamp':-1,
                'Radius':-1,
                'Latitude':-8.4702,
                'Longitude':114.1521,
            })
            p_s_mag_prediction.update({epochs_loop+1:sub_p_s_mag_prediction})

        elif is_p_detected==True and is_s_detected==True:
            sub_p_s_mag_prediction.update({
                'P-Wave Date':p_date,
                'P-Wave TimeStamp':p_timestamp,
                'S-Wave Date':s_date,
                'S-Wave TimeStamp':s_timestamp,
                'Radius':(s_timestamp-p_timestamp)*8.4,
                'Latitude':-8.4702,
                'Longitude':114.1521,
            })
            p_s_mag_prediction.update({epochs_loop+1:sub_p_s_mag_prediction})
            break
            
        epochs_loop+=1
        epochs+=1
        start_+=mult
        end_+=mult
        time.sleep(1.25)

        
    except Exception as e:
        print(e)
        break

#create an instance of Flask
app = flask.Flask('Earthquake Model Deployment')
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/')
def home(): 
    return flask.jsonify(p_s_mag_prediction)

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)
