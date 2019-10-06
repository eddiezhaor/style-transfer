
from flask import Flask, render_template,send_file, redirect, request,url_for
app = Flask(__name__)
import os
import io
import numpy as np
from PIL import Image

from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave



img_array_2 = np.array([[1]])
org_array_2 = np.array([[1]])
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
@app.route('/')
def home():
    global img_array_2
    global org_array_2
    img_array_2 = np.array([[1]])
    org_array_2 = np.array([[1]])
    return render_template('index.html',display1 = 'hidden',display2='hidden')

@app.route('/api/uploader', methods=['POST'])
def upload():
    global img_array_2
    if org_array_2.shape[1]>1:
        vis = 'visible'
    else:
        vis = 'hidden'
    f = request.files['file']
    img = Image.open(f)
    img = img.resize((224,224))
    #convert it into numpy array
    img_array = np.asarray(img,dtype='float32')
    #expand 1 dim,
    img_array_2 = np.expand_dims(img_array, axis=0)
    img_array_2 = preprocess_input(img_array_2)
    return  render_template('upload.html',display1 = 'visible', display2=vis)

@app.route('/api/uploader_original', methods=['POST'])
def orginal():
    global org_array_2
    if img_array_2.shape[1]> 1:
        vis = 'visible'
    else:
        vis = 'hidden'

    print(img_array_2.shape )
    s = request.files['file']
    img = Image.open(s)
    img = img.resize((224,224))
    org_array = np.asarray(img,dtype='float32')
    if org_array.shape[2]>3:
        org_array = np.asarray(org_array[:,:,:3],dtype='float32')
    else:  
        org_array  = np.asarray(org_array,dtype='float32')
    org_array_2 = np.expand_dims(org_array, axis=0)
    org_array_2 = preprocess_input(org_array_2)
    return render_template('combined.html',display1 =vis, display2='visible')
@app.route('/api/download')
def download():
    K.clear_session()
    height = 224
    width = 224
    global img_array_2
    global org_array_2
    content = K.variable(org_array_2)
    style = K.variable(img_array_2)
    combined = K.placeholder((1,height, width,3))

    input_tensor = K.concatenate([content,style,combined],axis=0)
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    #extract layers
    layers = dict([(layer.name, layer.output) for layer in model.layers])
    content_weight = 0.025
    style_weight = 1
    total_variation_weight = 1.0

    loss = K.variable(0.)
    def content_loss(content, combination):
        return K.sum(K.square(combination - content))

    layer_feature = layers['block2_conv2']
    content_feature = layer_feature[0,:,:,:]
    combined_feature =  layer_feature[2,:,:,:]
    loss += content_weight*content_loss(content_feature, combined_feature)

    def gram_matrix(x):
        flatten = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram_m = K.dot(flatten, K.transpose(flatten))
        return gram_m
    def style_loss(style,combination):
        style = gram_matrix(layer_feature[1,:,:,:])
        combination = gram_matrix(layer_feature[2,:,:,:])
        M = height*width
        return (K.sum(K.square(combination - style)))/(4*3*3*M**2)
        
    feature_layers = ['block1_conv2', 'block2_conv2',
                    'block3_conv3', 'block4_conv3',
                    'block5_conv3']
    for layer_name in feature_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features)
        loss += (style_weight / len(feature_layers)) * sl


    def total_variation_loss(x):
        a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
        b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
        return K.sum(K.pow(a + b, 1.25))

    loss += total_variation_weight * total_variation_loss(combined)

    grads = K.gradients(loss, combined)[0]
    grads = K.l2_normalize(grads)
    outputs=[loss]
    outputs.append(grads)
    f_outputs = K.function([combined],outputs)


    def loss_eval(m):
        m = m.reshape(1,height,width,3)
        loss_,grad_ = f_outputs([m])
        grad_ = grad_.flatten().astype('float64')
        return loss_,grad_


    import time
    x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.
    #loss_, grad_ = loss_eval(x)
    iterations = 10

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(loss_eval,x0=x.flatten(),
                                        maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
        
    x2 = np.reshape(x,(224,224,3))
    x2 = x2[:, :, ::-1]
    x2[:, :, 0] += 103.939
    x2[:, :, 1] += 116.779
    x2[:, :, 2] += 123.68

    x2 = np.clip(x2, 0, 255).astype('uint8')
    img = Image.fromarray(x2)
    # convert numpy array to PIL Image



    # img = Image.fromarray(arr.astype('uint8'))


    # create file-object in memory
    file_object = io.BytesIO()
    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)
    img_array_2 = np.array([[1]])
    org_array_2 = np.array([[1]])
    # return 
    return send_file(file_object,as_attachment=True, cache_timeout=0,attachment_filename='result.png')



if __name__ == '__main__':
    app.run(debug=True)