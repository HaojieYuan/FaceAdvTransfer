import tensorflow as tf
import cv2
import os
import sys
import numpy as np
import facenet.src.align.detect_face as FaceDet
from scipy import misc
import torch


def load_imgdir(source_imgdir, resize=None):
    """
    Load images in source dir for further use.
    
    Args:
        source_imgdir: Path where victim images stored.
        resize: Image width and height.
    
    Returns:
        A numpy array (nb_imgs, width, height, channel).
        Images will be read in RGB mode.
    """
    assert os.path.isdir(source_imgdir)
    img_paths = [os.path.join(source_imgdir, image_name) for \
                image_name in os.listdir(source_imgdir)]
    if resize:
        imgs = [cv2.resize(cv2.imread(img_path), (resize, resize)) \
                for img_path in img_paths]
    else:
        imgs = [cv2.imread(img_path) for img_path in img_paths]
    imgs = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) \
                    for image in imgs])
    
    return imgs
    
def infer_batch(img_batch, sess, model_in, model_out):
    return sess.run(model_out, feed_dict={model_in: img_batch})

def get_iteration(nb, batch_size):
    """
    Given total sample size and batch size, return
    number of batch.
    """
    basic = nb//batch_size
    total = basic + (0 if nb%batch_size==0 else 1)
    return  total

def infer_imgs(img_list, session, model_in, model_out, batch_size=16):
    """Calculate image embeddings.
    
    Args:
        img_list: Python list or numpy array of input images.
        session: Tensorflow model session.
        model_in: Input variable of embedding model.
        model_out: Output variable of embedding model.
        batch_size: Batch_size for model infer.
        
    Returns:
        Numpy array of embedding results.
    """
    result = np.empty((0, *model_out.get_shape().as_list()[1:]))
    iteration = get_iteration(len(img_list), batch_size)
    i = -1 # In order iteration is 1
    for i in range(iteration-1):
        result = np.append(result, 
                           infer_batch(img_list[i*batch_size:(i+1)*batch_size], 
                                       session, model_in, model_out), axis=0)
    result = np.append(result, 
                       infer_batch(img_list[(i+1)*batch_size:], 
                                   session, model_in, model_out), axis=0)
    
    return result


class Detector():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                                    log_device_placement=False))
            with sess.as_default():
                self.pnet, self.rnet, self.onet = FaceDet.create_mtcnn(sess, None)

    def detect(self, img):
        """
        img: rgb 3 channel
        """
        minsize = 20  # minimum size of face
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor

        bounding_boxes, _ = FaceDet.detect_face(
                img, minsize, self.pnet, self.rnet, self.onet, threshold, factor)
        num_face = bounding_boxes.shape[0]
        assert num_face == 1, num_face
        bbox = bounding_boxes[0][:4]  # xy,xy

        margin = 32
        x0 = np.maximum(bbox[0] - margin // 2, 0)
        y0 = np.maximum(bbox[1] - margin // 2, 0)
        x1 = np.minimum(bbox[2] + margin // 2, img.shape[1])
        y1 = np.minimum(bbox[3] + margin // 2, img.shape[0])
        x0, y0, x1, y1 = bbox = [int(k + 0.5) for k in [x0, y0, x1, y1]]
        cropped = img[y0:y1, x0:x1, :]
        scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
        
        return scaled, bbox



def remove_prefix_string(string, prefix):
    assert string.startswith(prefix), "can not remove prefix."
    return string[len(prefix):]

def remove_prefix_from_state_dict(state_dict, prefix):
    for old_key in list(state_dict.keys()):
        if old_key.startswith(prefix):
            new_key = remove_prefix_string(old_key, prefix)
            state_dict[new_key] = state_dict.pop(old_key)

def load_state_local(path, model, ignore=[], optimizer=None):
    def map_func(storage, location):
        return storage.cuda()
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        if len(ignore) > 0:
            assert optimizer == None
            for k in list(checkpoint['state_dict'].keys()):
                flag = False
                for prefix in ignore:
                     if k.startswith(prefix):
                         flag = True
                         the_prefix = prefix
                         break
                if flag:
                    print('ignoring {} (prefix: {})'.format(k, the_prefix))
                    del checkpoint['state_dict'][k]
        remove_prefix_from_state_dict(checkpoint['state_dict'], 'module.base.')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        keys1 = set(checkpoint['state_dict'].keys())
        keys2 = set([k for k,_ in model.named_parameters()])
        not_loaded = keys2 - keys1
        for k in not_loaded:
            print('caution: {} not loaded'.format(k))
        if optimizer != None:
            assert len(ignore) == 0
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (step {})".format(path, checkpoint['step']))
            return checkpoint['step']
    else:
        assert False, "=> no checkpoint found at '{}'".format(path)
        
def image2tensor(image_obj):
    """ After load image by Image.open
    Transfrom it into 4d torch tensor.
    Could only apply for a single image.
    """
    np_img = np.array(image_obj)
    np_img = np_img.reshape((-1,*np_img.shape))
    t_img = torch.from_numpy(np_img)
    
    return t_img.permute(0, 3, 1, 2)
