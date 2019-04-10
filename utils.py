import tensorflow as tf
import cv2
import os
import numpy as np
import facenet.src.align.detect_face as FaceDet
from scipy import misc

def load_imgdir(source_imgdir, resize=160):
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
    imgs = [cv2.resize(cv2.imread(img_path), (resize, resize)) \
            for img_path in img_paths]
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