# Now support 1 image at a time.

import tensorflow as tf

def reconstruct(image):
    """Reconstruct image by resizing and random padding.
    
    Args:
        image: Input image tensor.
        
    Returns:
        image: Image after resize and padding.
    """
    image_size = max(image.shape.as_list()[-2], image.shape.as_list()[-2])
    resize_shape = tf.random_uniform((), int(0.8*image_size), image_size,
                                     dtype=tf.int32)
    rescaled = tf.image.resize_images(image, [resize_shape, resize_shape],
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    h_rem = image_size - resize_shape
    w_rem = image_size - resize_shape
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
                    pad_left, pad_right], [0, 0]])
    padded.set_shape((input_tensor.shape[0], image.shape.as_list()[-2], 
                      image.shape.as_list()[-2], 3))
    output = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.9),
                     lambda: padded, lambda: image)
    return output

def pgd_attack(model_in, model_infer_func, targets, epsilon=16):
    """PGD attack for embedding network.
    Build up the workflow from model out to adv examples.
    Aim at minimizing mean distance between model_in and targets.
    
    Args:
        model_in: Input variable of tf model.
        model_infer_func: Fuction that do model inference, in/ouput
                          are tf tensors.
        targets: List of target embedding vectors.
        epsilon: l-inf constraint.
        
    Returns:
        tf variable of attack result.
    """
    target_tensor = tf.constant(targets, dtype=tf.float32)
    
    def attack_step(image, perturb_former):
        """Inner loop of PGD attack
        Core components of attack step:
        (a) PGD attack (projecting perturb to epsilon ball).
        (b) Momentum for good direction of interative attack.
        (c) Resize and pad image radomly to increase robustness.
        (d) Add random noise to image.
        This will be used as body function of tf while loop.
        
        Args:
            image: Adv image from last iteration.
            perturb_former: Accumulated perturbation for momentum.
        
        Returns:
            adv_image: Adv image after this iteration
            perturb: Perturbation added on image.
        """
        orig_img = image # Save original image.
        
        # Reconsturct image by resizing and random padding.
        image = reconstruct(image)
        image = (image - 127.5) / 128.0
        image = image + tf.random_uniform(tf.shape(image), minival=-1e-2, maxval=1e-2)
        
        embeddings = model_infer_func(image)
        embeddings = tf.reshape(embeddings[0], 
                                [embeddings.shape.as_list()[-1], 1])
        objective = tf.reduce_mean(tf.matmul(targets, embeddings)) # to be maximized
        
        # Normalize perturb and momentum it.
        perturb = tf.gradients(objective, orig_img)
        perturb = perturb / tf.reduce_mean(tf.abs(perturb), [1,2,3], keep_dims=True)
        perturb = 0.9*perturb_former + perturb
        
        # Make sure image under l_inf constraint.
        adv_img = tf.clip_by_value(orig_img + tf.sign(perturb)*1.0,
                                   lower_bound, upper_bound)
        
        return adv_img, perturb
    
    # It's ok to define some varible in loop body after loop defination.
    model_in = tf.to_float(model_in)
    lower_bound = tf.clip_by_value(model_in - epsilon, 0., 255.)
    upper_bound = tf.clip_by_value(model_in + epsilon, 0., 255.)
    
    # Attack loop.
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        adv_img, _ = tf.while_loop(lambda _, __: True, attack_step,
                                   (model_in, tf.zeros_like(model_in)),
                                   back_prop=False,
                                   maximum_iterations=100,
                                   parallel_iterations=1)
        
    return adv_img