""" pytorch implementation of pgd attack """
import torch
import torch.nn.functional as F
import numpy as np
import os
import sys

def reconstruct(model_input, resize_min=0.8):
    """ Reconstruct input sample by resizeing and padding.

    Args:
        model_input: Input tensor with same height and width.
        resize_min: Min resize factor for shape.

    Returns:
        Tensor after resize and padding.
    """
    input_shape = model_input.shape[-1]
    rand_factor = (1-resize_min)*torch.rand(1)+resize_min
    reshaped_input = F.interpolate(model_input.clone(),
                                   scale_factor=float(rand_factor))

    # computer padding parameters.
    size_diff = int(input_shape - reshaped_input.shape[-1])
    pad_left = int(torch.randint(0, size_diff, (1,)))
    pad_top = int(torch.randint(0, size_diff, (1,)))
    pad_right = size_diff - pad_left
    pad_bottom = size_diff - pad_top

    padding_method = torch.nn.ConstantPad2d((pad_left, pad_right,
                                             pad_top, pad_bottom), 0.)
    padded = padding_method(reshaped_input)

    # pad input with prob of 0.9
    return padded if (float(torch.rand(1))<0.9) else model_input

def cos_objective_func(source, targets):
    """ Cosine loss function.

    Args:
        source: Torch tensors.
        targets: Torch tensors.

    Returns:
        Mean cosine value of each source wrt target.
    """
    # m here for matrix.
    object_m = torch.mm(source, torch.t(targets))
    source_norm = torch.norm(source, 2, 1, keepdim=True)
    targets_norm = torch.norm(targets, 2, 1, keepdim=True)
    norm_m = torch.mm(source_norm, torch.t(targets_norm))

    return (object_norm/norm_m).mean()

def pgd_attack(victims_in, targets_in, model, device,
               model_in_min=0., model_in_max=255.,
               epsilon=16, max_iter=100, step_size=0.1):
    """ Attack a given model wrt input using pgd.

    Args:
        victims_in: Victims raw image tensors, range from 0~255.
        targets_in: Targets raw image tensors, range from 0~255.
        model: Torch model.
        device: Specify cpu or gpu to use.
        model_in_min: Input min value that model takes.
                      Typically 0. or -1.
        model_in_max: Input max value that model takes.
                      Typically 1. or 255.
                      We will resize input to this range during attack.
        epsilon: L inf perturb limit for attack in 0~255. It will be
                 resized same as victims and targets do.
        max_iter: Max iterations for attack loops.
        step_size: Attack step in each iteration.

    Returns:
        Perturbed victims in 0~255.
    """
    def preprocess(in_, min_, max_):
        return in_*(max_ - min_)/255.0 + min_

    # Prepare victims, targets, epsilon and model.
    victims_in = victims_in.to(device, dtype=torch.float32)
    targets_in = targets_in.to(device, dtype=torch.float32)
    victims_in = preprocess(victims_in, model_in_min, model_in_max)
    targets_in = preprocess(targets_in, model_in_min, model_in_max)
    epsilon = preprocess(epsilon, model_in_min, model_in_max)
    model = model.to(device)
    model.eval()

    # Initialize attack result & get targets embeddings.
    attack_result = victims_in.clone().detach()
    with torch.no_grad():
        targets_embeddings = model(targets_in)

    perturb = torch.zeros(victims_in.shape)

    # main attack loop
    for i in range(max_iter):
        """ PGD attack can be divided into following steps:
        1. Set random start point near attack result from last step.
        2. Perform 1-step FGSM attack.
        3. Project back to epsilon limit sapce.
        """
        # Set random start point.
        random_perturb = torch.FloatTensor(attack_result.shape).uniform(-1e-2, 1e-2) \
                                                               .to(device, dtype=Troch.float32)
        attack_result = attack_result + random_perturb
        attack_result = attack_result.clone().detach().requires_grad_(True) \
                                                      .to(device, dtype=torch.float32)
        
        # Perform 1-step FGSM attack.
        embeddings = model(attack_result)
        objective = cos_objective_func(embeddings, targets_embeddings)
        objective.backward()
        attack_result = attack_result + step_size*torch.sign(attack_result.grad)

        # Project back to epsilon limit space.
        perturb = attack_result - victims_in
        perturb = perturb/perturb.max() * epsilon
        attack_result = victims_in + perturb
        attack_result = torch.clamp(attack_result, model_in_min, model_in_max)

    # Resize tensor back to 0~255 for save.
    attack_result = (attack_result - model_in_min)/(model_in_max - model_in_min)*255.0

    return attack_result

