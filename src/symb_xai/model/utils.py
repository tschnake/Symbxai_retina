import torch
import numpy as np


def load_pretrained_weights(
        model,
        path
):
    params = torch.load(
        path,
        map_location=torch.device('cpu')
    )

    state_dict = model.state_dict()
    state_dict['bert.embeddings.position_embeddings.weight'] = params[
        'embeds.position_embeddings.weight']
    state_dict['bert.embeddings.token_type_embeddings.weight'] = params[
        'embeds.token_type_embeddings.weight']
    state_dict['bert.embeddings.word_embeddings.weight'] = params[
        'embeds.word_embeddings.weight']
    state_dict['bert.embeddings.position_ids'] = params[
        'embeds.position_ids']

    state_dict['bert.embeddings.LayerNorm.weight'] = params[
        'embeds.LayerNorm.weight']
    state_dict['bert.embeddings.LayerNorm.bias'] = params[
        'embeds.LayerNorm.bias']


    for i in range(3):
        state_dict['bert.encoder.layer.{}.self.query.weight'.format(i)] = params[
            'attention_layers.{}.query.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.self.key.weight'.format(i)] = params[
            'attention_layers.{}.key.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.self.value.weight'.format(i)] = params[
            'attention_layers.{}.value.weight'.format(i)]

        state_dict['bert.encoder.layer.{}.self.query.bias'.format(i)] = params[
            'attention_layers.{}.query.bias'.format(i)]
        state_dict['bert.encoder.layer.{}.self.key.bias'.format(i)] = params[
            'attention_layers.{}.key.bias'.format(i)]
        state_dict['bert.encoder.layer.{}.self.value.bias'.format(i)] = params[
            'attention_layers.{}.value.bias'.format(i)]

        state_dict['bert.encoder.layer.{}.output.dense.weight'.format(i)] = params[
            'attention_layers.{}.output.dense.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.output.dense.bias'.format(i)] = params[
            'attention_layers.{}.output.dense.bias'.format(i)]

        state_dict['bert.encoder.layer.{}.output.LayerNorm.weight'.format(i)] = params[
            'attention_layers.{}.output.LayerNorm.weight'.format(i)]
        state_dict['bert.encoder.layer.{}.output.LayerNorm.bias'.format(i)] = params[
            'attention_layers.{}.output.LayerNorm.bias'.format(i)]

        state_dict['bert.pooler.dense.weight'] = params['pooler.dense.weight']
        state_dict['bert.pooler.dense.bias'] = params['pooler.dense.bias']

        state_dict['classifier.weight'] = params['classifier.weight']
        state_dict['classifier.bias'] = params['classifier.bias']

    model.load_state_dict(state_dict)


def get_masked_patch_ids(image, mask, patch_size):
    """
    Returns unique IDs of patches that include parts of the segmentation mask.

    Args:
        image (np.array): Original image array.
        mask (np.array): Binary mask array with the same height and width as the image.
        patch_size (tuple): Size of each patch (height, width).
    
    Returns:
        List[int]: List of unique integer IDs for patches that contain the mask.
    """
    # Get image dimensions
    img_height, img_width = image.shape[:2]
    patch_height, patch_width = patch_size
    
    # Calculate the total number of patches along each dimension
    num_patches_y = (img_height + patch_height - 1) // patch_height
    num_patches_x = (img_width + patch_width - 1) // patch_width
    
    # Initialize list to store patch IDs containing the mask
    masked_patch_ids = []
    
    # Loop over the image in steps of the patch size
    for i in range(0, img_height, patch_height):
        for j in range(0, img_width, patch_width):
            # Calculate the current patch ID
            patch_id = (i // patch_height) * num_patches_x + (j // patch_width)
            
            # Extract the current patch from the mask
            mask_patch = mask[i:i + patch_height, j:j + patch_width]
            
            # Check if there are any mask pixels in this patch
            if np.any(mask_patch):
                # Store the patch ID
                masked_patch_ids.append(patch_id)
    
    return masked_patch_ids