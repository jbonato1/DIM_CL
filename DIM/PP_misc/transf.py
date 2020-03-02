import numpy as np

# already included into loader
def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):
    """
    Here we get a batch of PIL imgs and we return them normalized as for
    the pytorch pre-trained models.

        Args:
            img_batch (tensor): batch of images.
            scale (bool): if we want to scale the images between 0 an 1.
            channel_first (bool): if the channel dimension is before of after
                                  the other dimensions (width and height).
            norm (bool): if we want to normalize them.
        Returns:
            tensor: pre-processed batch.

    """

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch