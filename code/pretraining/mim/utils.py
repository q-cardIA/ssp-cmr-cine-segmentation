import torch


class PatchMasker:
    """Masks patches of images in a batch dictionary.

    Expects collated batch dictionary with image tensors and corresponding meta data
    dict. Splits each image into patches, masks some of them based on the given mode and
    masking probability, and stitches them back into an image. Saves a mask of which
    patches were masked.

    Attributes:
        patch_size (list[int, int]): size of each patch in pixels [height, width], the
            images must be exactly divisible by the patch size.
        masking_setting (str | float): mode of masking, can be:
            "gaussian_noise": fill masked patches with gaussian noise (mean=0, var=1).
            "shuffle_pixels_patch": shuffle all pixel intensities in the masked patches
                within each patch separately.
            "shuffle_pixels_image": shuffle all pixel intenisties in the masked patches
                for each image separately.
            "shuffle_pixels_batch": shuffle all pixel intensities in the masked patches
                across the batch.
            "shuffle_patches_image": shuffle the positions of the masked patches within
                each image separately.
            "shuffle_patches_batch": shuffle the positions of the masked patches across
                the batch.
            "replace_patches": replace each masked patch with a patch of a random other
                image in the batch with the same location.
            float: set all pixels in the masked patches to the given value.
        mask_ratio (float): ratio of patches to mask.
        ratio_as_prob (bool): interpret mask ratio as probability, resulting in varying
            numbers of masked patches per image. Defaults to False, ensuring the same
            number of masked patches per image.
    """

    def __init__(
        self,
        patch_size: list[int, int],
        masking_setting: str | float,
        mask_ratio: float,
        ratio_as_prob: bool = False,
    ):
        self.patch_size = torch.tensor(patch_size, dtype=torch.int32)
        self.masking_setting = masking_setting
        self.mask_ratio = mask_ratio
        self.ratio_as_prob = ratio_as_prob

    def __call__(self, batch_data: dict, image_key: str) -> dict:
        image_shape = torch.tensor(batch_data[image_key].shape[-2:], dtype=torch.int32)
        assert all(
            image_shape % self.patch_size == 0,
        ), f"patch size {self.patch_size} not valid for image shape {image_shape}"

        # convert image to patches, with corresponding mask
        # patches shape:
        #   [batch, nr_patches_y, nr_patches_x, channel, patch_height, patch_width]
        # mask shape:
        #   [batch, nr_patches_y, nr_patches_x]
        patches = self.patchify(batch_data[image_key].as_tensor())
        if self.ratio_as_prob:
            mask = torch.rand(patches.shape[:3]) < self.mask_ratio
        else:
            nr_patches = patches.size(1) * patches.size(2)
            nr_masked_patches = round(self.mask_ratio * nr_patches)
            rand_mask = torch.rand(patches.size(0), nr_patches)
            values = torch.topk(rand_mask, nr_masked_patches, 1, largest=False)[0]
            thresholds = values[:, -1][..., None, None]
            mask = rand_mask.reshape(patches.shape[:3]) <= thresholds

        if isinstance(self.masking_setting, float):
            # set to given value
            patches[mask] = self.masking_setting

        elif self.masking_setting == "gaussian_noise":
            # sample new values from a normal distribution with mean 0 and variance 1
            patches[mask] = torch.randn_like(patches[mask])

        elif self.masking_setting == "shuffle_pixels_patch":
            # shuffle masked pixels within each patch separately by geting flattened
            # pixel level subset vectors for each patch:
            # (varying) `nr_masked_patches` subsets of size `nr_pixels_per_patch`
            nr_pixels_per_patch = torch.prod(self.patch_size)
            nr_patches = torch.numel(patches[mask]) // nr_pixels_per_patch
            patch_sizes = torch.ones(nr_patches) * nr_pixels_per_patch

            # place shuffled flat pixel array back into appropriate masked patches
            self.masked_patches_pixel_shuffle(patches, mask, patch_sizes)

        elif self.masking_setting == "shuffle_pixels_image":
            # shuffle masked pixels within each image separately by getting flattened
            # pixel level subset vectors for each image:
            # `nr_images` subsets of (varying) size `nr_masked_pixels_per_image`
            nr_pixels_per_patch = torch.prod(self.patch_size)
            nr_masked_patches = mask.sum(1).sum(1)
            image_sizes = nr_masked_patches * nr_pixels_per_patch

            # place shuffled flat pixel array back into appropriate masked patches
            self.masked_patches_pixel_shuffle(patches, mask, image_sizes)

        elif self.masking_setting == "shuffle_pixels_batch":
            # shuffle masked pixels within the whole batch, shuffling all masked pixels
            masked_shape = patches[mask].shape
            pixel_array = patches[mask].flatten()
            permutation = torch.randperm(len(pixel_array))
            patches[mask] = pixel_array[permutation].reshape(masked_shape)

        elif self.masking_setting == "shuffle_patches_image":
            # get indices for each dimension separately
            mask_idxs = mask.nonzero(as_tuple=True)
            batch_idxs, y_idxs, x_idxs = mask_idxs

            # only change x and y location of patches within the same image, keeping x
            # and y index pairs together to only shuffle masked patches, by getting
            # flattened patch level subset vectors for each image:
            # `nr_images` subsets of (varying) size `nr_masked_patches_per_image`
            shuffled_idxs = self.rand_subset_permutations(mask.sum(1).sum(1))
            y_idxs = y_idxs[shuffled_idxs]
            x_idxs = x_idxs[shuffled_idxs]
            patches[mask] = patches[batch_idxs, y_idxs, x_idxs, :, :, :]

        elif self.masking_setting == "shuffle_patches_batch":
            # shuffle the positions of all masked patches across the batch
            patches[mask] = patches[mask][torch.randperm(mask.sum())]

        elif self.masking_setting == "replace_patches":
            # create indices for replacing patches
            batch_idxs, y_idxs, x_idxs = mask.nonzero(as_tuple=True)
            nr_patches = len(batch_idxs)
            batch_size = mask.size(0)

            # valid indices to replace patches without self-replacement
            idxs = torch.arange(batch_size)
            valid_idxs = torch.concat(
                [idxs[idxs != batch_idx].unsqueeze(0) for batch_idx in batch_idxs],
                dim=0,
            )

            # select a random valid index for each patch
            rand_indices = torch.randint(0, batch_size - 1, (nr_patches,))
            shuffled_batch_idxs = valid_idxs[range(nr_patches), rand_indices]
            patches[mask] = patches[shuffled_batch_idxs, y_idxs, x_idxs, :, :, :]

        else:
            raise ValueError(f"masking_setting {self.masking_setting} not recognized")

        # save masked image and mask in batch dictionary
        batch_data[f"{image_key}_masked"] = self.unpatchify(patches)
        mask = mask.repeat_interleave(self.patch_size[0], dim=1)
        mask = mask.repeat_interleave(self.patch_size[1], dim=2)
        batch_data[f"{image_key}_mask"] = mask.unsqueeze(1)

    def masked_patches_pixel_shuffle(
        self,
        patches: torch.tensor,
        mask: torch.tensor,
        subset_sizes: torch.tensor,
    ) -> torch.tensor:
        """Shuffle pixels in masked patches according to subset sizes.

        Args:
            patches (torch.tensor): patches tensor [b, y, x, c, ph, pw] to be updates
            mask (torch.tensor): mask tensor [b, y, x] indicating which patches to mask
            subset_sizes (torch.tensor): sizes of subsets to shuffle within
        """
        permutation = self.rand_subset_permutations(subset_sizes)
        reordered_pixels = patches[mask].flatten()[permutation]
        patches[mask] = reordered_pixels.reshape(patches[mask].shape)

    def rand_subset_permutations(self, subset_sizes: torch.Tensor) -> torch.Tensor:
        """Generate random permutation of each subset in a vector.

        Allows for shuffling of elements within each subset separately, while in a flat
        tensor with possibly varying sizes.

        example: sizes = tensor([4, 2, 3, 5]), could give the output indices:
            tensor([ 3,  1,  2,  0,  5,  4,  6,  8,  7, 10, 12, 11, 13,  9])

        split into the subsets of sizes given in the input tensor:
            3,  1,  2,  0       (size 4)
            5,  4               (size 2)
            6,  8,  7           (size 3)
            10, 12, 11, 13, 9   (size 5)

        Args:
            sizes (torch.Tensor): number of elements in each subset

        Returns:
            torch.Tensor: indices for a shuffle of each subset of the given sizes
        """
        subset_sizes = subset_sizes.long()  # ensure int64 datatype
        idx_bias = torch.zeros_like(subset_sizes)
        idx_bias[1:] = subset_sizes[:-1].cumsum(0)
        idx_bias = idx_bias.repeat_interleave(subset_sizes)
        return torch.concat([torch.randperm(nr) for nr in subset_sizes]) + idx_bias

    def patchify(self, image: torch.Tensor) -> torch.tensor:
        """Convert an image batch into a batch of patches.

        Changing shape from:
        [batch, channel, height, width]
        to
        [batch, nr_patches_y, nr_patches_x, channel, patch_height, patch_width]

        This order of dimensions makes subsequent operations easier.

        Args:
            image (torch.Tensor): input image tensor [b, c, h, w]

        Returns:
            torch.tensor: output patches tensor [b, y, x, c, ph, pw]
        """
        image = image.clone()
        new_shape = (
            image.size(0),  # batch (b)
            image.size(1),  # channel (c)
            image.size(-2) // self.patch_size[0].item(),  # nr patches in y (y)
            self.patch_size[0].item(),  # patch size height (h)
            image.size(-1) // self.patch_size[1].item(),  # nr patches in x (x)
            self.patch_size[1].item(),  # patch size width (w)
        )
        return torch.einsum("bcyhxw->byxchw", image.reshape(shape=new_shape))

    def unpatchify(self, image: torch.Tensor) -> torch.tensor:
        """Convert a batch of patches back into an image batch.

        Changing shape from:
        [batch, nr_patches_y, nr_patches_x, channel, patch_height, patch_width]
        to
        [batch, channel, height, width]

        Args:
            image (torch.Tensor): input patches tensor [b, y, x, c, ph, pw]

        Returns:
            torch.tensor: otuput image tensor [b, c, h, w]
        """
        image = image.clone()
        new_shape = (
            image.size(0),  # batch
            image.size(3),  # channel
            image.size(1) * image.size(4),  # image height
            image.size(2) * image.size(5),  # image width
        )
        return torch.einsum("byxchw->bcyhxw", image).reshape(shape=new_shape)
