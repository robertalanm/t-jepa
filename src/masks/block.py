import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()
class TextMaskCollator(object):

    def __init__(
        self,
        max_seq_len=512,
        mask_scale=(0.2, 0.8),
        nmask=2,
        min_keep=4,
        allow_overlap=False
    ):
        super(TextMaskCollator, self).__init__()

        self.max_seq_len = max_seq_len
        self.mask_scale = mask_scale
        self.nmask = nmask
        self.min_keep = min_keep  # minimum number of tokens to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_mask_size(self, generator, scale):
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.max_seq_len * mask_scale)
        return max_keep

    def _sample_mask(self, mask_size, acceptable_regions=None):

        def constrain_mask(mask, tries=0):
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            start = torch.randint(0, self.max_seq_len - mask_size, (1,))
            mask = torch.zeros((self.max_seq_len), dtype=torch.int32)
            mask[start:start+mask_size] = 1
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask)
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        mask_complement = torch.ones((self.max_seq_len), dtype=torch.int32)
        mask_complement[start:start+mask_size] = 0
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create masks when collating texts into a batch
        '''
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        mask_size = self._sample_mask_size(
            generator=g,
            scale=self.mask_scale)

        collated_masks = []
        min_keep = self.max_seq_len
        for _ in range(B):

            masks = []
            for _ in range(self.nmask):
                mask, mask_C = self._sample_mask(mask_size)
                masks.append(mask)
                min_keep = min(min_keep, len(mask))
            collated_masks.append(masks)

        collated_masks = [[cm[:min_keep] for cm in cm_list] for cm_list in collated_masks]
        collated_masks = torch.utils.data.default_collate(collated_masks)

        return collated_batch, collated_masks