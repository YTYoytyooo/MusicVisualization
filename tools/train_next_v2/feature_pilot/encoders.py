"""Offline, pinned encoder loading and physical-time audio windows."""
from __future__ import annotations

from collections import defaultdict
import gc
import importlib.machinery
import importlib.util
import sys
import types

import numpy as np

from tools.train_next.core import ROOT, sha256
from tools.train_next.features import centered_window
from tools.train_next_v2.common import BASE, read_json


RESOURCES = BASE / 'resources/20261003'
MERT_SOURCES = {
    'configuration_MERT': 'ae0ec2bab8f59c724ba9878a7c20b67210189536ea62d34a56775968e9decb03',
    'modeling_MERT': '6c3ee73cef6f0c30ef494f88d96f891fa6925ffe663fa391b512f4b57abecc6c',
}


def conv_geometry(kernels, strides):
    stride, receptive_field = 1, 1
    for kernel, step in zip(kernels, strides, strict=True):
        receptive_field += (kernel - 1) * stride
        stride *= step
    return stride, receptive_field


def frame_count(samples, kernels, strides):
    for kernel, stride in zip(kernels, strides, strict=True):
        samples = (samples - kernel) // stride + 1
    return max(0, int(samples))


def frame_centers(count, offset_seconds, sample_rate, stride, receptive_field):
    return offset_seconds + ((receptive_field - 1) / 2 + np.arange(count) * stride) / sample_rate


def central_indices(centers, timestamp):
    result = np.flatnonzero((centers >= timestamp - .25) & (centers < timestamp + .25))
    if not len(result):
        raise ValueError(f'No valid central frame at {timestamp}')
    return result


def clipped_window(audio, sample_rate, timestamp, seconds):
    start = round((timestamp - seconds / 2) * sample_rate)
    stop = start + round(seconds * sample_rate)
    start, stop = max(0, start), min(len(audio), stop)
    if stop <= start:
        raise ValueError('Empty source context')
    return audio[start:stop], start


def load_pinned_mert_classes():
    """Import only the exact statically reviewed local source bytes, offline."""
    directory = RESOURCES / 'pinned-model-files/mert95m'
    for name, expected in MERT_SOURCES.items():
        if sha256(directory / (name + '.py.txt')) != expected:
            raise ValueError('Pinned MERT source changed: ' + name)
    package_name = '_vm_pinned_mert_12af15fef9d0'
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(directory)]
        sys.modules[package_name] = package
        for name in MERT_SOURCES:
            full_name = package_name + '.' + name
            loader = importlib.machinery.SourceFileLoader(full_name, str(directory / (name + '.py.txt')))
            spec = importlib.util.spec_from_loader(full_name, loader)
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            loader.exec_module(module)
    return (sys.modules[package_name + '.configuration_MERT'].MERTConfig,
            sys.modules[package_name + '.modeling_MERT'].MERTModel)


class FrozenEncoder:
    def __init__(self, specification, device='cuda'):
        import torch
        self.name, self.device = specification['name'], torch.device(device)
        directory = RESOURCES / 'pinned-model-files' / self.name
        weight = RESOURCES / 'encoder-weights' / self.name / specification['revision'] / 'pytorch_model.bin'
        if weight.stat().st_size != specification['weight']['bytes'] or sha256(weight) != specification['weight']['sha256']:
            raise ValueError('Weight identity mismatch: ' + str(weight))
        if self.name == 'music-clap':
            from transformers import ClapConfig, ClapFeatureExtractor, ClapModel
            config = ClapConfig.from_dict(read_json(directory / 'config.json'))
            self.model = ClapModel(config)
            self.processor = ClapFeatureExtractor.from_pretrained(directory, local_files_only=True)
            self.rate, self.dimensions = 48000, 512
        elif self.name == 'mert95m':
            from transformers import Wav2Vec2FeatureExtractor
            config_class, model_class = load_pinned_mert_classes()
            config = config_class.from_dict(read_json(directory / 'config.json'))
            if config.deepnorm or config.feature_extractor_cqt:
                raise ValueError('Unaudited MERT branch enabled')
            # Select the reference attention implementation explicitly. No model
            # weights, architecture or source code are modified.
            config._attn_implementation = 'eager'
            self.model = model_class(config)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(directory, local_files_only=True)
            self.rate, self.dimensions = 24000, 768
            self.stride, self.receptive_field = conv_geometry(config.conv_kernel, config.conv_stride)
            if (self.stride, self.receptive_field) != (320, 400):
                raise ValueError('MERT convolution geometry differs from protocol')
            for layer, kernel, stride in zip(self.model.feature_extractor.conv_layers, config.conv_kernel, config.conv_stride, strict=True):
                conv = layer.conv
                if conv.kernel_size != (kernel,) or conv.stride != (stride,) or conv.padding != (0,) or conv.dilation != (1,):
                    raise ValueError('Runtime convolution geometry mismatch')
        else:
            raise ValueError('Unknown frozen encoder: ' + self.name)
        weights = torch.load(weight, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        self.weight_tensor_count = len(weights)
        del weights
        self.model.requires_grad_(False).eval().to(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def encode(self, audio, times, seconds, batch_size=4):
        """Return embeddings and source/frame metadata, in the requested order.

        MERT has temporal GroupNorm before its attention mask. Equal-length
        buckets avoid changing that normalization with synthetic batch padding.
        Tail batches may contain fewer than four real windows.
        """
        import torch
        windows, offsets = [], []
        for time in times:
            if self.name == 'music-clap':
                windows.append(centered_window(audio, self.rate, time, seconds))
                offsets.append(round((time - seconds / 2) * self.rate))
            else:
                window, offset = clipped_window(audio, self.rate, time, seconds)
                windows.append(window)
                offsets.append(offset)
        groups = defaultdict(list)
        for index, window in enumerate(windows):
            groups[len(window)].append(index)
        values = np.empty((len(times), self.dimensions), dtype=np.float32)
        metadata = [None] * len(times)
        batches = []
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                selected = indices[start:start + batch_size]
                crops = [windows[index] for index in selected]
                if self.name == 'music-clap':
                    if max(map(len, crops)) > self.processor.nb_max_samples:
                        raise ValueError('Refusing random CLAP truncation')
                    # Keep the pinned music model's Slaney/rand_trunc frontend.
                    # All crops <=10s, so the random truncation branch is unused.
                    inputs = self.processor(crops, sampling_rate=self.rate, return_tensors='pt', padding='repeatpad')
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        hidden = self.model.audio_model(input_features=inputs['input_features'],
                            is_longer=torch.zeros((len(selected), 1), dtype=torch.bool, device=self.device))
                        output = self.model.audio_projection(hidden.pooler_output).cpu().float().numpy()
                    for row, index in enumerate(selected):
                        values[index] = output[row]
                        first = offsets[index]
                        metadata[index] = dict(timestamp=float(times[index]), window_start_sample=first,
                            window_samples=len(windows[index]), source_start_sample=max(0, first),
                            source_end_sample=min(len(audio), first + len(windows[index])),
                            left_zero_padding_samples=max(0, -first),
                            right_zero_padding_samples=max(0, first + len(windows[index]) - len(audio)))
                else:
                    inputs = self.processor(crops, sampling_rate=self.rate, padding=True,
                                            return_attention_mask=True, return_tensors='pt')
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    sample_mask = inputs['attention_mask']
                    for row, crop in enumerate(crops):
                        if not torch.all(sample_mask[row, :len(crop)] == 1) or not torch.all(sample_mask[row, len(crop):] == 0):
                            raise ValueError('Audio attention mask is not right padded')
                    with torch.inference_mode():
                        output = self.model(**inputs, return_dict=True).last_hidden_state
                        frame_mask = self.model._get_feature_vector_attention_mask(output.shape[1], sample_mask)
                    for row, index in enumerate(selected):
                        count = frame_count(len(windows[index]), self.model.config.conv_kernel, self.model.config.conv_stride)
                        if int(frame_mask[row].sum()) != count or output.shape[1] != count:
                            raise ValueError('Invalid MERT frame count or padding mask')
                        centers = frame_centers(count, offsets[index] / self.rate, self.rate, self.stride, self.receptive_field)
                        pool = central_indices(centers, times[index])
                        values[index] = output[row, pool.tolist()].mean(0).cpu().float().numpy()
                        metadata[index] = dict(timestamp=float(times[index]), source_start_sample=offsets[index],
                            source_end_sample=offsets[index] + len(windows[index]), source_samples=len(windows[index]),
                            valid_frames=count, central_frames=len(pool), first_frame_center_seconds=float(centers[0]),
                            last_frame_center_seconds=float(centers[-1]), central_first_seconds=float(centers[pool[0]]),
                            central_last_seconds=float(centers[pool[-1]]), right_padding_samples=0)
                batches.append(len(selected))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite encoder output')
        return values, metadata, batches

    def unload(self):
        import torch
        del self.model, self.processor
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
