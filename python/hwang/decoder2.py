from ._python import *
import numpy as np

# adapted from the plain Decoder,
# modified the to reduce retrieve() overhead for random sampling
# over long videos encoded with short intervals between keyframes

class Decoder2(object):
    def __init__(self,
                 f_or_path,
                 video_index=None,
                 device_type=DeviceType.CPU,
                 device_id=0):
        if video_index is None:
            video_index = hwang.index_video(f_or_path)
        self.video_index = video_index

        if isinstance(f_or_path, str):
            f = open(f_or_path, 'rb')
        else:
            f = f_or_path
        self.f = f
        self.vlen = self.video_index.frames()


        # Setup decoder
        handle = DeviceHandle()
        handle.type = device_type
        handle.id = device_id
        decoder_type = VideoDecoderType.SOFTWARE
        if device_type == DeviceType.GPU:
            decoder_type = VideoDecoderType.NVIDIA
        self._decoder = DecoderAutomata(handle, 1, decoder_type)

        # Setup arrays
        # add sentinels to array to simplify code.
        self.keyframes = np.array(
            [-1] + [k for k in self.video_index.keyframe_indices()] + [self.vlen + 1, self.vlen + 2])

        sample_offsets = self.video_index.sample_offsets()
        sample_sizes = self.video_index.sample_sizes()
        sample_offsets.append(sample_offsets[-1] + sample_sizes[-1])
        sample_sizes.append(0)

        self.sample_offsets = np.array(sample_offsets)
        self.sample_sizes = np.array(sample_sizes)

    def _get_keyframes_between(self, start_index, end_index):
        assert end_index <= self.vlen
        assert start_index >= 0

        #           keyframes = [
        #                 k for k in self.video_index.keyframe_indices()
        #                 if k >= start_index and k <= end_index
        #             ]
        s, e = np.searchsorted(self.keyframes, [start_index, end_index], side='left')
        assert self.keyframes[s - 1] < start_index
        assert self.keyframes[e] >= end_index
        assert self.keyframes[e + 1] > end_index

        if self.keyframes[e] == end_index:
            e = e + 1

        incl = self.keyframes[s:e]
        assert (incl >= start_index).all() and (incl <= end_index).all()
        return incl

    def retrieve(self, rows):
        # Grab video index intervals
        video_intervals = slice_into_video_intervals(self.video_index, rows)
        frames = []

        for (start_index, end_index), valid_frames in video_intervals:
            # Figure out start and end offsets
            start_offset = self.sample_offsets[start_index]
            end_offset = (self.sample_offsets[end_index] + self.sample_sizes[end_index])

            # Read data buffer
            self.f.seek(start_offset, 0)
            encoded_data = self.f.read(end_offset - start_offset)

            data = EncodedData()
            data.width = self.video_index.frame_width()
            data.height = self.video_index.frame_height()
            data.start_keyframe = start_index
            data.end_keyframe = end_index
            data.sample_offsets = self.sample_offsets[start_index:end_index] - start_offset
            data.sample_sizes = self.sample_sizes[start_index:end_index]
            data.valid_frames = valid_frames
            data.keyframes = self._get_keyframes_between(start_index, end_index)
            data.encoded_video = encoded_data
            args = [data]
            self._decoder.initialize(args, self.video_index.metadata_bytes())
            frames += self._decoder.get_frames(self.video_index,
                                               len(valid_frames))

        return frames