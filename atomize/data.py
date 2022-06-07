import torch
import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
from torch.utils.data import DataLoader
from pedalboard.io import AudioFile
from torch.nn.utils.rnn import pad_sequence
import typing
from typing import List
import hydra

@functional_datapipe('audioload')
class AudioLoaderIterDataPipe(IterDataPipe):
    def __init__(self, filename_dp: IterDataPipe, duration=5) -> None:
        super().__init__()
        self.source_dp = filename_dp
        self.duration = duration
    def __iter__(self):
        for filename in self.source_dp:
            try:
                with AudioFile(filename, 'r') as f:
                    sample_rate = f.samplerate
                    duration = self.duration * sample_rate
                    if f.frames < duration:
                        # sample too short
                        continue
                    start_pos = torch.rand(1) * (f.frames-duration -1)
                    f.seek(int(start_pos))
                    audio = f.read(int(duration))
                yield audio[None, 0], sample_rate
            except (ValueError, RuntimeError) as e:
                # not an audio file
                continue
    def __len__(self):
        return len(self.source_dp)

@functional_datapipe('truncate')
class TruncaterIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe,
                 start, stop,
                ):
        self.source_dp = source_dp
        self.start = start
        self.stop = stop

    def __iter__(self):
        for x in self.source_dp:
            yield {
                key:x[key][self.start:self.stop] if (isinstance(x[key], torch.Tensor) and x[key].numel()>1)
                    else x[key]
                for key in x}

@functional_datapipe('atoms_shuffle')
class AtomsShufflerIterDataPipe(IterDataPipe):
    """
    shuffles the atoms within each sample of the datapipe.
    the new order is the same for all the keys.
    """
    def __init__(self, source_dp: IterDataPipe):
        self.source_dp = source_dp

    def __iter__(self):
        for x in self.source_dp:
            result = {}
            perm = None
            for key in x:
                if (isinstance(x[key], torch.Tensor) and x[key].numel()>1):
                    if perm is None:
                        perm = torch.randperm(len(x[key]))
                    result.update({key:x[key][perm]})
                else:
                    result.update({key:x[key]})
            yield result


@functional_datapipe('atoms_random_skip')
class AtomsRandomSkipperIterDataPipe(IterDataPipe):
    """
    skip a random number of first atoms within each sample
    of the datapipe for all the keys.
    """
    def __init__(self, source_dp: IterDataPipe, max_skip=1000):
        self.source_dp = source_dp
        self.max_skip = max_skip

    def __iter__(self):
        for x in self.source_dp:
            result = {}
            skipped = None
            for key in x:
                if (isinstance(x[key], torch.Tensor) and x[key].numel()>1):
                    if skipped is None:
                        skipped = torch.randint(self.max_skip, size=(1,) ).item()
                    result.update({key:x[key][skipped:]})
                else:
                    result.update({key:x[key]})
            yield result



@functional_datapipe('drop_keys')
class KeysDropperIterDataPipe(IterDataPipe):
    """
    drop keys from samples dict, returned samples are tupples as follows:
    (sample with missing keys, dict with dropped keys from sample)
    """
    def __init__(self, source_dp: IterDataPipe, keys: List[str]):
        self.source_dp = source_dp
        self.keys = keys

    def __iter__(self):
        for x in self.source_dp:
            yield (
                {key:x[key] for key in x if key not in self.keys},
                {key:x[key] for key in self.keys if key in x}
            )

class AtomizerDatapipeWrapper(IterDataPipe):
    def __init__(self, source_dp, atomizer, forward_direction=True):
        self.source_dp = source_dp
        self.atomizer = atomizer
        self.forward_direction = True

    def __iter__(self):
        for x in self.source_dp:
            if self.forward_direction:
                (audio, sample_rate) = x
                audio = torch.as_tensor(audio)
                yield self.atomizer.forward(audio[None])[0]
            else:
                yield self.atomizer.backward(x)


class AtomsDatapipe(IterDataPipe):
    """The main class for creating an IterDataPipe for atoms."""
    def __init__(
            self,
            audio_folder,
            atomizer,
            excerpts_length=3,
            random_skip=None,
            sample_size=10000,
            shuffle=True,
            num_atoms_context=10000,
            num_atoms_target=1000,
            target_keys=['times','freqs', 'signs', 'chans']
        ):
        # list files
        datapipe = dp.iter.FileLister(
            [audio_folder],
            recursive=True,
        )

        # audio loading part
        datapipe = datapipe.audioload(duration=excerpts_length)
        
        # atomizer part
        self.atomizer = atomizer
        datapipe = AtomizerDatapipeWrapper(datapipe, atomizer)

        # post processing part
        if random_skip:
           datapipe = datapipe.atoms_random_skip(random_skip)
        datapipe = datapipe.truncate(0, sample_size)
        if shuffle:
            datapipe = datapipe.atoms_shuffle()

        (context_dp, target_dp) = datapipe.fork(num_instances=2)

        context_dp = context_dp.truncate(0, num_atoms_context)
        target_dp = target_dp.truncate(
            num_atoms_context,
            num_atoms_context + num_atoms_target)
        target_dp, groundtruth_dp = target_dp.drop_keys(
            list(target_keys)).unzip(sequence_length=2)

        self.datapipe = context_dp.zip(target_dp, groundtruth_dp)

    def __iter__(self):
        for x in self.datapipe:
            yield x


def collate(batch):
    """
    collate function for sequences of atoms 

    batch is a list of `batchsize` items, where each item is a tuple of entries,
    typically item= (context, target, groundtruth)

    with each entry being dicts whose values are for each atom of this item/entry.
    
    """
    result = []
    for entry in range(len(batch[0])): # loop over context, target, groundtruth
        # loop over the keys
        result.append({
            key: pad_sequence([torch.atleast_1d(x[entry][key]) for x in batch], batch_first=True)
            for key in batch[0][entry]
        })
    return result

def uncollate(batch):
    """invert the `collate_sequence_dicts` function
    
    batch is a list of entries.

    Each entry is a dict of key:values, where each value has shape (batchsize, num_atoms)

    output is a list of length batchsize items. each of which is a list of entries.
    each entry is a dict whose values are tensors of length num_atoms
    """

    if isinstance(batch, dict):
        keys = list(batch.keys())
        batch_size = batch[keys[0]].shape[0]

        result = [
            {key:batch[key][item] for key in batch.keys()}
            for item in range(batch_size)
        ]
    else:    
        num_entries = len(batch)
        keys = list(batch[0].keys())
        batch_size = batch[0][keys[0]].shape[0]

        result = [
            [{key:batch[entry][key][item] for key in batch[entry].keys()} for entry in range(num_entries)]
            for item in range(batch_size)
        ]
    return result


def get_data(data_cfg):
    atomizer = hydra.utils.instantiate(data_cfg.atomizer)
    pipe = hydra.utils.instantiate(data_cfg.pipe, atomizer=atomizer)
    loader = hydra.utils.instantiate(data_cfg.loader, dataset=pipe, collate_fn=collate)

    return atomizer, pipe, loader
