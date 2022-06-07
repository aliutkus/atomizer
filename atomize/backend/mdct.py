import torch
import torchdata.datapipes as dp
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes import functional_datapipe
import mdct
from . import AtomFeature


class AtomizerMDCT:
    def __init__(self,
                 frame_length=2048,
                 dynamic_range_db=140,
                 sorted=True,
                 num_velocities=1024
                ):
        
        self.frame_length = frame_length
        self.dynamic_range_db = dynamic_range_db
        self.sorted = sorted
        self.num_velocities = num_velocities
        self.shape=None
        self.indices=None
        self.features = {
            'velocities': AtomFeature(cardinality=num_velocities, is_location=False),
            'freqs': AtomFeature(cardinality=frame_length//2, is_location=True),
            'chans': AtomFeature(cardinality=2, is_location=False),
            'signs': AtomFeature(cardinality=2, is_location=False),
            'times': AtomFeature(cardinality=None, is_location=True) # dynamic
        }

    def forward(self, batch):
        """
        encodes audio to atoms
        audio is a torch tensor with shape 
        (channels, samples) or (batchsize, channels, samples)
        

        returns a tuple of dicts. Each one for a sample
        """
        if len(batch.shape) == 2:
            # ensuring we have a batch dimension
            batch=batch[None]

        (batch_size, num_chans, samples) = batch.shape
        # group channels and batchsize for mdct and put them last
        batch = batch.cpu().numpy().reshape(batch_size * num_chans, samples).T 
        batch = torch.tensor(mdct.mdct(batch, framelength=self.frame_length))
        if len(batch.shape)==2:
            # mdct only returns 2D signal if number of channels is 1
            batch = batch[..., None] # we need a channel dimension
        (num_freqs, num_times) = batch.shape[:2]
        batch = batch.view(num_freqs, num_times, batch_size, num_chans)

        # make batch a tuple of mdct
        batch = batch.unbind(dim=2)

        result = []
        for x in batch:
            # compute the cartesian product of all freqs, times, chans
            # if not done or if the one we have does not match desired shape
            if not self.shape or self.shape != x.shape:
                self.shape=x.shape
                indices=torch.cartesian_prod( #Nx3
                    *[torch.arange(0,k) for k in self.shape])
                indices = indices.transpose(1, 0)
                self.indices = indices.unbind(0)

            # compute negvelocities (smaller is louder)
            velocities = x.abs()
            max_mag = velocities.max()
            velocities = velocities / max_mag
            eps = torch.tensor(10**(-self.dynamic_range_db / 20.))
            velocities = -20 * torch.log10(torch.maximum(velocities, eps))
            # extract the (possibly sorted velocities) and signs as lists
            if self.sorted:
                velocities, indices = velocities[self.indices].sort()
                indices = tuple([v[indices] for v in self.indices])
            else:
                velocities = velocities[self.indices]
                indices = self.indices

            # now making velocities integer
            velocities = velocities.clamp(0, self.dynamic_range_db)
            velocities = (velocities) * (self.num_velocities-1) / self.dynamic_range_db # from 0 to num_velocities-1
            velocities = velocities.to(int)

            signs = x.sign()[indices]
            # make signs -1 => 0 (negative), 1 => 1 (positive)
            signs = torch.where(signs < 0, 0, 1) 

            # output the atoms
            (freqs, times, chans) = indices

            result.append({
                # atom velocities
                'velocities':velocities,

                # atom parameters
                'freqs':freqs,
                'times':times,
                'chans':chans,
                'signs':signs,

                # masks
                'masks':torch.ones(velocities.numel()),

                # utilitary for reconstruction
                'misc_max_mags':max_mag,
                'misc_num_freqs':torch.tensor(x.shape[0]),
                'misc_num_chans':torch.tensor(x.shape[2]),
                'misc_num_times':torch.tensor(x.shape[1])
            })
        return result

    def backward(self, atoms):
        """a batch of atoms to a waveform
        atoms should be a list of dicts.
        
        output is a list of tensors with shape (num_channels, times)"""
        batchsize = len(atoms)
        result = []
        for sample in atoms:
            max_time = int(sample['times'].max() + 1)
            num_chans = int(sample['chans'].max() + 1)
            spec = torch.zeros(
                (sample['misc_num_freqs'], max_time, num_chans),
                dtype = torch.float64,
                device=sample['velocities'].device
            )
            # back from an int in [0, num_velocities-1] to [0, dynamic_range]
            velocities = sample['velocities'].to(spec.dtype)
            velocities = velocities.clamp(0, self.num_velocities-1)
            velocities = velocities * self.dynamic_range_db / (self.num_velocities - 1)
            # now go back from dB to linear
            values = 10**(-velocities / 20.) * sample['misc_max_mags']
            # multiply by sign (reajusted from {0, 1} to {-1, 1})
            values = values *  torch.where(sample['signs']==0, -1, 1)

            spec[(sample['freqs'], sample['times'], sample['chans'])] = values
            
            waveform = mdct.imdct(spec.cpu().numpy(), framelength=self.frame_length)
            waveform = torch.as_tensor(waveform)
            result.append(waveform.permute(1, 0)) # (times, num_chans) -> (num_chans, times)

        return result