"""
Create positional encoding embedder
"""
import torch
import torch.nn.functional as F
class Embedder:
    @staticmethod
    def get_by_name(embedder_type:str=None, *args, **kwargs):
        if embedder_type is None or embedder_type == 'None':
            embedder_type = "BasicEmbedder"
        try:
            embedder = eval(embedder_type)(*args, **kwargs)
        except:
            raise ValueError(f"Type {embedder_type} not recognized!")
        return embedder

class BasicEmbedder:
    """
    An embedder that do nothing to the input tensor.
    Used as a template.
    """
    def __init__(self, *args, **kwargs):
        """To be overwrited"""
        del args, kwargs
        self.out_dim = 0
    
    def embed(self, inputs):
        """To be overwrited"""
        return []

class FourierEncoding(BasicEmbedder):
    def __init__(self,
                 multires=0,
                 multiresZ=None,
                 input_dim=3,
                 embed_bases=[torch.sin, torch.cos],
                 include_input=True,
                 log_sample=True,
                 *args, **kwargs):
        super().__init__()
        self.multires = multires
        self.multiresZ = multiresZ
        self.in_dim = input_dim
        self.embed_bases = embed_bases
        self.include_input = include_input
        self.log_sample = log_sample

        self.freq_bands = None      # Will be initialized on first forward
        self.freq_bands_Z = None    # Optional for anisotropic 3D case

        self.out_dim = self._compute_out_dim()

    def _compute_out_dim(self):
        out_dim = 0
        if self.in_dim > 2 and self.multiresZ is not None:
            in_dim = 2
        else:
            in_dim = self.in_dim

        if self.include_input:
            out_dim += in_dim

        num_bands = self.multires
        out_dim += in_dim * num_bands * len(self.embed_bases)

        if self.multiresZ is not None:
            if self.include_input:
                out_dim += 1
            out_dim += 1 * self.multiresZ * len(self.embed_bases)

        return out_dim

    def _get_freq_bands(self, num_bands, device):
        max_freq = num_bands - 1
        if self.log_sample:
            return 2. ** torch.linspace(0., max_freq, steps=num_bands, device=device)
        else:
            return torch.linspace(2. ** 0., 2. ** max_freq, steps=num_bands, device=device)

    def embed(self, inputs):
        """
        inputs: [B, Q, D], D = 2 or 3
        return: [B, Q, out_dim]
        """
        device = inputs.device
        B, Q, D = inputs.shape

        out = []

        # isotropic dims
        if self.in_dim==2 or self.multiresZ is None :
            coord_main = inputs
        else:
            coord_main = inputs[...,1:]

        if self.include_input:
            out.append(coord_main)

        # get freq bands dynamically based on device
        self.freq_bands = self._get_freq_bands(self.multires, device)
        for freq in self.freq_bands:
            for fn in self.embed_bases:
                out.append(fn(coord_main * freq))

        # optional Z dimension (anisotropic)
        if self.multiresZ is not None:
            coord_z = inputs[..., 0:1]  # [B, Q, 1]
            if self.include_input:
                out.append(coord_z)

            self.freq_bands_Z = self._get_freq_bands(self.multiresZ, device)
            for freq in self.freq_bands_Z:
                for fn in self.embed_bases:
                    out.append(fn(coord_z * freq))

        return [torch.cat(out, dim=-1)]  # [B, Q, out_dim]

class VirtualProjectionEncoding:
    def __init__(self,
                 multires=6,
                 input_dim=3,
                 num_dirs=6,
                 embed_bases=[torch.sin, torch.cos],
                 include_input=True,
                 log_sample=True,
                 *args,
                 **kwargs
                 ):
        self.multires = multires
        self.in_dim = input_dim
        self.num_dirs = num_dirs
        self.embed_bases = embed_bases
        self.include_input = include_input
        self.log_sample = log_sample
        self.embed_fns = []
        self.out_dim = 0
        self._create_embed_fn()

    def _create_embed_fn(self):
        self.embed_fns = []
        d = 1  # since each projection becomes scalar
        if self.include_input:
            self.embed_fns.append(lambda x: x)
            self.out_dim += d

        if self.log_sample:
            freq_bands = 2. ** torch.linspace(0., self.multires - 1, steps=self.multires)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** (self.multires - 1), steps=self.multires)

        def create_fn(p_fn, freq):
            return lambda x: p_fn(x * freq)

        for freq in freq_bands:
            for p_fn in self.embed_bases:
                # self.embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                self.embed_fns.append(create_fn(p_fn, freq))
                self.out_dim += d

        self.out_dim *= self.num_dirs  # multiply by number of projection directions

    def _get_virtual_dirs(self, device):
        dirs = torch.tensor([
            [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 1], [-1, 1, 1], [1, -1, 1]
        ], dtype=torch.float32)
        dirs = F.normalize(dirs, dim=1)  # [num_dirs, 3]
        return dirs.to(device)

    def embed(self, inputs):  # inputs: [B, Q, 3]
        B, Q, _ = inputs.shape
        device = inputs.device
        dirs = self._get_virtual_dirs(device)  # [D, 3]

        out = []
        for d in dirs:
            proj = torch.matmul(inputs, d)  # [B, Q]
            proj = proj.unsqueeze(-1)  # [B, Q, 1]
            pe = [fn(proj) for fn in self.embed_fns]
            out.append(torch.cat(pe, dim=-1))  # [B, Q, C]
        return [torch.cat(out, dim=-1)]  # [B, Q, total_out_dim]