import torch
import torch.nn as nn
import math
import numpy as np

class Lipreading(nn.Module):
    def __init__( self, visual_frontend = None, landmarks_frontend = None, fusion_frontend = None, backend = None):
        super(Lipreading, self).__init__()

        assert not (visual_frontend is None and landmarks_frontend is None), "The frontend network is not defined!"
        if fusion_frontend is not None:
            assert visual_frontend is not None and landmarks_frontend is not None, "Fusion frontend requires both visual and landmarks frontends"
        else:
            assert visual_frontend is not None or landmarks_frontend is not None, "At least one of visual or landmarks frontends must be provided"
            assert not (visual_frontend and landmarks_frontend), "Either visual or landmarks frontends must be provided"



        self.visual_frontend = visual_frontend
        self.landmarks_frontend = landmarks_frontend
        self.fusion_frontend = fusion_frontend
        self.backend = backend

        self._initialize_weights_randomly()


    def forward(self, frames, landmarks, lengths):

        if self.visual_frontend is not None:
            v = self.visual_frontend(frames)
        if self.landmarks_frontend is not None:
            l = self.landmarks_frontend(landmarks)
        if self.fusion_frontend is not None:
            x = self.fusion_frontend(l, v)
        else:
            x = l if self.visual_frontend is None else v

        return self.backend(x, lengths, x.shape[0])

    def _initialize_weights_randomly(self):
        use_sqrt = True
        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

