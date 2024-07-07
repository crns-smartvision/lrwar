import math
import torch.nn as nn
import torch

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VisualFrontend(nn.Module):
    def __init__(self, cfg):
        super(VisualFrontend, self).__init__()
        frontend_relu = nn.PReLU(num_parameters=cfg.params.conv_3D.output_dim) if cfg.params.conv_3D.relu_type == 'prelu' else nn.ReLU()

        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, cfg.params.conv_3D.output_dim, kernel_size=tuple(cfg.params.conv_3D.kernel_size), stride=tuple(cfg.params.conv_3D.stride), padding=tuple(cfg.params.conv_3D.padding),
                      bias=cfg.params.conv_3D.bias),
            nn.BatchNorm3d(cfg.params.conv_3D.output_dim),
            frontend_relu,
            nn.MaxPool3d(kernel_size=tuple(cfg.params.MaxPool3D.kernel_size), stride=tuple(cfg.params.MaxPool3D.stride), padding=tuple(cfg.params.MaxPool3D.padding))
        )

        # for the moment BasicBlock is the only block used
        self.resnet_trunk = ResNet(BasicBlock, cfg.params.ResNet.layers, input_dims = cfg.params.ResNet.layers_input_dims, relu_type=cfg.params.ResNet.relu_type)



    def forward(self, x):
        B, C, _, H, W = x.size()
        x = self.frontend3D(x)
        T = x.shape[2]  # output should be B x C2 x T x H x W
        n_batch, n_channels, s_time, sx, sy = x.shape
        x = x.transpose(1, 2)
        x = x.reshape(n_batch * s_time, n_channels, sx, sy)
        x = self.resnet_trunk(x)
        x = x.view(B, T, x.size(1))

        return x

class LandmarksFrontend(nn.Module):
    def __init__(self, cfg):
        super(LandmarksFrontend, self).__init__()

        self.num_landmarks = cfg.params.num_landmarks
        self.gat = GAT()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model= cfg.params.transformer_encoder_layer.hidden_dim,
                                                        nhead=cfg.params.transformer_encoder_layer.num_heads,
                                                        dim_feedforward=cfg.params.transformer_encoder_layer.feedforward_dim,
                                                        dropout= cfg.params.transformer_encoder_layer.dropout)

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=cfg.params.transformer_encoder_layer.num_layers)

        self.positionalEncoding = PositionalEncoding(dModel=cfg.params.positional_encoding.hidden_dim,
                                                     maxLen=cfg.params.positional_encoding.max_len)

        self.fc = nn.Linear(cfg.params.num_landmarks*cfg.params.fc.hidden_dim, cfg.params.fc.output_dim)


    def forward(self, landmarks):
        # TODO change x to landmarks to get the sizes
        seq_length = landmarks.shape[1]
        batch_size = landmarks.shape[0]

        landmarks_data = torch.tensor(landmarks, dtype=torch.float32).to(device)
        landmarks_data = landmarks_data.view(batch_size * seq_length, landmarks_data.size(2), landmarks_data.size(3))
        adj = adjacency_k(5, self.num_landmarks)
        dataset = []
        for idx in range(batch_size * seq_length):
            l = Data(x=landmarks_data[idx], edge_index=adj.nonzero().to(device), y=landmarks_data[idx])
            dataset.append(l)

        dataloader = DataLoader(dataset, batch_size=batch_size * seq_length, shuffle=False)
        l_out = torch.empty([len(dataloader),batch_size * seq_length, self.num_landmarks, 64]).to(device)
        for i, data in enumerate(dataloader):
            l_new, edge_index = data.x, l.edge_index
            edge_index = edge_index.transpose(0, 1).contiguous()  # transpose edge_index to shape (2, num_edges)

            embedding = self.gat(l_new, edge_index)
            l_out[i] = embedding.view(batch_size * seq_length,self.num_landmarks, -1)

        out_tensor = l_out.view(batch_size, seq_length, -1)
        out_tensor = self.fc(out_tensor)
        out_tensor = out_tensor.transpose(0, 1)
        out_tensor = self.positionalEncoding(out_tensor)
        out_tensor = self.transformer_encoder(out_tensor)
        out_tensor = out_tensor.transpose(0, 1)
        return out_tensor

class FusionFrontend(nn.Module):
    def __init__(self, cfg):
        super(FusionFrontend, self).__init__()
        self.AttentionFusion = AttentionFusion(cfg.params.input_dim, cfg.params.num_heads)

    def forward(self, x_landmarks, x_video):
        return self.AttentionFusion(x_landmarks, x_video)


class PositionalEncoding(nn.Module):
    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, input_batch):
        outputBatch = input_batch + self.pe[:input_batch.shape[0],:,:]
        return outputBatch

class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.conv1 = GATConv(2, 16)
        self.conv2 = GATConv(16, 64)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        return x


class AttentionFusion(nn.Module):
    def __init__(self, fusion_dim, num_heads):
        super(AttentionFusion, self).__init__()

        # Define attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=fusion_dim, num_heads=num_heads)

    def forward(self, landmark_input, vision_input):
        # Compute attention weights and fused features
        fusion_vision, _ = self.attention(vision_input, landmark_input, vision_input)
        fusion_landmarks, _ = self.attention(landmark_input, vision_input, landmark_input)

        fusion = torch.cat((fusion_vision, fusion_landmarks), dim=2)
        return fusion


def adjacency_k(k, number_landmarks):
    # create the adjacency matrix for the k-nearest neighbor graph
    points = torch.randn(number_landmarks, 2)
    distances = torch.cdist(points, points)
    _, indices = torch.topk(distances, k=k, largest=False)
    indices = indices.reshape(-1)
    adj = torch.zeros((number_landmarks, number_landmarks))
    adj[torch.arange(number_landmarks).repeat_interleave(k), indices] = 1
    return adj

# 2D ResNet
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def downsample_basic_block(input_dim, output_dim, stride):
    return nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(output_dim),
    )

def downsample_basic_block_with_average_pooling(input_planes, out_planes, stride):
    return nn.Sequential(
        nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
        nn.Conv2d(input_planes, out_planes, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(out_planes),
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, downsample=None, relu_type='relu'):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu', 'prelu']

        self.conv1 = conv3x3(input_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)

        # type of ReLU is an input option
        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')
        # --------

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, input_dims, relu_type='relu', gamma_zero=False):
        self.input_dims = input_dims
        self.input_dim = input_dims[0]
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, self.input_dims[0], layers[0])
        self.layer2 = self._make_layer(block, self.input_dims[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.input_dims[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.input_dims[3], layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # nn.init.ones_(m.weight)
                # nn.init.zeros_(m.bias)

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        downsample = None
        if stride != 1 or self.input_dim != planes * block.expansion:
            downsample = self.downsample_block(input_dim=self.input_dim,
                                               output_dim=planes * block.expansion,
                                               stride=stride)

        layers = []
        layers.append(block(self.input_dim, planes, stride, downsample, relu_type=self.relu_type))
        self.input_dim = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.input_dim, planes, relu_type=self.relu_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
