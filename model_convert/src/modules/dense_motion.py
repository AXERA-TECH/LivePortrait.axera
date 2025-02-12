# coding: utf-8

"""
The module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
"""

from torch import nn
import torch.nn.functional as F
import torch
from .util import Hourglass, make_coordinate_grid, kp2gaussian

def bonds_check(val, posx, posy, posz, minx,miny,minz,maxx,maxy,maxz):
    condi = torch.where(posx > minx, 1, 0)
    condi = torch.where(posx < maxx, condi, 0)
    condi = torch.where(posy > miny, condi, 0)
    condi = torch.where(posy < maxy, condi, 0)
    condi = torch.where(posz > minz, condi, 0)
    condi = torch.where(posz < maxz, condi, 0)
    
    #print("condi shape {}, val shape {}".format(condi.shape, val.shape))
    n,c,d,h,w = val.shape
    out = condi.view(n,1,d,h,w).repeat(1,c,1,1,1) * val
    return out

def grid_sample_3d(input, grid, align_corners):
    N,C,ID,IH,IW = input.shape
    _,D,H,W,_=grid.shape
    print("input shape{}".format(input.shape))
    print("grid shape{}".format(grid.shape))
    
    ix = grid[...,0]
    iy = grid[...,1]
    iz = grid[...,2]
    
    if(align_corners == False):
        ix = ((ix + 1) * IW -1) / 2
        iy = ((iy + 1) * IH -1) / 2
        iz = ((iz + 1) * ID -1) / 2
    else:
        ix = ((ix + 1) / 2) * (IW - 1)
        iy = ((iy + 1) / 2) * (IH - 1)
        iz = ((iz + 1) / 2) * (ID - 1)
    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)
        
        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw
        
        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw
        
        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw
        
        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1
        
        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1
        
        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1
        
        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1
    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)
    
    print("bse {}".format(bse.shape))
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (ix_tnw.device.type != 'cpu'):
            zero_tensor = torch.tensor(0).float().to(device)
            iw_tensor = torch.tensor(IW - 1).float().to(device)
            ih_tensor = torch.tensor(IH - 1).float().to(device)
            id_tensor = torch.tensor(ID - 1).float().to(device)
        else:
            zero_tensor = torch.tensor(0).float()
            iw_tensor = torch.tensor(IW - 1).float()
            ih_tensor = torch.tensor(IH - 1).float()
            id_tensor = torch.tensor(ID - 1).float()
        ix_tnw_tmp = torch.where(ix_tnw < 0, zero_tensor, ix_tnw.float())
        ix_tnw_tmp = torch.where(ix_tnw > IW - 1, iw_tensor, ix_tnw_tmp.float())
        
        iy_tnw_tmp = torch.where(iy_tnw < 0, zero_tensor, iy_tnw.float())
        iy_tnw_tmp = torch.where(iy_tnw > IH - 1, ih_tensor, iy_tnw_tmp.float())
        
        iz_tnw_tmp = torch.where(iz_tnw < 0, zero_tensor, iz_tnw.float())
        iz_tnw_tmp = torch.where(iz_tnw > ID - 1, id_tensor, iz_tnw_tmp.float())
        
        ix_tne_tmp = torch.where(ix_tne < 0, zero_tensor, ix_tne.float())
        ix_tne_tmp = torch.where(ix_tne > IW - 1, iw_tensor, ix_tne_tmp.float())
        
        iy_tne_tmp = torch.where(iy_tne < 0, zero_tensor, iy_tne.float())
        iy_tne_tmp = torch.where(iy_tne > IH - 1, ih_tensor, iy_tne_tmp.float())
        
        iz_tne_tmp = torch.where(iz_tne < 0, zero_tensor, iz_tne.float())
        iz_tne_tmp = torch.where(iz_tne > ID - 1, id_tensor, iz_tne_tmp.float())
        
        ix_tsw_tmp = torch.where(ix_tsw < 0, zero_tensor, ix_tsw.float())
        ix_tsw_tmp = torch.where(ix_tsw > IW - 1, iw_tensor, ix_tsw_tmp.float())
        
        iy_tsw_tmp = torch.where(iy_tsw < 0, zero_tensor, iy_tsw.float())
        iy_tsw_tmp = torch.where(iy_tsw > IH - 1, ih_tensor, iy_tsw_tmp.float())
        
        iz_tsw_tmp = torch.where(iz_tsw < 0, zero_tensor, iz_tsw.float())
        iz_tsw_tmp = torch.where(iz_tsw > ID - 1, id_tensor, iz_tsw_tmp.float())
        
        ix_tse_tmp = torch.where(ix_tse < 0, zero_tensor, ix_tse.float())
        ix_tse_tmp = torch.where(ix_tse > IW - 1, iw_tensor, ix_tse_tmp.float())
        
        iy_tse_tmp = torch.where(iy_tse < 0, zero_tensor, iy_tse.float())
        iy_tse_tmp = torch.where(iy_tse > IH - 1, ih_tensor, iy_tse_tmp.float())
        
        iz_tse_tmp = torch.where(iz_tse < 0, zero_tensor, iz_tse.float())
        iz_tse_tmp = torch.where(iz_tse > ID - 1, id_tensor, iz_tse_tmp.float())
        
        ix_bnw_tmp = torch.where(ix_bnw < 0, zero_tensor, ix_bnw.float())
        ix_bnw_tmp = torch.where(ix_bnw > IW - 1, iw_tensor, ix_bnw_tmp.float())
        
        iy_bnw_tmp = torch.where(iy_bnw < 0, zero_tensor, iy_bnw.float())
        iy_bnw_tmp = torch.where(iy_bnw > IH - 1, ih_tensor, iy_bnw_tmp.float())
        
        iz_bnw_tmp = torch.where(iz_bnw < 0, zero_tensor, iz_bnw.float())
        iz_bnw_tmp = torch.where(iz_bnw > ID - 1, id_tensor, iz_bnw_tmp.float())
        
        ix_bne_tmp = torch.where(ix_bne < 0, zero_tensor, ix_bne.float())
        ix_bne_tmp = torch.where(ix_bne > IW - 1, iw_tensor, ix_bne_tmp.float())
        
        iy_bne_tmp = torch.where(iy_bne < 0, zero_tensor, iy_bne.float())
        iy_bne_tmp = torch.where(iy_bne > IH - 1, ih_tensor, iy_bne_tmp.float())
        
        iz_bne_tmp = torch.where(iz_bne < 0, zero_tensor, iz_bne.float())
        iz_bne_tmp = torch.where(iz_bne > ID - 1, id_tensor, iz_bne_tmp.float())
        
        ix_bsw_tmp = torch.where(ix_bsw < 0, zero_tensor, ix_bsw.float())
        ix_bsw_tmp = torch.where(ix_bsw > IW - 1, iw_tensor, ix_bsw_tmp.float())
        
        iy_bsw_tmp = torch.where(iy_bsw < 0, zero_tensor, iy_bsw.float())
        iy_bsw_tmp = torch.where(iy_bsw > IH - 1, ih_tensor, iy_bsw_tmp.float())
        
        iz_bsw_tmp = torch.where(iz_bsw < 0, zero_tensor, iz_bsw.float())
        iz_bsw_tmp = torch.where(iz_bsw > ID - 1, id_tensor, iz_bsw_tmp.float())
        
        ix_bse_tmp = torch.where(ix_bse < 0, zero_tensor, ix_bse.float())
        ix_bse_tmp = torch.where(ix_bse > IW - 1, iw_tensor, ix_bse_tmp.float())
        
        iy_bse_tmp = torch.where(iy_bse < 0, zero_tensor, iy_bse.float())
        iy_bse_tmp = torch.where(iy_bse > IH - 1, ih_tensor, iy_bse_tmp.float())
        
        iz_bse_tmp = torch.where(iz_bse < 0, zero_tensor, iz_bse.float())
        iz_bse_tmp = torch.where(iz_bse > ID - 1, id_tensor, iz_bse_tmp.float())

    print("iz bse {}".format(iz_bse.shape))
    #input shape  22x 4x16x64x64
    #iz_bse shape 22x16x64x64
    #bse shape    22x16x64x64
    input = input.view(N,C,ID * IW * IH)
    tnw_val = torch.gather(input, 2, (iz_tnw_tmp * IW * IH + iy_tnw_tmp * IW + ix_tnw_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(input, 2, (iz_tne_tmp * IW * IH + iy_tne_tmp * IW + ix_tne_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(input, 2, (iz_tsw_tmp * IW * IH + iy_tsw_tmp * IW + ix_tsw_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(input, 2, (iz_tse_tmp * IW * IH + iy_tse_tmp * IW + ix_tse_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(input, 2, (iz_bnw_tmp * IW * IH + iy_bnw_tmp * IW + ix_bnw_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(input, 2, (iz_bne_tmp * IW * IH + iy_bne_tmp * IW + ix_bne_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(input, 2, (iz_bsw_tmp * IW * IH + iy_bsw_tmp * IW + ix_bsw_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(input, 2, (iz_bse_tmp * IW * IH + iy_bse_tmp * IW + ix_bse_tmp).long().view(N, 1, D * H * W).repeat(1, C, 1))
    
    tnw_val = bonds_check(tnw_val.view(N,C,D,H,W), iz_tnw, iy_tnw,ix_tnw, 0, 0, 0, ID, IH,IW)
    tne_val = bonds_check(tne_val.view(N,C,D,H,W), iz_tne, iy_tne,ix_tne, 0, 0, 0, ID, IH,IW)
    tsw_val = bonds_check(tsw_val.view(N,C,D,H,W), iz_tsw, iy_tsw,ix_tsw, 0, 0, 0, ID, IH,IW)
    tse_val = bonds_check(tse_val.view(N,C,D,H,W), iz_tse, iy_tse,ix_tse, 0, 0, 0, ID, IH,IW)
    bnw_val = bonds_check(bnw_val.view(N,C,D,H,W), iz_bnw, iy_bnw,ix_bnw, 0, 0, 0, ID, IH,IW)
    bne_val = bonds_check(bne_val.view(N,C,D,H,W), iz_bne, iy_bne,ix_bne, 0, 0, 0, ID, IH,IW)
    bsw_val = bonds_check(bsw_val.view(N,C,D,H,W), iz_bsw, iy_bsw,ix_bsw, 0, 0, 0, ID, IH,IW)
    bse_val = bonds_check(bse_val.view(N,C,D,H,W), iz_bse, iy_bse,ix_bse, 0, 0, 0, ID, IH,IW)
    
    out_val = (tnw_val.view(N,C,D,H,W) * tnw.view(N,1,D,H,W) +
               tne_val.view(N,C,D,H,W) * tne.view(N,1,D,H,W) +
               tsw_val.view(N,C,D,H,W) * tsw.view(N,1,D,H,W) +
               tse_val.view(N,C,D,H,W) * tse.view(N,1,D,H,W) +
               bnw_val.view(N,C,D,H,W) * bnw.view(N,1,D,H,W) +
               bne_val.view(N,C,D,H,W) * bne.view(N,1,D,H,W) +
               bsw_val.view(N,C,D,H,W) * bsw.view(N,1,D,H,W) +
               bse_val.view(N,C,D,H,W) * bse.view(N,1,D,H,W))
    return out_val

class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress, estimate_occlusion_map=True):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)  # ~60+G

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)  # 65G! NOTE: computation cost is large
        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)  # 0.8G
        self.norm = nn.BatchNorm3d(compress, affine=True)
        self.num_kp = num_kp
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None
        self.ac = nn.ReLU()

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 4, 16, 64, 64)
        identity_grid = make_coordinate_grid((d, h, w), ref=kp_source)  # (16, 64, 64, 3)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)  # (1, 1, d=16, h=64, w=64, 3)
        coordinate_grid = identity_grid - kp_driving.view(bs, self.num_kp, 1, 1, 1, 3)

        k = coordinate_grid.shape[1]

        # NOTE: there lacks an one-order flow
        driving_to_source = coordinate_grid + kp_source.view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        # adding background feature
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)  # (bs, 1+num_kp, d, h, w, 3)
        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        #import pdb
        #pdb.set_trace()
        sparse_deformed = grid_sample_3d(feature_repeat, sparse_motions, align_corners=False)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)

        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        spatial_size = feature.shape[3:]  # (d=16, h=64, w=64)
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w)
        heatmap = gaussian_driving - gaussian_source  # (bs, num_kp, d, h, w)

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.dtype).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)
        heatmap = heatmap.unsqueeze(2)         # (bs, 1+num_kp, 1, d, h, w)
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 32, 16, 64, 64)

        feature = self.compress(feature)  # (bs, 4, 16, 64, 64)
        feature = self.norm(feature)  # (bs, 4, 16, 64, 64)
        feature = self.ac(feature)  # (bs, 4, 16, 64, 64)

        out_dict = dict()

        # 1. deform 3d feature
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)  # (bs, 1+num_kp, d, h, w, 3)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)  # (bs, 1+num_kp, c=4, d=16, h=64, w=64)
        out_dict['deformed_feature'] = deformed_feature

        # 2. (bs, 1+num_kp, d, h, w)
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)  # (bs, 1+num_kp, 1, d, h, w)

        input = torch.cat([heatmap, deformed_feature], dim=2)  # (bs, 1+num_kp, c=5, d=16, h=64, w=64)
        input = input.view(bs, -1, d, h, w)  # (bs, (1+num_kp)*c=105, d=16, h=64, w=64)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)  # (bs, 1+num_kp, d=16, h=64, w=64)
        out_dict['mask'] = mask
        mask = mask.unsqueeze(2)                                   # (bs, num_kp+1, 1, d, h, w)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs, num_kp+1, 3, d, h, w)
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)  mask take effect in this place
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs, d, h, w, 3)

        out_dict['deformation'] = deformation

        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = prediction.shape
            prediction_reshape = prediction.view(bs, -1, h, w)
            occlusion_map = torch.sigmoid(self.occlusion(prediction_reshape))  # Bx1x64x64
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
