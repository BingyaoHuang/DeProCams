import numpy as np
import ImgProc
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from utils import softTrue


# DeProCams ShadingNet
class ShadingNet(nn.Module):
    def __init__(self, opt=None):
        super(ShadingNet, self).__init__()
        self.name = self.__class__.__name__
        self.relu = nn.ReLU()
        # self.leakyRelu = nn.LeakyReLU(0.1)

        self.s_chan = 3 if 'No_rough' in opt else 9

        # backbone branch
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)

        # surface image feature extraction branch
        self.conv1_s = nn.Conv2d(self.s_chan, 32, 3, 2, 1)
        self.conv2_s = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3_s = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv4_s = nn.Conv2d(128, 256, 3, 1, 1)

        # transposed conv
        self.transConv1 = nn.ConvTranspose2d(128, 64, 2, 2, 0)
        self.transConv2 = nn.ConvTranspose2d(64, 32, 2, 2, 0)
        self.conv6 = nn.Conv2d(32, 3, 3, 1, 1)

        # skip layers
        self.skipConv1 = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu,
            nn.Conv2d(3, 3, 3, 1, 1),
            self.relu
        )

        self.skipConv2 = nn.Conv2d(32, 32, 1, 1, 0)
        self.skipConv3 = nn.Conv2d(64, 64, 1, 1, 0)
        # self.skipConv4 = Interpolate((240, 320), 'bilinear')

        # stores biases of surface feature branch (net simplification)
        self.register_buffer('res1_s', None)
        self.register_buffer('res2_s', None)
        self.register_buffer('res3_s', None)
        self.register_buffer('res4_s', None)

        # initialization function, first checks the module type,
        def _initialize_weights(m):
            if type(m) == nn.Conv2d:
                nn.init.kaiming_normal_(m.weight)

        self.apply(_initialize_weights)

    # simplify trained model by trimming surface branch to biases
    def simplify(self, s, n):
        self.res1_s = self.relu(self.conv1_s(s))
        self.res2_s = self.relu(self.conv2_s(self.res1_s))
        self.res3_s = self.relu(self.conv3_s(self.res2_s))
        self.res4_s = self.relu(self.conv4_s(self.res3_s))

        self.res1_s = self.res1_s.squeeze()
        self.res2_s = self.res2_s.squeeze()
        self.res3_s = self.res3_s.squeeze()
        self.res4_s = self.res4_s.squeeze()

    # x is the input uncompensated image, s is a 1x3x256x256 surface image
    def forward(self, x, *argv):
        s = torch.cat(argv, 1)

        # surface feature extraction
        res1_s = self.relu(self.conv1_s(s)) if self.res1_s is None else self.res1_s
        res2_s = self.relu(self.conv2_s(res1_s)) if self.res2_s is None else self.res2_s
        res3_s = self.relu(self.conv3_s(res2_s)) if self.res3_s is None else self.res3_s
        res4_s = self.relu(self.conv4_s(res3_s)) if self.res4_s is None else self.res4_s

        # backbone
        res1 = self.relu(self.conv1(x) + res1_s)
        res2 = torch.cat((res1, self.skipConv2(res1_s)), 1)
        x = self.relu(self.conv2(res1) + res2_s)
        res3 = torch.cat((x, self.skipConv3(res2_s)), 1)
        x = self.relu(self.conv3(x) + res3_s)
        x = self.relu(self.conv4(x) + res4_s)

        x = self.relu(self.conv5(x) + res3)
        x = self.relu(self.transConv1(x) + res2)
        x = self.relu(self.transConv2(x))
        x = torch.clamp(self.relu(self.conv6(x) + self.skipConv1(argv[0])), max=1)

        return x


# DeProCams DepthToAttribute
class DepthToAttribute(nn.Module):
    def __init__(self, calib_data, opt):
        super(DepthToAttribute, self).__init__()
        self.name = self.__class__.__name__
        self.calib_data = calib_data
        self.opt = opt

        # cam and prj image sizes
        self.cam_h, self.cam_w = calib_data['cam_h'], calib_data['cam_w']
        self.prj_h, self.prj_w = calib_data['prj_h'], calib_data['prj_w']

        # create Konia pinhole camera (src) and projector (dst) using calibration data
        self.cam = kornia.geometry.PinholeCamera(calib_data['camK'], calib_data['camRT'], torch.Tensor([self.cam_h]), torch.Tensor([self.cam_w]))
        self.prj = kornia.geometry.PinholeCamera(calib_data['prjK'], calib_data['prjRT'], torch.Tensor([self.prj_h]), torch.Tensor([self.prj_w]))

        # compute stereo rectification grid
        self.computeRectGrid()

        # uniform grid containing cam pixel coords
        cam_pts2d = F.pad(kornia.utils.create_meshgrid(self.cam_h, self.cam_w, normalized_coordinates=False), [0, 1], "constant", 1.0)
        cam_pts3d_normalized = kornia.geometry.transform_points(self.cam.intrinsics_inverse()[:, None], cam_pts2d)
        self.register_buffer('cam_pts3d_normalized', cam_pts3d_normalized)

        # near and far plane of camera. Note that this near and far are for normalized d using scale_T
        self.far = 10
        self.near = 0.1

        # initialize learnable cam_depth
        r1, r2 = 1 / self.far, 1 / self.far + (1 / self.near - 1 / self.far) * 0.1
        self.cam_depth = nn.Parameter((r1 - r2) * torch.rand(self.cam_h, self.cam_w) + r2)  # initialize inverse depth

        # precompute some parameters to speedup training. Register as buffers for faster computation
        # self.register_buffer('cam_invK', self.cam.intrinsics_inverse())
        self.register_buffer('cam_KRT', self.cam.camera_matrix)
        self.register_buffer('prj_KRT', torch.matmul(self.prj.intrinsics, torch.matmul(self.prj.extrinsics, self.cam.extrinsics.inverse())))
        self.register_buffer('prj_org', self.prj.extrinsics.inverse()[0, :-1, -1])
        self.register_buffer('prj_size', torch.Tensor([self.prj_h, self.prj_w]))

        # a white image for projector fov computation
        self.register_buffer('cam_im_white', torch.ones(1, 1, self.cam_h, self.cam_w))
        self.register_buffer('prj_im_white', torch.ones(1, 1, self.prj_h, self.prj_w))

        # for occlusion mask warp_grid
        self.register_buffer('prj_K', self.prj.intrinsics)
        self.register_buffer('prj_RT', self.prj.extrinsics)

        # sobel gradient
        self.register_buffer('sobel_kernel', kornia.filters.kernels.normalize_kernel2d(kornia.filters.get_spatial_gradient_kernel2d('sobel', 1)).repeat(3, 1, 1, 1, 1).flip(-3))

    # compute stereo rectification sampling grid, then normalize
    def computeRectGrid(self):
        cam_rect_grid, cam_unrect_grid = ImgProc.getRectifyGrid(self.calib_data)

        # OpenCV should use (h, w), while Torch/Kornia use (w, h)
        self.register_buffer('cam_rect_grid', cam_rect_grid * 2 / (torch.Tensor([self.cam_w, self.cam_h]) - 1) - 1)
        self.register_buffer('cam_unrect_grid', cam_unrect_grid * 2 / (torch.Tensor([self.cam_w, self.cam_h]) - 1) - 1)

    # initialize inverse depth map 1/d with triangulated affine mappings
    def initDepth(self, mask_corners, device):
        src_pts = np.array([[-1, -1], [1, -1], [1, 1]]).astype(np.float32)
        dst_pts = np.array(mask_corners[0:3]).astype(np.float32)
        affine_mat = torch.Tensor(cv.getAffineTransform(dst_pts, src_pts))  # prj -> cam
        prj_grid = F.affine_grid(affine_mat.view(-1, 2, 3), torch.Size([1, 3, self.cam_h, self.cam_w])).permute(0, 3, 1, 2)
        prj_grid[0, 0, ...] = 0.5 * (prj_grid[0, 0, ...] + 1) * (self.prj_w - 1)
        prj_grid[0, 1, ...] = 0.5 * (prj_grid[0, 1, ...] + 1) * (self.prj_h - 1)
        cam_grid = kornia.utils.create_meshgrid(self.cam_h, self.cam_w, False).permute(0, 3, 1, 2)

        cam_pts2d = cam_grid.contiguous().view(2, -1).numpy()
        prj_pts2d = prj_grid.contiguous().view(2, -1).numpy()

        # triangulate
        cam_KRT = torch.matmul(self.calib_data['camK'], self.calib_data['camRT']).squeeze().numpy()[0:3, 0:4]
        prj_KRT = torch.matmul(self.calib_data['prjK'], self.calib_data['prjRT']).squeeze().numpy()[0:3, 0:4]
        pts3d = cv.triangulatePoints(cam_KRT, prj_KRT, cam_pts2d, prj_pts2d)
        pts3d = pts3d[0:3, :] / pts3d[3, :]

        # initialize cam_depth (note that here cam_depth is inverse depth, i.e., 1/d)
        self.cam_depth.data = 1 / torch.Tensor(pts3d[2, ...]).view(self.cam_h, self.cam_w).clone().to(device)

    # using sobel kernel to convert pts3d to normal (from kornia.depth_to_normals, but faster if sobel kernel is registered as buffer
    def pts3d2normal(self, cam_pts3d):
        d_grad = F.conv3d(F.pad(F.pad(cam_pts3d, [1, 1, 1, 1], 'replicate')[:, :, None], [0, 0, 0, 0, 1, 1], 'constant', 0), self.sobel_kernel, padding=0, groups=3)
        dzdx, dzdy = d_grad[:, :, 0], d_grad[:, :, 1]
        n = F.normalize(torch.cross(dzdx, dzdy, dim=1), dim=1, p=2)
        return n

    # derived from kornia.depth_warper.warp_grid, but it also returns the 3d coords in dst view space
    def warp_grid(self, d):
        cam_pts3d = self.cam_pts3d_normalized * d[None, ..., None]

        # 3d and 2d points in prj space
        prj_pts3d = kornia.geometry.src2dst(cam_pts3d, self.prj_RT)
        prj_pts2d = kornia.geometry.cam2pixelNoTrans(prj_pts3d, self.prj_K)

        # normalize prj2cam_grid to (-1, 1)
        prj2cam_grid = prj_pts2d * 2 / self.prj_size - 1
        return prj2cam_grid, cam_pts3d, prj_pts3d

    # compute an occlusion mask (differentiable)
    def occlusionMask(self, xy, z):
        xy = (xy[0, None, ...] + 1) * (self.prj_size - 1) / 2  # denormalize
        xyz = torch.cat((xy.permute(0, 3, 1, 2), z[None, None]), 1)
        xyz = F.grid_sample(xyz, self.cam_rect_grid, align_corners=True)
        b, c, h, w = xyz.shape

        # hack to perform lexsort on x and y (TODO: replace this with torch.lexsort once torch implements it)
        x_scale = 1e4  # here we assume x coord scale < 10000, which is true for most ProCams setup (image res < 10000x10000)
        y_scale = 1e1

        xyz_sum = x_scale * torch.round(xyz[:, 0]) + y_scale * torch.round(xyz[:, 1]) + xyz[:, 2]
        idx = torch.argsort(xyz_sum, dim=2, descending=False)

        row_idx = torch.linspace(0, h - 1, h).long().view(-1, 1).repeat(1, w).view(-1)
        col_idx = idx.view(-1)
        xyz_sorted = xyz[0, :, row_idx, col_idx].view(3, h, w)
        diff_sorted = F.pad(xyz_sorted[..., :, 1:] - xyz_sorted[..., :-1], [1, 0], 'constant', 0)
        diff = torch.zeros_like(xyz)
        diff[..., row_idx, col_idx] = diff_sorted.view(1, 3, -1)

        # not occluded
        t_xy = 1  # x, y coord difference threshold for occlusion
        t_z = 1e-3  # z coord difference threshold for occlusion

        if self.prj.tx > 0:  # for vertical ProCams, check self.prj.ty instead
            # camera on the right
            mask_rectified = (1 - softTrue(t_xy - torch.abs(diff[:, 0:2]).sum(1)) * softTrue(diff[:, 2] - t_z))[0, ...]
        else:
            mask_rectified = (1 - softTrue(t_xy - torch.abs(diff[:, 0:2]).sum(1)) * softTrue(t_z - diff[:, 2]))[0, ...]

        # 1's theshold
        mask_thresh = 0.5
        mask = F.grid_sample(mask_rectified.unsqueeze(0).unsqueeze(0), self.cam_unrect_grid, align_corners=True)
        mask = softTrue(mask - mask_thresh)

        return mask

    def forward(self, Ip, s):
        # get projector to camera image warping grid (\Omega) using camera depth map (d)
        d = 1 / torch.clamp(self.cam_depth, min=1 / self.far, max=1 / self.near)  # clamp inverse depth to far~near, then convert to true depth map
        prj_pts2d, cam_pts3d, prj_pts3d = self.warp_grid(d)

        # visualize warping grid using pseudo color
        # fs(cv.applyColorMap(np.uint8((0.5+0.5*prj_pts2d[0,:,:,0].detach()).cpu().numpy()*255), cv.COLORMAP_JET)) # x
        # fs(cv.applyColorMap(np.uint8((0.5+0.5*prj_pts2d[0,:,:,1].detach()).cpu().numpy()*255), cv.COLORMAP_JET)) # y

        # warp projector input image x to the canonical camera frontal view
        Ip = F.grid_sample(Ip, prj_pts2d.expand(Ip.shape[0], -1, -1, -1))

        # compute projector direct light mask
        occ_mask = self.occlusionMask(prj_pts2d, prj_pts3d[0, ..., 2])  # must be in projector's view space

        if 'No_mask' in self.opt:
            dl_mask = self.cam_im_white
        else:
            dl_mask = F.grid_sample(self.prj_im_white, prj_pts2d, align_corners=True) * occ_mask

        # mask Ip' using prj direct light mask (M)
        Ip = Ip * dl_mask

        # compute normals
        cam_pts3d = cam_pts3d.permute(0, 3, 1, 2)  # Bx3xHxW
        n = self.pts3d2normal(cam_pts3d)

        # the normal points inward the surface, negate it as l and v. However, to visualize normal, negate Z again, because in OpenCV
        # coord system, Z points inward, thus all surface normals that point outward will be < 0.
        n *= -1

        # each surface point's incident light ray direction in cam view space. Note the ray dir is from pts3d to prj_org
        l = F.normalize(self.prj_org[None, :, None, None] - cam_pts3d, dim=1, p=2)
        l *= dl_mask  # light dir should be consistent with prj direct light mask (M)

        # view direction is the ray from pts3d to camera optical center (0,0,0)
        v = F.normalize(-cam_pts3d, dim=1, p=2)

        # expand to batch
        n = n.expand(s.shape[0], -1, -1, -1)
        l = l.expand(s.shape[0], -1, -1, -1)
        v = v.expand(s.shape[0], -1, -1, -1)

        if 'No_rough' in self.opt:  # no rough diffuse or specular shadings
            Ic_diff, Ic_spec = None, None
        else:
            # rough diffuse
            n_dot_l = (n * l).sum(dim=1)[:, None, ...]
            Ic_diff = torch.clamp(n_dot_l * Ip * s, max=1, min=0)  # Ic_diff = Kd*Ld*(l*n)

            # rough specular, ks is set to a gray-scale version of s for rough specular-like shading, alpha=1
            h = F.normalize(l + v, dim=1, p=2)  # half vector
            r_dot_v = (h * n).sum(dim=1)[:, None, ...]  # Blinn-Phong specular
            Ic_spec = torch.clamp(r_dot_v * Ip * s.mean(1)[:, None], max=1, min=0)  # Ic_spec = Ks*Ls*(r*v)^(alpha)

        return Ip, Ic_diff, Ic_spec, n, prj_pts2d, dl_mask


class DeProCams(nn.Module):
    def __init__(self, depth_to_attribute, shading_net, opt=None):
        super(DeProCams, self).__init__()
        self.name = self.__class__.__name__
        self.opt = opt

        # initialize from existing models
        self.depth_to_attribute = depth_to_attribute
        self.shading_net = shading_net

    # s is Bx3x256x256 surface image
    def forward(self, Ip, s):

        Ip, Ic_diff, Ic_spec, n, prj_pts2d, dl_mask = self.depth_to_attribute(Ip, s)

        if 'No_rough' in self.opt:  # no rough diffuse or specular shadings
            Ic = self.shading_net(Ip, s)
        else:
            Ic = self.shading_net(Ip, s, Ic_diff, Ic_spec)

        return Ic, n[0, None], prj_pts2d, Ic_diff, dl_mask