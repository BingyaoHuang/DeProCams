"""
Projector compensation demo (as shown in our video). After training, compensation images will be saved to `[data_root]/prj/cmp/`
"""

# %% Set environment
import os

# set which GPU(s) to use
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device_ids = [0]

from time import localtime, strftime
from trainNetwork import *
import Models
from scipy.spatial import cKDTree  # for point cloud alignment error

printConfig()

# set PyTorch device to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# did not see significantly differences when set to True, but True is significantly slower.
reproducible = False

# reproducibility
torch.backends.cudnn.deterministic = True if reproducible else False
torch.backends.cudnn.benchmark = False if reproducible else True

dataset_root = fullfile(os.getcwd(), '../../data')

# New dataset and is not in the 23 relighting and shape reconstruction benchmark
data_list = [
    'setups/camo_cmp',
]

checkDataList(dataset_root, data_list)

# flags that decides whether to generate and save the relit and compensated images
save_relighting = 1  # save relit images
save_compensation = 1  # enable it for projector compensation, and put desired effects to cam/desire/test. See CompenNet++ paper/code for more details
plot_on = 1  # plot training progress, final point cloud and normal map in visdom ([server]:8098), enable it for debugging

# Training configurations of ReconNet reported in the paper
num_train_list = [500]
# num_train_list = [50, 100, 250, 500]

# You can create your own models in Models.py and put their names in this list for comparisons.
model_list = ['DeProCams']
# model_list = ['DeProCams', 'No_rough', 'No_mask', 'No_const'] # ablation study on degraded versions

train_option_default = {'data_name': '',  # will be set later
                        'model_name': '',
                        'num_train': '',
                        'max_iters': 1000,  # reduce/increase it for faster/slower training with some performance drop/increase
                        'batch_size': 24,  # reduced it for faster training and less GPU memory usage, but performance may change
                        'lr': [1e-2, 1e-3],  # learning rate for DepthToAttribute and ShadingNet
                        'lr_drop_ratio': [0.2, 0.2],  # lr drop ratio for DepthToAttribute and ShadingNet
                        'lr_drop_rate': [[50, 800], [800, 1000]],  # lr milestones for DepthToAttribute and ShadingNet
                        'loss': 'l1+ssim',  # loss function
                        'l2_reg': [0, 1e-4],  # l2 regularization
                        'device': device,
                        'plot_on': plot_on,  # plot training progress using visdom, set to true for debugging
                        'train_plot_rate': 50,  # training and visdom plot rate
                        'valid_rate': 200}  # validation and visdom plot rate

# batch size must <= num_train
train_option_default['batch_size'] = min(train_option_default['batch_size'], num_train_list[0])

# log file
log_dir = '../../log'
if not os.path.exists(log_dir): os.makedirs(log_dir)
log_file_name = strftime('%Y-%m-%d_%H_%M_%S', localtime()) + '.txt'
log_file = open(fullfile(log_dir, log_file_name), 'w')
title_str = '{:30s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}\n'
log_file.write(title_str.format('data_name', 'model_name', 'loss_function', 'num_train', 'batch_size', 'max_iters', 'masked_psnr', 'masked_rmse',
                                'masked_ssim', 'whole_psnr', 'whole_rmse', 'whole_ssim', 'd_err'))
log_file.close()

# resize the input images if input_size is not None
cam_size = (240, 320)  # can set to a smaller size to save memory and speed up computation when debugging

# % evaluate all setups
for data_name in data_list:
    # load training and validation data
    data_root = fullfile(dataset_root, data_name)
    cam_surf, cam_train, cam_valid, prj_train, prj_valid, mask, mask_corners, calib_data = loadData(dataset_root, data_name, cam_size)
    prj_size = prj_train.shape[2:4]

    # surface image for training and validation
    cam_surf_train = cam_surf
    cam_surf_valid = cam_surf.expand(cam_valid.shape[0], -1, -1, -1)

    # convert all data to CUDA tensor if you have sufficient GPU memory (faster), otherwise comment them
    cam_train = cam_train.to(device)
    prj_train = prj_train.to(device)

    cam_valid = cam_valid.to(device)
    prj_valid = prj_valid.to(device)

    # validation data
    valid_data = dict(cam_surf=cam_surf_valid, cam_valid=cam_valid, prj_valid=prj_valid)

    # stats for different #Train
    for num_train in num_train_list:
        train_option = train_option_default.copy()
        train_option['num_train'] = num_train
        train_option['data_name'] = data_name.replace('/', '_')

        # select a subset to train
        train_data = dict(cam_surf=cam_surf_train[:num_train, :, :, :], cam_train=cam_train[:num_train, :, :, :],
                          prj_train=prj_train[:num_train, :, :, :], mask=mask)

        # stats for different models
        for model_name in model_list:
            log_file = open(fullfile(log_dir, log_file_name), 'a')

            # set seed of rng for repeatability
            resetRNGseed(0)

            # train option for current configuration
            train_option['model_name'] = model_name.replace('/', '_')

            # create a DeProCams model
            deprocams = Models.DeProCams(opt=model_name,
                                         depth_to_attribute=Models.DepthToAttribute(calib_data, opt=model_name),
                                         shading_net=Models.ShadingNet(opt=model_name))
            deprocams.depth_to_attribute.initDepth(mask_corners, device)
            if torch.cuda.device_count() >= 1: deprocams = nn.DataParallel(deprocams, device_ids=device_ids).to(device)

            print('-------------------------------------- Training Options -----------------------------------')
            print('\n'.join('{}: {}'.format(k, v) for k, v in train_option.items()))
            print('------------------------------------ Start training {:s} ---------------------------'.format(model_name))

            # train
            deprocams, _, _, _ = trainDeProCams(deprocams, train_data, valid_data, train_option)

            # done training and switch to eval mode to accomplish three tasks: relighting, compensation and shape reconstruction
            deprocams.eval()

            # eval
            valid_psnr, valid_rmse, valid_ssim, camj_valid_pred = evaluate(deprocams, valid_data)

            model_config = '{}_{}_{}_{}_{}'.format(train_option['model_name'], train_option['loss'], num_train, train_option['batch_size'],
                                                   train_option['max_iters'])
            # [1] relighting
            if save_relighting:
                print('------------------------------------ Saving relit results for {:s} ---------------------------'.format(model_name))

                for effect in ['test', 'frames']:
                    # create output relit image path.
                    relit_path = fullfile(data_root, 'pred/relit', effect, model_config)

                    if effect == 'test':
                        saveImgs(camj_valid_pred.detach().cpu(), relit_path)
                        print('Relit {} images saved to {}'.format(effect, relit_path))
                    else:
                        # put animated frames here to create fancy relit projection mapping effects, as shown in our video
                        prj_relit_path = fullfile(data_root, 'prj/relit', effect)
                        if os.path.isdir(prj_relit_path):
                            prj_relit = readImgsMT(prj_relit_path, prj_size)
                            cam_relit = torch.empty((len(prj_relit), 3, *cam_size))
                            cam_surf_device = cam_surf.to(device)
                            i = 0
                            with torch.no_grad():
                                for im in prj_relit:
                                    cam_relit[i], _, _, _, _ = deprocams(im[None].to(device), cam_surf_device)
                                    i += 1
                            saveImgs(cam_relit.detach().cpu(), relit_path)
                            print('Relit {} images saved to {}'.format(effect, relit_path))
                        else:
                            print('Projector relighting frames folder {:s} does not exist, skipping'.format(prj_relit_path))

            # [2] compensation
            if save_compensation:
                print('\n------------------------------------ Projector compensation using {:s} ---------------------------'.format(model_name))

                # desired test images are created such that they can fill the optimal displayable area, see CompenNet++ for details.
                for effect in ['test', 'frames']:
                    desire_effects_path = fullfile(data_root, 'cam/desire', effect)  # You can also compensate movie frames, as shown in our video.
                    if os.path.isdir(desire_effects_path):
                        print('Found desired camera-captured effects at {:s}, compensating....'.format(desire_effects_path))

                        # compensate using DeProCams and save compensated images
                        desire_effects = readImgsMT(desire_effects_path)
                        desired_mask = desire_effects.mean(0).mean(0) > 0

                        # min max image for clipping
                        im_min = readImg(fullfile(data_root, 'cam/raw/ref/img_0001.png')).to(device)[None]
                        im_max = readImg(fullfile(data_root, 'cam/raw/ref/img_0002.png')).to(device)[None]

                        compen_data = dict(cam_surf=cam_surf, cam_desire=desire_effects, prj_size=prj_size, mask=desired_mask, im_min=im_min, im_max=im_max)
                        prj_cmp_effects = compensate(deprocams, compen_data)  # generate projector compensation images

                        # create image save path
                        prj_cmp_path = fullfile(data_root, 'prj/cmp', effect, model_config)

                        # save compensated images
                        saveImgs(prj_cmp_effects, prj_cmp_path)  # compensated testing images, i.e., to be projected to the surface
                        print('Compensation images saved to ' + prj_cmp_path)
                    else:
                        print('Desired camera-captured effect image folder {:s} does not exist, skipping compensation'.format(desire_effects_path))

            # [3] shape reconstruction
            print('\n------------------------------------ Saving {:s} learned depth map ---------------------------'.format(model_name))
            # convert inverse depth map (1/d) to real depth (d) and scale it using T's scale
            d_pred = (calib_data['scale_T'] / deprocams.module.depth_to_attribute.cam_depth.detach().cpu())
            depth_pred_path = fullfile(data_root, 'pred/depth', model_config)
            saveDepth(d_pred, fullfile(depth_pred_path, 'depth_pred.txt'))
            print('Learned depth saved to {:s}'.format(depth_pred_path))

            # GT depth (manually cleaned SL point cloud converted depth map)
            d_gt = torch.Tensor(readDepth(fullfile(data_root, 'gt/depthGT.txt')))
            sl_mask = d_gt > 0

            # convert depth map to point cloud
            cam_KRT = deprocams.module.depth_to_attribute.cam.camera_matrix  # camera projection matrix (K*[R|t])
            pc_pred_tensor = kornia.geometry.depth_to_3d(d_pred[None, None, ...], cam_KRT)  # Bx3xHxW
            pc_pred = pc_pred_tensor.permute(0, 2, 3, 1).squeeze()  # Nx3, model predicted point cloud
            pc_gt = kornia.geometry.depth_to_3d(d_gt[None, None, ...], cam_KRT).permute(0, 2, 3, 1).squeeze()  # GT point cloud

            # compute point cloud alignment error (d_err)
            d_err, _ = cKDTree(pc_gt[sl_mask]).query(pc_pred[sl_mask], 1)

            # reconstructed normal map
            normal = deprocams.module.depth_to_attribute.pts3d2normal(pc_pred_tensor.to(device)).squeeze().cpu()  # actual normal
            normal_vis = normal.clone()  # for visualization
            normal_vis[0:2, ...] *= -1  # flip x and y
            normal_vis = (normal_vis * 0.5 + 0.5) * mask

            # save normal map
            normal_path = fullfile(data_root, 'pred/normal', model_config)
            if not os.path.exists(normal_path): os.makedirs(normal_path)
            cv.imwrite(fullfile(normal_path, 'normal_pred.png'), np.uint8(255 * normal_vis.permute(1, 2, 0))[:, :, ::-1])
            print('Normal map saved to ' + normal_path)

            if plot_on:  # plot point cloud and normal map in visdom webpage (slow, should only use it for debug)
                print('Plotting point cloud in visdom webpage (slow) ...')
                # ref/img_0004 is only for point cloud color
                pc_color = cv.imread(fullfile(data_root, 'cam/raw/ref/img_0004.png'))[:, :, ::-1]

                # plot point cloud (recommend installing the latest visdom from source to allow the option 'markerborderwidth=0')
                opts = dict(webgl=True, width=800, height=600, markersize=2, markerborderwidth=0, layoutopts={'plotly': {'paper_bgcolor': 'black'}})
                pc_pred_win = vis.scatter(pc_pred[mask].view(-1, 3), opts={**opts, 'markercolor': pc_color[mask].reshape(-1, 3), 'title': 'DeProCams'})
                pc_gt_win = vis.scatter(pc_gt[sl_mask].view(-1, 3), opts={**opts, 'markercolor': pc_color[sl_mask].reshape(-1, 3), 'title': 'Cleaned & interp SL (GT)'})

                # plot normal map
                normal_vis_win = vis.image(normal_vis)

            # save results to log file
            cam_valid_cpu = cam_valid.cpu()
            ret_str = '{:35s}{:<30}{:<20}{:<15}{:<15}{:<15}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}{:<15.4f}\n'
            log_file.write(
                ret_str.format(data_name, model_name, train_option['loss'], num_train, train_option['batch_size'], train_option['max_iters'],
                               psnr(camj_valid_pred * sl_mask, cam_valid_cpu * sl_mask),
                               rmse(camj_valid_pred * sl_mask, cam_valid_cpu * sl_mask),
                               ssim(camj_valid_pred * sl_mask, cam_valid_cpu * sl_mask),
                               valid_psnr, valid_rmse, valid_ssim, d_err.mean()))
            log_file.close()
            print('-------------------------------------- Done! ---------------------------\n')

print('All dataset done!')
