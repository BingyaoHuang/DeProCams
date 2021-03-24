'''
DeProCams training functions
'''

from utils import *
import ImgProc
import torch.nn.functional as F
import torch.optim as optim
import time
import kornia

l1_fun = nn.L1Loss()
l2_fun = nn.MSELoss()
ssim_fun = pytorch_ssim.SSIM().cuda()
smoothLoss = kornia.losses.InverseDepthSmoothnessLoss()


# %% load training, validation and calibration data for DeProCams
def loadData(dataset_root, data_name, input_size):
    # data paths
    data_root = fullfile(dataset_root, data_name)
    cam_ref_path = fullfile(data_root, 'cam/raw/ref')
    cam_train_path = fullfile(data_root, 'cam/raw/train')
    prj_train_path = fullfile(dataset_root, 'train')
    cam_valid_path = fullfile(data_root, 'cam/raw/test')
    prj_valid_path = fullfile(dataset_root, 'test')
    params_path = fullfile(data_root, 'params')
    print("Loading data from '{}'".format(data_root))

    # ref image with gray illumination
    cam_surf = readImgsMT(cam_ref_path, size=input_size, index=[2])  # ref/img_0003 is the cam-captured surface i.e., s when img_gray projected

    # training data
    cam_train = readImgsMT(cam_train_path, size=input_size)
    prj_train = readImgsMT(prj_train_path)

    # validation data
    cam_valid = readImgsMT(cam_valid_path, size=input_size)
    prj_valid = readImgsMT(prj_valid_path, index=list(range(cam_valid.shape[0])))

    # Use Nayar TOG'06 method to get direct light mask with two extra shifted checkerboard images.
    # Although the 23 benchmark setups did not use this method, we recommend it for new setups for better mask extraction.
    cam_cb_path = fullfile(data_root, 'cam/raw/cb')
    if os.path.exists(cam_cb_path):
        # find projector direct light mask
        im_cb = readImgsMT(cam_cb_path, size=input_size)
        im_cb = im_cb.numpy().transpose((2, 3, 1, 0))

        # find direct light mask using Nayar's TOG'06 method (also see Moreno 3DV'12)
        l1 = im_cb.max(axis=3)  # max image L+
        l2 = im_cb.min(axis=3)  # max image L-
        b = 0.9  # projector back light strength (for mask use a large b, for real direct/indirect separation, use a smaller b)
        im_direct = (l1 - l2) / (1 - b)  # direct light image
        # im_indirect = 2 * (l2 - b * l1) / (1 - b * b)  # indirect (global) light image

        im_direct = im_direct.clip(0, 1)
        im_mask, _, mask_corners = ImgProc.threshDeProCams(im_direct)  # use thresholded as mask
        im_mask = torch.Tensor(im_mask).bool()
    else:  # without using extra shifted checkerboard images
        # find projector FOV mask
        im_diff = readImgsMT(cam_ref_path, index=[2], size=input_size) - readImgsMT(cam_ref_path, index=[0], size=input_size)
        im_diff = im_diff.numpy().transpose((2, 3, 1, 0))

        im_mask, _, mask_corners = ImgProc.threshDeProCams(im_diff[..., 0], thresh=10)  # use thresholded surface image as mask
        im_mask = torch.Tensor(im_mask).bool()

    # load calibration data
    calib_data = loadCalib(fullfile(params_path, 'params.yml'))
    calib_data['cam_w'] = cam_train.shape[3]
    calib_data['cam_h'] = cam_train.shape[2]
    calib_data['prj_w'] = prj_train.shape[3]
    calib_data['prj_h'] = prj_train.shape[2]

    # resize projection matrix if resized
    _, _, org_h, org_w = readImgsMT(cam_ref_path, index=[0], size=None).shape
    scale_fx = input_size[1] / org_w
    scale_fy = input_size[0] / org_h
    calib_data['camK'][0, 0, 0] *= scale_fx
    calib_data['camK'][0, 0, 2] *= scale_fx
    calib_data['camK'][0, 1, 1] *= scale_fy
    calib_data['camK'][0, 1, 2] *= scale_fy

    # normalize T to a unit vector and save its scale
    calib_data['scale_T'] = torch.norm(calib_data['prjRT'][0, 0:3, 3])
    calib_data['prjRT'][0, 0:3, 3] /= calib_data['scale_T']

    return cam_surf, cam_train, cam_valid, prj_train, prj_valid, im_mask, mask_corners, calib_data


# %% same as trainReconNet, but formated code for adversarial
def trainDeProCams(model, train_data, valid_data, train_option):
    device = train_option['device']

    # empty cuda cache before training
    if device.type == 'cuda': torch.cuda.empty_cache()

    # training data
    cam_mask = train_data['mask'].to(device)  # thresholded camera-captured surface image, s*
    cam_surf_train = train_data['cam_surf'].to(device)  # s
    cam_train = train_data['cam_train']  # Ic
    prj_train = train_data['prj_train']  # Ip

    # only use one surf
    cam_surf_train_batch = cam_surf_train.expand(train_option['batch_size'], -1, -1, -1)

    # params, optimizers and lr schedulers
    depth_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ['module.depth_to_attribute.cam_depth'], model.named_parameters()))))
    other_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in ['module.depth_to_attribute.cam_depth'], model.named_parameters()))))

    # optimizer
    d_optimizer = optim.Adam([{'params': depth_params}], lr=train_option['lr'][0], weight_decay=train_option['l2_reg'][0])
    s_optimizer = optim.Adam([{'params': other_params}], lr=train_option['lr'][1], weight_decay=train_option['l2_reg'][1])

    # learning rate drop scheduler
    d_lr_scheduler = optim.lr_scheduler.MultiStepLR(d_optimizer, milestones=train_option['lr_drop_rate'][0], gamma=train_option['lr_drop_ratio'][0])
    s_lr_scheduler = optim.lr_scheduler.MultiStepLR(s_optimizer, milestones=train_option['lr_drop_rate'][1], gamma=train_option['lr_drop_ratio'][1])

    # %% start train
    start_time = time.time()

    # get model name
    if 'model_name' not in train_option: train_option['model_name'] = model.name if hasattr(model, 'name') else model.module.name

    # initialize visdom data visualization figure
    if 'plot_on' not in train_option: train_option['plot_on'] = True

    # title string of current training option
    title = optionToString(train_option)
    if train_option['plot_on']:
        # initialize visdom figures
        vis_train_fig = None
        vis_valid_fig = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]), name='origin',
                                 opts=dict(width=1300, height=500, markers=True, markersize=3,
                                           layoutopts=dict(plotly=dict(
                                               title={'text': title, 'font': {'size': 20}},
                                               font={'family': 'Arial', 'size': 20},
                                               hoverlabel={'font': {'size': 20}},
                                               xaxis={'title': 'Iteration'},
                                               yaxis={'title': 'Metrics', 'hoverformat': '.4f'}))))

    # main loop
    iters = 0

    while iters < train_option['max_iters']:
        model.train()  # explicitly set to train mode in case batchNormalization and dropout are used

        # randomly sample training batch and send to GPU
        idx = random.sample(range(train_option['num_train']), train_option['batch_size'])
        cam_train_batch = cam_train[idx, :, :, :].to(device) if cam_train.device.type != 'cuda' else cam_train[idx, :, :, :]
        prj_train_batch = prj_train[idx, :, :, :].to(device) if prj_train.device.type != 'cuda' else prj_train[idx, :, :, :]

        # predict and compute loss
        pred_train_batch, normal, grid, Ic_diff, pred_mask = model(prj_train_batch, cam_surf_train_batch)

        # image reconstruction loss (l1+ssim)
        train_loss_batch, train_l2_loss_batch = computeLoss(pred_train_batch, cam_train_batch, train_option['loss'])
        train_rmse_batch = math.sqrt(train_l2_loss_batch.item() * 3)  # 3 channel, rgb

        # other constraints/losses
        if 'No_mask' not in model.module.opt:
            train_loss_batch += l2_fun(pred_mask[0, 0, ...], cam_mask * 1.0)  # direct light mask consistency
        if 'No_rough' not in model.module.opt and 'No_const' not in model.module.opt:
            train_loss_batch += 0.5 * l2_fun(Ic_diff * pred_mask[0, 0, ...], cam_train_batch * pred_mask[0, 0, ...])  # rough shading consistency
        if 'No_depth_smooth' not in model.module.opt:
            train_loss_batch += 2 * smoothLoss(model.module.depth_to_attribute.cam_depth[None, None, ...], cam_surf_train)  # depth smoothness
        if 'No_grid_smooth' not in model.module.opt:
            train_loss_batch += smoothLoss(grid.permute(0, 3, 1, 2), cam_surf_train)  # grid (Omega) smoothness
        if 'No_normal_smooth' not in model.module.opt:
            train_loss_batch += 0.01 * smoothLoss(normal[0, None], cam_surf_train)  # normal map (n) smoothness

        # backpropagation and update params
        d_optimizer.zero_grad()
        s_optimizer.zero_grad()
        train_loss_batch.backward()
        d_optimizer.step()
        s_optimizer.step()

        # record running time
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))

        # plot train
        if train_option['plot_on']:
            if iters % train_option['train_plot_rate'] == 0 or iters == train_option['max_iters'] - 1:
                vis_train_fig = plotMontage(prj_train_batch, pred_train_batch, cam_train_batch, win=vis_train_fig, title='[Train]' + title)
                appendDataPoint(iters, train_loss_batch.item(), vis_curve_fig, 'train_loss')
                appendDataPoint(iters, train_rmse_batch, vis_curve_fig, 'train_rmse')

        # validation
        valid_psnr, valid_rmse, valid_ssim = 0., 0., 0.
        if valid_data is not None and (iters % train_option['valid_rate'] == 0 or iters == train_option['max_iters'] - 1):
            valid_psnr, valid_rmse, valid_ssim, pre_valid = evaluate(model, valid_data)

            # plot validation
            if train_option['plot_on']:
                vis_valid_fig = plotMontage(valid_data['prj_valid'], pre_valid, valid_data['cam_valid'], win=vis_valid_fig, title='[Valid]' + title)
                appendDataPoint(iters, valid_rmse, vis_curve_fig, 'valid_rmse')
                appendDataPoint(iters, valid_ssim, vis_curve_fig, 'valid_ssim')

        # print to console
        print('Iter:{:5d} | Time: {} | Train Loss: {:.4f} | Train RMSE: {:.4f} | Valid PSNR: {:7s}  | Valid RMSE: {:6s}  '
              '| Valid SSIM: {:6s}  | Learn Rate: {:.5f}/{:.5f} |'.format(iters, time_lapse, train_loss_batch.item(), train_rmse_batch,
                                                                          '{:>2.4f}'.format(valid_psnr) if valid_psnr else '',
                                                                          '{:.4f}'.format(valid_rmse) if valid_rmse else '',
                                                                          '{:.4f}'.format(valid_ssim) if valid_ssim else '',
                                                                          d_optimizer.param_groups[0]['lr'],
                                                                          s_optimizer.param_groups[0]['lr']))

        d_lr_scheduler.step()  # update learning rate according to schedule
        s_lr_scheduler.step()  # update learning rate according to schedule
        iters += 1

    # done training and save the model
    checkpoint_dir = '../../checkpoint'
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
    checkpoint_file_name = fullfile(checkpoint_dir, title + '.pth')
    torch.save(model.state_dict(), checkpoint_file_name)
    print('Checkpoint saved to {}\n'.format(checkpoint_file_name))

    return model, valid_psnr, valid_rmse, valid_ssim


# only update the pixels that are within the projector fov, used by register_hook
def mask_grad(x, mask):
    x[~mask] = 0
    return x


# %% using DeProCams to compensate images, optimize a batch at a time
def compensate(model, compen_data):
    cmp_option = {'data_name': 'Compensation',  # will be set later
                  'model_name': '',
                  'num_cmp': '',
                  'max_iters': 100,
                  'batch_size': 50,  # use a smaller batch size in case out of GPU memory
                  'lr': 2e-2,  # learning rate
                  'lr_drop_ratio': 0.2,
                  'lr_drop_rate': 60,  # adjust this according to max_iters (lr_drop_rate < max_iters)
                  'loss': 'l1+ssim',
                  'l2_reg': 0,  # l2 regularization
                  'plot_on': 0,  # plot training progress using visdom
                  'train_plot_rate': 10,  # training and visdom plot rate
                  'valid_rate': 200}  # validation and visdom plot rate

    cmp_option['batch_size'] = min(cmp_option['batch_size'], compen_data['cam_desire'].shape[0])
    cmp_option['num_cmp'] = compen_data['cam_desire'].shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_cmp = cmp_option['num_cmp']
    batch_size = cmp_option['batch_size']

    # compensation data
    prj_size = compen_data['prj_size']  # projector w and h
    cam_mask = compen_data['mask']  # fov mask of the desired cam-captured effects (square)
    prj_cmp = torch.empty((num_cmp, 3, *prj_size))  # output compensated projector images, 3 for RGB
    cam_desired = compen_data['cam_desire']

    # get cam->prj warping grid from DepthToAttribute prj->cam grid. This explicit warping step is not necessary but it significantly improves speed
    prj2cam_grid, _, _ = model.module.depth_to_attribute.warp_grid(1 / model.module.depth_to_attribute.cam_depth)
    cam2prj_grid = torch.Tensor(ImgProc.invertGrid(prj2cam_grid.detach().cpu(), prj_size))[None]  # warps desired cam-captured image to the prj space

    # warp prj image to the correct geometry, a better initialization
    # desired cam-captured image warped to the prj image space
    prj_cmp_init = F.grid_sample(cam_desired, cam2prj_grid.expand(cam_desired.shape[0], -1, -1, -1), align_corners=True)

    # desired cam-captured image fov warped to prj image space
    prj_mask = F.grid_sample(cam_mask.float()[None, None], cam2prj_grid, align_corners=True)
    prj_cmp_init = (prj_cmp_init * prj_mask).to(device)

    # only use one surf
    cam_surf_batch = compen_data['cam_surf'].to(device).expand_as(cam_desired)

    # no need to compute model weights gradient for compensation
    model.module.depth_to_attribute.cam_depth.requires_grad = False
    model.module.shading_net.requires_grad = False
    model.eval()

    # get model name
    if 'model_name' not in cmp_option: cmp_option['model_name'] = model.name if hasattr(model, 'name') else model.module.name

    # initialize visdom data visualization figure for debugging
    if 'plot_on' not in cmp_option: cmp_option['plot_on'] = True

    if cmp_option['plot_on']:
        # title string of current training option
        title = optionToString(cmp_option)

        # initialize visdom figures
        vis_train_fig = None
        vis_curve_fig = vis.line(X=np.array([0]), Y=np.array([0]), name='origin',
                                 opts=dict(width=800, height=500, markers=True, markersize=3,
                                           layoutopts=dict(
                                               plotly=dict(title={'text': title, 'font': {'size': 20}},
                                                           font={'family': 'Arial', 'size': 20},
                                                           hoverlabel={'font': {'size': 20}},
                                                           xaxis={'title': 'Iteration'},
                                                           yaxis={'title': 'Metrics', 'hoverformat': '.4f'}))))

    cam_mask_cuda = cam_mask.to(device)
    cam_desired_cuda = cam_desired.to(device)

    # start optimizing the projector compensation image
    sta_loc = 0
    start_time = time.time()

    for i in range(0, num_cmp // batch_size + 1):
        end_loc = min(num_cmp, sta_loc + batch_size)
        if sta_loc == end_loc:
            break
        # current batch ranges
        idx = range(sta_loc, end_loc)

        # a copy of prj_cmp_init used to be optimized
        prj_cmp_opt_batch = 0.5 * prj_cmp_init[idx].clone()  # prj cmp image, if too bright, this init should be lower
        prj_cmp_opt_batch.requires_grad = True

        # to speed up computation, only update those prj pixels that are within the camera desired fov (warped to prj image space)
        prj_cmp_opt_batch.register_hook(lambda x: mask_grad(x, prj_mask.squeeze().bool().expand_as(prj_cmp_opt_batch)))

        # optimizer and lr scheduler
        optimizer = optim.Adam([prj_cmp_opt_batch], lr=1e-1, weight_decay=0)  # no weight_decay (may get dimmer compensations)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 100], gamma=0.1)

        iters = 0
        while iters < cmp_option['max_iters']:
            prj_cmp_batch = torch.clamp(prj_cmp_opt_batch, min=0, max=1)  # pixel value should be in (0, 1)
            cam_cap_pred, normal, grid, Ic_diff, pred_mask = model(prj_cmp_batch, cam_surf_batch[idx])

            # use min max image to clip inferred relit image pixel values to physically plausible ranges.
            im_min_batch, im_max_batch = compen_data['im_min'].expand_as(cam_cap_pred), compen_data['im_max'].expand_as(cam_cap_pred)
            cam_cap_pred[cam_cap_pred < im_min_batch] = im_min_batch[cam_cap_pred < im_min_batch]
            cam_cap_pred[cam_cap_pred > im_max_batch] = im_max_batch[cam_cap_pred > im_max_batch]

            cmp_rmse_batch = math.sqrt(l2_fun(cam_cap_pred, cam_desired_cuda[idx]).item() * 3)  # RMSE, 3 channel, rgb

            # image reconstruction loss (l1 + ssim)
            cmp_loss_batch = 1 * l1_fun(cam_cap_pred * cam_mask_cuda, cam_desired_cuda[idx])  # similar to cam desired
            cmp_loss_batch += 1 * (1 - ssim_fun(cam_cap_pred * cam_mask_cuda, cam_desired_cuda[idx]))

            # smoothness losses
            cmp_loss_batch += 1 * smoothLoss(prj_cmp_batch, prj_cmp_init[idx], grad_weight=True)  # prj cmp smooth
            cmp_loss_batch += 1 * smoothLoss(Ic_diff, cam_desired_cuda[idx], grad_weight=True)  # I_diff smooth

            # saturation loss
            lower_sat_err = ((prj_cmp_opt_batch * (prj_cmp_opt_batch < 0)) ** 2).mean()
            upper_sat_err = (((prj_cmp_opt_batch * (prj_cmp_opt_batch > 1) - 1)) ** 2).mean()

            if lower_sat_err.item() == lower_sat_err.item():  # get rid of nan
                cmp_loss_batch += 10 * lower_sat_err
            if upper_sat_err.item() == upper_sat_err.item():  # get rid of nan
                cmp_loss_batch += 10 * upper_sat_err

            # backpropagation and update params
            optimizer.zero_grad()
            cmp_loss_batch.backward()
            optimizer.step()

            # plot compensation process
            if cmp_option['plot_on']:
                if iters % cmp_option['train_plot_rate'] == 0 or iters == cmp_option['max_iters'] - 1:
                    vis_train_fig = plotMontage(prj_cmp_batch.detach(), cam_cap_pred.detach(), cam_desired_cuda[idx].detach(),
                                                win=vis_train_fig, title='[Compensation]' + title)
                    appendDataPoint(iters, cmp_loss_batch.item(), vis_curve_fig, 'train_loss')
                    appendDataPoint(iters, cmp_rmse_batch, vis_curve_fig, 'cmp_rmse')

            lr_scheduler.step()  # update learning rate according to schedule
            iters += 1

        # print to console
        time_lapse = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        print('Img idx:{:5d} - {:5d} | Time: {} | Loss: {:.4f} | RMSE: {:.4f} | '.format(sta_loc, end_loc, time_lapse, cmp_loss_batch.item(),
                                                                                         cmp_rmse_batch))

        prj_cmp[idx] = prj_cmp_batch.detach().cpu() * prj_mask
        sta_loc = end_loc

    return prj_cmp


# %% local functions

# compute loss between prediction and ground truth
def computeLoss(prj_pred, prj_train, loss_option):
    train_loss = 0

    # l1
    if 'l1' in loss_option:
        l1_loss = l1_fun(prj_pred, prj_train)
        train_loss += l1_loss

    # l2
    l2_loss = l2_fun(prj_pred, prj_train)
    if 'l2' in loss_option:
        train_loss += l2_loss

    # ssim
    if 'ssim' in loss_option:
        ssim_loss = 1 * (1 - ssim_fun(prj_pred, prj_train))
        train_loss += ssim_loss

    return train_loss, l2_loss


# append a data point to the curve in win
def appendDataPoint(x, y, win, name, env=None):
    vis.line(
        X=np.array([x]),
        Y=np.array([y]),
        env=env,
        win=win,
        update='append',
        name=name,
        opts=dict(markers=True, markersize=3)
    )


# plot sample predicted images using visdom, more than three rows
def plotMontage(*argv, index=None, win=None, title=None, env=None):
    with torch.no_grad():  # just in case
        # compute montage grid size
        if argv[0].shape[0] > 5:
            grid_w = 5
            idx = random.sample(range(argv[0].shape[0]), grid_w) if index is None else index
        else:
            grid_w = argv[0].shape[0]
            # idx = random.sample(range(cam_im.shape[0]), grid_w)
            idx = range(grid_w)

        # resize to (256, 256) for better display
        tile_size = (256, 256)
        im_resize = torch.empty((len(argv) * grid_w, argv[0].shape[1]) + tile_size)
        i = 0
        for im in argv:
            if im.shape[2] != tile_size[0] or im.shape[3] != tile_size[1]:
                im_resize[i:i + grid_w] = F.interpolate(im[idx, :, :, :], tile_size)
            else:
                im_resize[i:i + grid_w] = im[idx, :, :, :]
            i += grid_w

        # title
        plot_opts = dict(title=title, caption=title, font=dict(size=18), width=1300, store_history=False)

        im_montage = torchvision.utils.make_grid(im_resize, nrow=grid_w, padding=10, pad_value=1)
        win = vis.image(im_montage, win=win, opts=plot_opts, env=env)

    return win


# evaluate model on validation dataset
def evaluate(model, valid_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cam_surf = valid_data['cam_surf']
    cam_valid = valid_data['cam_valid']
    prj_valid = valid_data['prj_valid']

    with torch.no_grad():
        model.eval()  # explicitly set to eval mode

        # if you have limited GPU memory, we need to predict in batch mode
        if cam_surf.device.type != device.type:
            last_loc = 0
            valid_mse, valid_ssim = 0., 0.

            pred_valid = torch.zeros(cam_valid.shape)
            num_valid = cam_valid.shape[0]
            batch_size = 50 if num_valid > 50 else num_valid

            for i in range(0, num_valid // batch_size):
                idx = range(last_loc, last_loc + batch_size)
                cam_surf_batch = cam_surf[idx, :, :, :].to(device) if cam_surf.device.type != 'cuda' else cam_surf[idx, :, :, :]
                cam_valid_batch = cam_valid[idx, :, :, :].to(device) if cam_valid.device.type != 'cuda' else cam_valid[idx, :, :, :]
                prj_valid_batch = prj_valid[idx, :, :, :].to(device) if prj_valid.device.type != 'cuda' else prj_valid[idx, :, :, :]

                # predict batch
                pred_valid_batch = model(prj_valid_batch, cam_surf_batch)
                if type(pred_valid_batch) == tuple and len(pred_valid_batch) > 1: pred_valid_batch = pred_valid_batch[0].detach()
                pred_valid[last_loc:last_loc + batch_size, :, :, :] = pred_valid_batch.cpu()

                # compute loss
                valid_mse += l2_fun(pred_valid_batch, cam_valid_batch).item() * batch_size
                valid_ssim += ssim(pred_valid_batch, cam_valid_batch) * batch_size

                last_loc += batch_size
            # average
            valid_mse /= num_valid
            valid_ssim /= num_valid

            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
        else:
            # if all data can be loaded to GPU memory
            # prj_valid_pred = predict(model, dict(cam=cam_valid, cam_surf=cam_surf)).detach()
            pred_valid = model(prj_valid, cam_surf)
            if type(pred_valid) == tuple and len(pred_valid) > 1: pred_valid = pred_valid[0].detach()
            valid_mse = l2_fun(pred_valid, cam_valid).item()
            valid_rmse = math.sqrt(valid_mse * 3)  # 3 channel image
            valid_psnr = 10 * math.log10(1 / valid_mse)
            valid_ssim = ssim_fun(pred_valid, cam_valid).item()

    return valid_psnr, valid_rmse, valid_ssim, pred_valid
