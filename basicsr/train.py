import datetime
import logging
import math
import time
import torch
from os import path as osp

from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

from basicsr.archs import build_network
from torch.nn.parallel import DataParallel, DistributedDataParallel
from copy import deepcopy

def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True

    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    copy_opt_file(args.opt, opt['path']['experiments_root'])

    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    if 'distill' in opt:
        print ("Confirmation: This is a distillation task.")
        net_teacher = build_network(opt['distill']['network_teacher'])
        net_teacher = net_teacher.to(torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu'))
        if opt['dist']:
            find_unused_parameters = opt.get('find_unused_parameters', False)
            net_teacher = DistributedDataParallel(
                net_teacher, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif opt['num_gpu'] > 1:
            net_teacher = DataParallel(net_teacher)
        
        path_teacher = opt['distill'].get('pretrain_network_g', None)
        if path_teacher is not None:
            param_key = opt['distill'].get('param_key_g', 'params')
            if isinstance(net_teacher, (DataParallel, DistributedDataParallel)):
                net_teacher = net_teacher.module
            load_net = torch.load(path_teacher, map_location=lambda storage, loc: storage)
            if param_key is not None:
                if param_key not in load_net and 'params' in load_net:
                    param_key = 'params'
                    logger.info('Loading: params_ema does not exist, use params.')
                load_net = load_net[param_key]
            logger.info(f'Loading {net_teacher.__class__.__name__} teacher model from {path_teacher}, with param key: [{param_key}].')
            for k, v in deepcopy(load_net).items():
                if k.startswith('module.'):
                    load_net[k[7:]] = v
                    load_net.pop(k)
            net_teacher.load_state_dict(load_net, opt['distill'].get('strict_load_g', True))
        net_teacher.eval()
        model.validation(val_loaders[0], 0, tb_logger, opt['val']['save_img'])

    model = build_model(opt)
    if resume_state:
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    if "CAT" in opt['network_g']['type']:
        layer_name = 'get_v'
    elif "SwinIR" in opt['network_g']['type']:
        layer_name = 'patch_embed'
    elif "EDSR" in opt['network_g']['type']:
        layer_name = 'relu'
    else:
        raise ValueError("network error")

    if 'distill' in opt:
        features_in_hook_teacher = []
        def hook_teacher(module, fea_in, fea_out):
            features_in_hook_teacher.append(fea_in)
            return None

        print ("teacher: ",net_teacher)
        
        for (name, module) in net_teacher.named_modules():
            if layer_name in name:
                module.register_forward_hook(hook=hook_teacher)

        features_in_hook = []

        def hook(module, fea_in, fea_out):
            features_in_hook.append(fea_in)
            return None
        
        for (name, module) in model.net_g.named_modules():
            if layer_name in name:
                module.register_forward_hook(hook=hook)

    if len(val_loaders) > 1:
        logger.warning('Multiple validation datasets are *only* supported by SRModel.')
    for val_loader in val_loaders:
        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

    vals, cnt_skip_blocks, new_iters = [], 0, 0

    cnt_skip_blocks = sum([layer.skip_block.data.item() == 1 for layer in model.net_g.module.layers])
    need_skip_blocks = len(model.net_g.module.layers) // 2

    st_ratio, ed_ratio = 0.2, 1.0
    skip_blocks = []
    for epoch in range(start_epoch, total_epochs + 1):
        cnt_skip_blocks = sum([layer.skip_block.data.item() == 1 for layer in model.net_g.module.layers])
    
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        features_in_hook, features_in_hook_teacher = [], []

        while train_data is not None:
            data_timer.record()

            # QSA (first stage)
            if current_iter == int(total_iters * st_ratio) and cnt_skip_blocks < need_skip_blocks:
                if hasattr(model.net_g, "body"):
                    max_block_id, max_learnable_shortcut = -1, 0.0
                    for i, qnblock in enumerate(model.net_g.body):
                        if float(qnblock.learnable_shortcut) > max_learnable_shortcut and qnblock.learnable_shortcut.requires_grad:
                            max_block_id = i
                            max_learnable_shortcut = float(qnblock.learnable_shortcut)
                    if max_block_id != -1:
                        with torch.no_grad():
                            model.net_g.body[max_block_id].learnable_shortcut.data = torch.ones_like(model.net_g.body[max_block_id].learnable_shortcut).to(model.net_g.body[max_block_id].learnable_shortcut.device)
                            model.net_g.body[max_block_id].learnable_shortcut.requires_grad = False
                            model.net_g.body[max_block_id].skip_block.data = 1 - model.net_g.body[max_block_id].skip_block.data
                        vals.append(model.best_metric_results[val_loaders[0].dataset.opt['name']]['psnr']['val'])
                        logger.warning(f'current iter: {current_iter} (20%), updating learnable shortcut! block id: {max_block_id}, last psnr: {vals[-1]}, learnable_shortcut: {float(model.net_g.body[max_block_id].learnable_shortcut)}, skip_block: {float(model.net_g.body[max_block_id].skip_block)}')
                        new_iters = 0
                        skip_blocks.append(max_block_id)
                elif hasattr(model.net_g, "layers") or hasattr(model.net_g.module, "layers"):
                    if not hasattr(model.net_g, "layers"):
                        _model = model.net_g.module
                    max_block_id, max_learnable_shortcut = -1, 0.0
                    for i, qnblock in enumerate(_model.layers):
                        if float(qnblock.learnable_shortcut) > max_learnable_shortcut and qnblock.learnable_shortcut.requires_grad:
                            max_block_id = i
                            max_learnable_shortcut = float(qnblock.learnable_shortcut)
                    if max_block_id != -1:
                        with torch.no_grad():
                            _model.layers[max_block_id].learnable_shortcut.data = torch.ones_like(_model.layers[max_block_id].learnable_shortcut).to(_model.layers[max_block_id].learnable_shortcut.device)
                            _model.layers[max_block_id].learnable_shortcut.requires_grad = False
                            _model.layers[max_block_id].skip_block.data = 1 - _model.layers[max_block_id].skip_block.data
                        vals.append(model.best_metric_results[val_loaders[0].dataset.opt['name']]['psnr']['val'])
                        logger.warning(f'current iter: {current_iter} (20%), updating learnable shortcut! block id: {max_block_id}, last psnr: {vals[-1]}, learnable_shortcut: {float(_model.layers[max_block_id].learnable_shortcut)}, skip_block: {float(_model.layers[max_block_id].skip_block)}')
                        new_iters = 0
                        skip_blocks.append(max_block_id)

            current_iter += 1
            new_iters += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)

            if 'distill' not in opt:
                model.optimize_parameters(current_iter)
            else:
                # SFD
                features_in_hook, features_in_hook_teacher = [], []
                model.lq = model.lq.to(torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu'))
                teacher_output = net_teacher(model.lq)
                model.optimize_parameters_kd(current_iter, features_in_hook, features_in_hook_teacher, skip_blocks)
                features_in_hook, features_in_hook_teacher = [], []
                
            iter_timer.record()
            if current_iter == 1:
                msg_logger.reset_start_time()
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                if hasattr(model.net_g, "body"):
                    log_vars.update({f'block {block_id} alpha': float(model.net_g.body[block_id].learnable_shortcut) for block_id in range(len(model.net_g.body)) })
                elif hasattr(model.net_g, "layers"):
                    log_vars.update({f'block {block_id} alpha': float(model.net_g.layers[block_id].learnable_shortcut) for block_id in range(len(model.net_g.layers)) })
                elif hasattr(model.net_g.module, "layers"):
                    log_vars.update({f'block {block_id} alpha': float(model.net_g.module.layers[block_id].learnable_shortcut) for block_id in range(len(model.net_g.module.layers)) })
                msg_logger(log_vars)

            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

                # QSA (second stage)
                if hasattr(model.net_g, "body"):
                    if current_iter > int(total_iters * st_ratio)  and cnt_skip_blocks < need_skip_blocks and (model.all_metric_results[val_loaders[0].dataset.opt['name']][-1]['psnr'] >= vals[-1] * 1.0 or new_iters >= total_iters * ed_ratio / len(model.net_g.body)) * 2:
                        max_block_id, max_learnable_shortcut = -1, 0.0
                        for i, qnblock in enumerate(model.net_g.body):
                            if float(qnblock.learnable_shortcut) > max_learnable_shortcut and qnblock.learnable_shortcut.requires_grad:
                                max_block_id = i
                                max_learnable_shortcut = float(qnblock.learnable_shortcut)
                        
                        if max_block_id != -1:
                            with torch.no_grad():
                                model.net_g.body[max_block_id].learnable_shortcut.data = torch.ones_like(model.net_g.body[max_block_id].learnable_shortcut).to(model.net_g.body[max_block_id].learnable_shortcut.device)
                                model.net_g.body[max_block_id].learnable_shortcut.requires_grad = False
                                model.net_g.body[max_block_id].skip_block.data = 1 - model.net_g.body[max_block_id].skip_block.data
                            vals.append(model.best_metric_results[val_loaders[0].dataset.opt['name']]['psnr']['val'])
                            logger.warning(f'current iter: {current_iter} (20%), updating learnable shortcut! block id: {max_block_id}, last psnr: {vals[-1]}')
                            new_iters = 0
                            skip_blocks.append(max_block_id)
                elif hasattr(model.net_g, "layers") or hasattr(model.net_g.module, "layers"):
                    if not hasattr(model.net_g, "layers"):
                        _model = model.net_g.module
                    if current_iter > int(total_iters * st_ratio)  and cnt_skip_blocks < need_skip_blocks and (model.all_metric_results[val_loaders[0].dataset.opt['name']][-1]['psnr'] >= vals[-1] * 1.0 or new_iters >= total_iters * ed_ratio / len(_model.layers) * 2):
                        max_block_id, max_learnable_shortcut = -1, 0.0
                        for i, qnblock in enumerate(_model.layers):
                            if float(qnblock.learnable_shortcut) > max_learnable_shortcut and qnblock.learnable_shortcut.requires_grad:
                                max_block_id = i
                                max_learnable_shortcut = float(qnblock.learnable_shortcut)
                        
                        if max_block_id != -1:
                            with torch.no_grad():
                                _model.layers[max_block_id].learnable_shortcut.data = torch.ones_like(_model.layers[max_block_id].learnable_shortcut).to(_model.layers[max_block_id].learnable_shortcut.device)
                                _model.layers[max_block_id].learnable_shortcut.requires_grad = False
                                _model.layers[max_block_id].skip_block.data = 1 - _model.layers[max_block_id].skip_block.data
                            vals.append(model.best_metric_results[val_loaders[0].dataset.opt['name']]['psnr']['val'])
                            logger.warning(f'current iter: {current_iter} (20%), updating learnable shortcut! block id: {max_block_id}, last psnr: {vals[-1]}')
                            new_iters = 0
                            skip_blocks.append(max_block_id)

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
