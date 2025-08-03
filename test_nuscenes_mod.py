import os
import random
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import shutil
from easydict import EasyDict
from tqdm import tqdm

import yaml
from utils.load_util import load_yaml
from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_nuScenes_label_name, get_pc_model_class
from dataloader.dataset2 import get_dataset_class, get_collate_class
from network.largekernel_model import get_model_class

from utils.load_save_util import load_checkpoint_old, load_checkpoint_model_mask
from utils.erk_sparse_core import Masking, CosineDecay

import warnings
warnings.filterwarnings("ignore")

import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors

#########

# Visual
def create_bbox(points):
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
    aabb.color = (1, 1, 1)  # white bounding box
    return aabb

def get_label_name(label_id, label_map):
    return label_map.get(label_id, f"class_{label_id}")

#########

# Testing settings
parser = argparse.ArgumentParser(description='LSKNet Testing')
parser.add_argument('--config_path', default='./config/lk-nuscenes_erk_tta.yaml')
parser.add_argument('--ip', default='127.0.0.1', type=str)
parser.add_argument('--port', default='3023', type=str)
parser.add_argument('--num_vote', type=int, default=28, help='number of voting in the test') #14 28
args = parser.parse_args()
config_path = args.config_path
configs = load_yaml(config_path)
configs.update(vars(args))  # override the configuration using the value in args
configs = EasyDict(configs)

configs['dataset_params']['val_data_loader']["batch_size"] = configs.num_vote
configs['dataset_params']['val_data_loader']["num_workers"] = 2 #configs.num_vote 14
if configs.num_vote > 1:
    configs['dataset_params']['val_data_loader']["rotate_aug"] = True
    configs['dataset_params']['val_data_loader']["flip_aug"] = True
    configs['dataset_params']['val_data_loader']["scale_aug"] = True
    configs['dataset_params']['val_data_loader']["transform_aug"] = True
elif configs.num_vote == 1:
    configs['dataset_params']['val_data_loader']["rotate_aug"] = False
    configs['dataset_params']['val_data_loader']["flip_aug"] = False
    configs['dataset_params']['val_data_loader']["scale_aug"] = False
    configs['dataset_params']['val_data_loader']["transform_aug"] = False

exp_dir_root = configs['model_params']['model_load_path'].split('/')
exp_dir_root = exp_dir_root[0] if len(exp_dir_root) > 1 else ''
exp_dir = './'+ exp_dir_root +'/'
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)
shutil.copy('test_nuscenes.py', str(exp_dir))
shutil.copy('config/lk-nuscenes_erk_tta.yaml', str(exp_dir))


def main(configs):
    configs.nprocs = torch.cuda.device_count()
    configs.train_params.distributed = True if configs.nprocs > 1 else False
    if configs.train_params.distributed:
        mp.spawn(main_worker, nprocs=configs.nprocs, args=(configs.nprocs, configs))
    else:
        main_worker(0, 1, configs)

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def main_worker(local_rank, nprocs, configs):
    torch.autograd.set_detect_anomaly(True)

    dataset_config = configs['dataset_params']
    model_config = configs['model_params']
    train_hypers = configs['train_params']
    train_hypers.local_rank = local_rank
    train_hypers.world_size = nprocs
    configs.train_params.world_size = nprocs
    
    if train_hypers['distributed']:
        init_method = 'tcp://' + args.ip + ':' + args.port
        dist.init_process_group(backend='nccl', init_method=init_method, world_size=nprocs, rank=local_rank)
        dataset_config.val_data_loader.batch_size = dataset_config.val_data_loader.batch_size // nprocs

    pytorch_device = torch.device('cuda:' + str(local_rank))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)

    seed = train_hypers.seed + local_rank * dataset_config.val_data_loader.num_workers * train_hypers['max_num_epochs']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    SemKITTI_label_name = get_nuScenes_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = get_model_class(model_config['model_architecture'])(configs)

    if train_hypers['distributed']:
        my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)

    if os.path.exists(model_config['model_load_path']):
        print('pre-train')
        try:
            my_model, pre_weight = load_checkpoint_model_mask(model_config['model_load_path'], my_model, pytorch_device)
        except:
            my_model = load_checkpoint_old(model_config['model_load_path'], my_model)

    my_model.to(pytorch_device)

    if train_hypers['distributed']:
        train_hypers.local_rank = train_hypers.local_rank % torch.cuda.device_count()
        my_model= DistributedDataParallel(my_model,device_ids=[train_hypers.local_rank],find_unused_parameters=False)


    # prepare dataset
    val_dataloader_config = dataset_config['val_data_loader']
    data_path = val_dataloader_config["data_path"]
    val_imageset = val_dataloader_config["imageset"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    val_pt_dataset = SemKITTI(dataset_config, data_path, imageset=val_imageset, num_vote = configs.num_vote)

    val_dataset = get_dataset_class(dataset_config['dataset_type'])(
        val_pt_dataset,
        config=dataset_config,
        loader_config=val_dataloader_config,
        num_vote = configs.num_vote)
    
    val_pt = val_pt_dataset
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=train_hypers.world_size, rank=train_hypers.local_rank, shuffle=False)
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=val_dataloader_config["batch_size"],
                                                    collate_fn=get_collate_class(dataset_config['collate_type']),
                                                    num_workers=val_dataloader_config["num_workers"],
                                                    pin_memory=True,
                                                    drop_last=False,
                                                    shuffle = False,
                                                    sampler=val_sampler)


    if val_imageset == 'val':
        if train_hypers.local_rank == 0:
            # validation
            print('*'*80)
            print('Test network performance on validation split')
            print('*'*80)
            pbar = tqdm(total=len(val_dataset_loader), ncols=80)
        else:
            pbar = None

        my_model.eval()
        hist_list = []
        time_list = []
        visualization_results = []
        with torch.no_grad():
            for i_iter_val, (val_data_dict) in enumerate(val_dataset_loader):
                print(i_iter_val)
                if i_iter_val > 20:
                    break  # Only process the first sample

                torch.cuda.empty_cache()
                raw_labels = val_data_dict['raw_labels'].to(pytorch_device)
                vote_logits = torch.zeros(raw_labels.shape[0], model_config['num_classes']).to(pytorch_device)
                indices = val_data_dict['indices'].to(pytorch_device)

                val_data_dict['points'] = val_data_dict['points'].to(pytorch_device)
                val_data_dict['normal'] = val_data_dict['normal'].to(pytorch_device)
                val_data_dict['batch_idx'] = val_data_dict['batch_idx'].to(pytorch_device)
                val_data_dict['labels'] = val_data_dict['labels'].to(pytorch_device)

                torch.cuda.synchronize()
                start_time = time.time()
                val_data_dict = my_model(val_data_dict)
                torch.cuda.synchronize()
                time_list.append(time.time()-start_time)
                logits = val_data_dict['logits']
                vote_logits.index_add_(0, indices, logits)

                if train_hypers['distributed']:
                    torch.distributed.barrier()
                    vote_logits = reduce_tensor(vote_logits, nprocs)

                ####################################################################
                # Radar
                # --- POST‐INFERENCE RADAR FUSION ---
                # 1) softmax → probs
                lidar_probs = F.softmax(vote_logits, dim=1).cpu().numpy()
                coords      = val_data_dict['points'][:, :3].cpu().numpy()

                # 2) get this sample’s nuScenes tokens
                lidar_token = val_pt.token_list[i_iter_val]['lidar_token']
                sd_data     = val_pt.nusc.get('sample_data', lidar_token)
                sample      = val_pt.nusc.get('sample', sd_data['sample_token'])

                # 3) loop through all radar sensors
                radar_keys = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']
                for rk in radar_keys:
                    if rk not in sample['data']:
                        continue
                    rtok      = sample['data'][rk]
                    rinfo     = val_pt.nusc.get('sample_data', rtok)
                    rpath     = os.path.join(data_path, rinfo['filename'])

                    # — load radar points robustly —
                    if rpath.endswith('.pcd'):
                        pcd       = o3d.io.read_point_cloud(rpath)
                        radar_pts = np.asarray(pcd.points)
                    else:
                        rpc       = np.fromfile(rpath, dtype=np.float32).reshape(-1, 18)
                        radar_pts = rpc[:, :3]

                    # find neighbors & boost
                    nbrs      = NearestNeighbors(n_neighbors=5, radius=1.0).fit(coords)
                    dists, idxs = nbrs.kneighbors(radar_pts)

                    targets   = [2,3,4,5,6,7,9,10]
                    touched = 0
                    for lids, dv in zip(idxs, dists):
                        for lid, dist in zip(lids, dv):
                            if dist > 1.0: 
                                continue
                            touched += 1
                            for c in targets:
                                lidar_probs[lid][c] *= 1.5 * np.exp(-dist)
                            lidar_probs[lid] /= lidar_probs[lid].sum()

                    print(f"Radar fusion touched {touched} LiDAR points in this frame")

                # 4) final labels from boosted probs
                predict_labels = np.argmax(lidar_probs, axis=1)

                ####################################################################

                # 1) extract your M model‑input points
                coords = val_data_dict['points'][:, :3].cpu().numpy()

                # 2) align labels by indices
                indices_np = val_data_dict['indices'].cpu().numpy()
                labels_aligned = predict_labels[indices_np]

                # 3) store for later
                visualization_results.append({
                    'points': coords,           # shape (M,3)
                    'labels': labels_aligned,   # shape (M,)
                })

                raw_labels_np = raw_labels.cpu().numpy().squeeze()
                hist = fast_hist_crop(predict_labels, raw_labels_np, unique_label)
                hist_list.append(hist)

        if train_hypers.local_rank == 0:
            print("Visualizing predictions...")
            
            def get_color_map():
                import matplotlib.pyplot as plt
                color_map = plt.get_cmap("tab20", 20)  # 20-class color map
                return {i: (np.array(color_map(i)[:3]) * 255).astype(int) for i in range(20)}

            color_map = get_color_map()

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="LSK3DNet Scene Viewer", width=960, height=720)
            render_option = vis.get_render_option()
            render_option.background_color = np.array([0, 0, 0])  # optional black background
            render_option.point_size = 1.0

            for i, scene in enumerate(visualization_results):
                vis.clear_geometries()

                points = scene['points']
                labels = scene['labels']
                unique_classes = np.unique(labels)
                colors = np.array([color_map[l] for l in labels]) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                vis.add_geometry(pcd)

                # ----- Bounding Box Clustering per Semantic Class -----
                unique_classes = np.unique(labels)

                # Filter class IDs of interest (based on 16-class mapping):
                # 4 = car, 6 = motorcycle, 7 = pedestrian, 3 = bus, 10 = truck, 5 = construction_vehicle, 9 = trailer, 2 = bicycle
                target_classes = [2, 3, 4, 5, 6, 7, 9, 10]
                print("Class Legend:")
                for cls_id in unique_classes:
                    if cls_id not in target_classes:
                        continue
                    label_name = SemKITTI_label_name.get(cls_id + 1, f"class_{cls_id+1}")
                    color = color_map[cls_id]
                    print(f"Class {cls_id}: {label_name} - Color RGB: {color}")

                for cls in unique_classes:
                    if cls not in target_classes:
                        continue
                    mask = labels == cls
                    class_points = points[mask]
                    if len(class_points) < 30:
                        continue
                    clustering = DBSCAN(eps=1.0, min_samples=10).fit(class_points)
                    cluster_labels = clustering.labels_
                    for cluster_id in np.unique(cluster_labels):
                        if cluster_id == -1:
                            continue
                        cluster_mask = cluster_labels == cluster_id
                        cluster_pts = class_points[cluster_mask]
                        if len(cluster_pts) < 20:
                            continue
                        bbox = create_bbox(cluster_pts)
                        bbox.color = np.array(color_map[cls]) / 255.0  # Color bbox by class
                        vis.add_geometry(bbox)

                ctr = vis.get_view_control()
                ctr.set_zoom(0.25)

                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.5)

            vis.destroy_window()


            iou = per_class_iu(sum(hist_list))
            print('Validation per class iou: ')
            for class_name, class_iou in zip(unique_label_str,iou):
                print('%s : %.2f%%' % (class_name, class_iou*100))
            val_miou = np.nanmean(iou) * 100
            pbar.close()
            print('Current val miou is %.3f ' % val_miou)
            print('Inference time per %d is %.4f seconds\n' %
                (dataset_config.val_data_loader.batch_size, np.mean(time_list)))

if __name__ == '__main__':
    print(' '.join(sys.argv))
    print(configs)
    main(configs)