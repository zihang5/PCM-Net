import glob
import os, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
#from EMCADnew import EMCADPR
#from EMCADnewlittle import EMCADPR
#from Pivit import pivit
import models
import random
import csv

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

same_seeds(24)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def main():
    TIME = []
    MEM_USAGE = AverageMeter()

    val_dir = './data/Mind/test/'
    weights = [1, 1]  # loss weights
    lr = 0.0001
    head_dim = 6
    num_heads = [8, 4, 2, 1, 1]
    model_folder = 'PR/'
    model_idx = -1
    model_dir = 'Mindexperiments/' + model_folder
    
    csv_file = 'MindPR.csv'
    headers = [
        "left caudal anterior cingulate",
        "left caudal middle frontal",
        "left cuneus",
        "left entorhinal",
        "left fusiform",
        "left inferior parietal",
        "left inferior temporal",
        "left isthmus cingulate",
        "left lateral occipital",
        "left lateral orbitofrontal",
        "left lingual",
        "left medial orbitofrontal",
        "left middle temporal",
        "left parahippocampal",
        "left paracentral",
        "left pars opercularis",
        "left pars orbitalis",
        "left pars triangularis",
        "left pericalcarine",
        "left postcentral",
        "left posterior cingulate",
        "left precentral",
        "left precuneus",
        "left rostral anterior cingulate",
        "left rostral middle frontal",
        "left superior frontal",
        "left superior parietal",
        "left superior temporal",
        "left supramarginal",
        "left transverse temporal",
        "left insula",
        "right caudal anterior cingulate",
        "right caudal middle frontal",
        "right cuneus",
        "right entorhinal",
        "right fusiform",
        "right inferior parietal",
        "right inferior temporal",
        "right isthmus cingulate",
        "right lateral occipital",
        "right lateral orbitofrontal",
        "right lingual",
        "right medial orbitofrontal",
        "right middle temporal",
        "right parahippocampal",
        "right paracentral",
        "right pars opercularis",
        "right pars orbitalis",
        "right pars triangularis",
        "right pericalcarine",
        "right postcentral",
        "right posterior cingulate",
        "right precentral",
        "right precuneus",
        "right rostral anterior cingulate",
        "right rostral middle frontal",
        "right superior frontal",
        "right superior parietal",
        "right superior temporal",
        "right supramarginal",
        "right transverse temporal",
        "right insula"]
    
    img_size = (160, 192, 160)
    #model = EMCADPR.srPR(1,3,8)
    #model = pivit.pivit(img_size)
    #model = models.Im2grid(img_size)
    model = models.PRNetplusplus(img_size)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total model parameters: {total_params}')

    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.LPBABrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    eval_hd95_def = AverageMeter()
    #eval_hd95_raw = AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for data in test_loader:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
    
                # Measure inference time
                start = time.time()
                x_def, flow = model(x, y)
                elapsed_time = time.time() - start
                TIME.append(elapsed_time)
    
                # Measure memory usage
                mem_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)  # in MB
                MEM_USAGE.update(mem_usage)
                torch.cuda.reset_peak_memory_stats()
    
                def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
                tar = y.detach().cpu().numpy()[0, 0, :, :, :]
                jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
    
                eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
                dscs, dsc_trans = utils.dice_val_VOI(def_out.long(), y_seg.long())
                writer.writerow(dscs.squeeze())
                _, dsc_raw = utils.dice_val_VOI(x_seg.long(), y_seg.long())

                hd95_trans = utils.hd95_val(def_out, y_seg)
                #hd95_raw = utils.hd95_val(x_seg, y_seg)
                
                print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
                eval_dsc_def.update(dsc_trans.item(), x.size(0))
                eval_dsc_raw.update(dsc_raw.item(), x.size(0))
                eval_hd95_def.update(hd95_trans.item(), x.size(0))
                #eval_hd95_raw.update(hd95_raw.item(), x.size(0))
                stdy_idx += 1

        avg_time = np.mean(TIME)
        std_time = np.std(TIME)
        print(f'Average inference time: {avg_time:.4f} seconds (std: {std_time:.4f})')
        print(f'Average memory usage: {MEM_USAGE.avg:.2f} MB (std: {MEM_USAGE.std:.2f})')

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('Deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        #print('Deformed ASSD: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_assd_def.avg,eval_assd_def.std,eval_assd_raw.avg,eval_assd_raw.std))
        print('Deformed HD95: {:.3f} +- {:.3f}'.format(eval_hd95_def.avg,eval_hd95_def.std))
        '''print('Deformed HD95: {:.3f} +- {:.3f}, Affine hd95: {:.3f} +- {:.3f}'.format(eval_hd95_def.avg,
                                                                                            eval_hd95_def.std,
                                                                                            eval_hd95_raw.avg,
                                                                                            eval_hd95_raw.std))'''
        

if __name__ == '__main__':
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
