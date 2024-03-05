# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""General-purpose training script for image-to-image translation.
You need to specify the dataset ('--data_path'),
and model ('--G_A_ckpt', '--G_B_ckpt', '--D_A_ckpt', '--D_B_ckpt').
Example:
    Train a resnet model:
        python train.py --data_path ./data/horse2zebra --G_A_ckpt ./G_A.ckpt
"""
import kornia
import mindspore
import mindspore as ms
import numpy as np

from mindspore import nn
from mindspore.profiler.profiling import Profiler
from tqdm import tqdm

from ITMODNet_evaluater import ITMODNet_Evaluater
from config.Base_option import Base_options
from models.ITMODNet import ITMODNet_Net
from dataset.P3M_data import create_dataset
from utils.util import get_yaml_data, set_yaml_to_args
import mindspore.ops as mop
from mindspore import context

ms.set_seed(1)


def criterion_nig(u, la, alpha, beta, y, step=None, totalStep=None):
    # our loss function
    om = 2 * beta * (1 + la)
    # log_phi = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)
    loss = 0.5 * mop.log(np.pi / la) - alpha * mop.log(om) + (alpha + 0.5) * mop.log(
        la * (u - y) ** 2 + om)  # + log_phi  #
    # weight = torch.zeros_like(u)
    # weight[torch.abs(u-y) > 0.1] = 1
    lossr = 0.01 * mop.abs(u - y) * (2 * la + alpha)
    # lossr = kl_divergence_nig(u, y, alpha, beta, la)
    loss = loss + lossr
    return loss


def compute_loss(image, gt_matte, trimap, pred_semantic, pred_detail, pred_matte, pred_la, pred_alpha, pred_beta, epoch,
                 args, semantic_scale, detail_scale, matte_scale):
    boundaries = mop.logical_or((trimap < 0.5), (trimap > 0.5))
    # gt_semantic = gt_matte.squeeze(1).clone()
    # gt_semantic[(gt_semantic > 0) * (gt_semantic < 1)] = 255
    # gt_semantic = gt_semantic.long()
    gt_semantic = mop.interpolate(gt_matte, scale_factor=1 / 16, recompute_scale_factor=True, mode='bilinear')
    # gt_semantic = kornia.filters.gaussian_blur2d(gt_semantic, (3, 3), (0.8, 0.8))
    semantic_loss = mop.mse_loss(gt_semantic, pred_semantic)
    semantic_loss = semantic_scale * semantic_loss.mean()

    # calculate the detail loss
    trimap = trimap.type(pred_detail.dtype)
    gt_matte = gt_matte.type(pred_detail.dtype)
    pred_boundary_detail = mop.where(boundaries, trimap, pred_detail)
    gt_detail = mop.where(boundaries, trimap, gt_matte)
    detail_loss = mop.l1_loss(pred_boundary_detail, gt_detail)
    detail_loss = detail_scale * detail_loss

    # calculate the matte loss
    pred_boundary_matte = mop.where(boundaries, trimap, pred_matte)
    matte_l1_loss = 4.0 * mop.l1_loss(pred_boundary_matte, gt_detail)  # + F.l1_loss(pred_matte, gt_matte)
    matte_compositional_loss = 4.0 * mop.l1_loss(image * pred_boundary_matte,
                                                 image * gt_detail)  # + F.l1_loss(image * pred_matte, image * gt_matte)

    matte_loss = matte_l1_loss + matte_compositional_loss
    matte_loss = matte_scale * matte_loss
    # matte_loss = 0

    matte_loss_nig = criterion_nig(pred_matte, pred_la, pred_alpha, pred_beta, gt_matte, step=epoch,
                                   totalStep=args.epoch)
    matte_loss_nig = matte_loss_nig.mean()

    return semantic_loss, detail_loss, matte_loss, matte_loss_nig


def forward_fn(merge_img, merge_alpha, merge_trimap, user_map, epoch):
    input = mop.cat([merge_img, user_map], axis=1)  # , user_map
    pred_semantic, pred_detail, pred_la, pred_alpha, pred_beta, pred_matte = net(input, False)
    # calculate the final loss, backward the loss, and update the model
    semantic_loss, detail_loss, matte_loss, matte_loss_nig = compute_loss(merge_img, merge_alpha, merge_trimap,
                                                                          pred_semantic,
                                                                          pred_detail,
                                                                          pred_matte, pred_la, pred_alpha,
                                                                          pred_beta, epoch, args,
                                                                          10.0, 10.0,
                                                                          1.0)
    loss = semantic_loss + detail_loss + matte_loss + matte_loss_nig
    return loss, [pred_semantic, pred_detail, pred_la, pred_alpha, pred_beta, pred_matte]


def train_step(merge_img, merge_alpha, merge_trimap, user_map, epoch):
    (loss, _), grads = grad_fn(merge_img, merge_alpha, merge_trimap, user_map, epoch)
    optimizer(grads)
    return loss


base_option = Base_options()
args = base_option.get_args()
yamls_dict = get_yaml_data('./config/' + args.model + '_config.yaml')
set_yaml_to_args(args, yamls_dict)
net = ITMODNet_Net(args)
optimizer = nn.Adam(net.trainable_params(), learning_rate=args.lr, beta1=0.5, beta2=0.999)

grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)


def train():
    """Train function."""
    dataset = create_dataset(args)
    val_dataset = create_dataset(args, 'val')
    # TODO
    # scheduler = nn.cosine_decay_lr(min_lr, max_lr, total_step, step_per_epoch, decay_epoch)
    max_val_loss = 100000
    step = 0
    for epoch in range(args.epoch):
        loop = tqdm(enumerate(dataset.create_dict_iterator()), total=len(dataset))
        loss_epoch = 0
        net.set_train()
        for (i, label_data) in loop:
            merge_img = label_data['ori']
            merge_alpha = label_data['mask']
            merge_trimap = label_data['trimap']
            user_map = label_data['prior']

            loss = train_step(merge_img, merge_alpha, merge_trimap, user_map, epoch)

            loop.set_description(args.model + '|epoch:{}'.format(epoch))
            loop.set_postfix_str('loss: {}'.format(loss.asnumpy()))

        if (epoch + 1) % args.val_per_epoch == 0 and epoch >= 0:
            net.set_train(False)
            error_mse_sum = 0
            val_loop = tqdm(enumerate(val_dataset.create_dict_iterator()), total=len(val_dataset))
            val_loop.set_description('val|')
            for (i, label_data) in val_loop:
                label_img = label_data['merge_img']
                label_alpha = label_data['merge_gt']  # .unsqueeze(1)
                trimap = label_data['trimap']
                instance_map = label_data['prior']
                eval_out = ITMODNet_Evaluater(net,
                                              label_img,
                                              label_alpha,
                                              trimap,
                                              fusion=args.fusion,
                                              interac=args.inter_num)
                error_mse_sum += eval_out[0]
            mindspore.save_checkpoint(net, "./checkpoint/model_{}.ckpt".format(error_mse_sum / len(val_dataset)))
            print('MSE: {}' .format(error_mse_sum / len(val_dataset)))


if __name__ == "__main__":
    context.set_context(device_target="GPU")
    train()
