# TODO: implement your version after imlplementing processing


import torch
from torch.optim import lr_scheduler
from os.path import join
from util.util import seg_accuracy, print_network
from . import Unet


class ClassifierModel:
    """Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        #
        self.nclasses = opt.nclasses

        # Init Network and run it on gpu to train
        conv_filters = [32, 64, 128, 256]
        down_convs = [opt.input_nc] + conv_filters
        up_convs = conv_filters + [opt.nclasses]
        # max edges , pool layers (different from paper)
        pool_res = [2280, 1800, 1350, 600]
        self.net = Unet.UNet(
            pool_res, down_convs, up_convs, blocks=3, transfer_data=True
        )
        if len(self.gpu_ids) > 0:
            assert torch.cuda.is_available()
            net.cuda(self.gpu_ids[0])
            net = net.cuda()
            net = torch.nn.DataParallel(net, self.gpu_ids)

        self.net.train(self.is_train)
        self.eval = torch.nn.CrossEntropyLoss(ignore_index=-1).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(
                self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)
            )

            def lambda_rule(epoch):
                lr = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(
                    opt.niter_decay + 1
                )
                return lr

            self.scheduler = lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=lambda_rule
            )
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data["edge_features"]).float()
        labels = torch.from_numpy(data["label"]).long()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(
            self.is_train
        )
        self.labels = labels.to(self.device)
        self.mesh = data["mesh"]
        if self.opt.dataset_mode == "segmentation" and not self.is_train:
            self.soft_label = torch.from_numpy(data["soft_label"])

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        self.loss = self.eval(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    ################## TODO: Implement it on your own after implementing data processing

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = "%s_net.pth" % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print("loading the model from %s" % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = "%s_net.pth" % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]["lr"]
        print("learning rate = %.7f" % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            pred_class = out.data.max(1)[1]
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation"""
        if self.opt.dataset_mode == "classification":
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == "segmentation":
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == "segmentation":
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
