import torch
import copy


class GANFactory:
    factories = {}

    def __init__(self):
        pass

    def add_factory(gan_id, model_factory):
        GANFactory.factories.put[gan_id] = model_factory

    add_factory = staticmethod(add_factory)

    # A Template Method:

    def create_model(gan_id, net_d=None, criterion=None):
        if gan_id not in GANFactory.factories:
            GANFactory.factories[gan_id] = \
                eval(gan_id + '.Factory()')
        return GANFactory.factories[gan_id].create(net_d, criterion)

    create_model = staticmethod(create_model)


class GANTrainer(object):
    def __init__(self, net_d, criterion):
        self.net_d = net_d
        self.criterion = criterion

    def loss_d(self, pred, gt):
        pass

    def loss_g(self, pred, gt):
        pass

    def get_params(self):
        pass


class NoGAN(GANTrainer):
    def __init__(self, net_d, criterion):
        GANTrainer.__init__(self, net_d, criterion)

    def loss_d(self, pred, gt):
        return [0]

    def loss_g(self, pred, gt):
        return 0

    def get_params(self):
        return [torch.nn.Parameter(torch.Tensor(1))]

    class Factory:
        @staticmethod
        def create(net_d, criterion): return NoGAN(net_d, criterion)


class SingleGAN(GANTrainer):
    def __init__(self, net_d, criterion):
        GANTrainer.__init__(self, net_d, criterion)
        self.net_d = self.net_d.cuda()

    def loss_d(self, pred, gt):
        return self.criterion(self.net_d, pred, gt)

    def loss_g(self, pred, gt):
        return self.criterion.get_g_loss(self.net_d, pred, gt)

    def get_params(self):
        return self.net_d.parameters()

    class Factory:
        @staticmethod
        def create(net_d, criterion): return SingleGAN(net_d, criterion)


class DoubleGAN(GANTrainer):
    def __init__(self, net_d, criterion):
        GANTrainer.__init__(self, net_d, criterion)
        self.patch_d = net_d['patch'].cuda()
        self.full_d = net_d['full'].cuda()
        self.full_criterion = copy.deepcopy(criterion)

    def loss_d(self, pred, gt):
        return (self.criterion(self.patch_d, pred, gt) + self.full_criterion(self.full_d, pred, gt)) / 2

    def loss_g(self, pred, gt):
        return (self.criterion.get_g_loss(self.patch_d, pred, gt) + self.full_criterion.get_g_loss(self.full_d, pred,
                                                                                                  gt)) / 2

    def get_params(self):
        return list(self.patch_d.parameters()) + list(self.full_d.parameters())

    class Factory:
        @staticmethod
        def create(net_d, criterion): return DoubleGAN(net_d, criterion)

