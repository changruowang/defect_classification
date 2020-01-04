from __future__ import print_function
from collections import defaultdict
from collections import deque
import torch
import torchvision.transforms as transforms
import errno
import os
import time
import datetime
import pickle
import torch.distributed as dist
from sklearn import metrics


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    # local_size = torch.tensor([tensor.numel()], device="cuda")
    # size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    local_size = torch.tensor([tensor.numel()]).cuda()
    size_list = [torch.tensor([0]).cuda() for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        # tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8).cuda())
    if local_size != max_size:
        # padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8).cuda()
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def collate_fn_coco(batch):
    return tuple(zip(*batch))

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=10, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)
'''
作用；迭代一个batch的同时 Print相关信息
'''
class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        self.logger = logger
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.iter_per_epoch = 0
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    def write_log(self, cnt):
        # print(self.meters.items())
        tmp = {}
        if self.logger is not None:
            for k, v in self.meters.items():
                tmp[k] = v.value
            self.logger.add_scalars("all loss", tmp, cnt)

    def add_meter(self, name, window_size=1, fmt='{value:.6f}'):
        self.meters[name] = SmoothedValue(window_size=window_size, fmt=fmt)

    def log_every(self, iterable, print_freq, epoch=None):
        i = 0
        if epoch is None:
            header = 'Test:'
        else:
            header = 'Epoch: [{}]'.format(epoch)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(window_size=1, fmt='{avg:.4f}')
        data_time = SmoothedValue(window_size=1, fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
                if epoch is not None:
                    self.write_log(i + epoch * (self.iter_per_epoch + print_freq))
            i += 1
            end = time.time()
            if epoch == 0:
                self.iter_per_epoch = i
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def get_transform(train=False):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomRotation(10),
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([transforms.Resize(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def save_checkpoint(state, filename):
    torch.save(state, filename)
from sklearn.utils.multiclass import type_of_target

@torch.no_grad()
def classify_evaluate(model, data_loader,print_feq, device, classes=(1,)):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    output_list = []
    labels_list = []
    result = {}
    for images, labels in metric_logger.log_every(data_loader, print_feq):
        images = images.to(device)
        torch.cuda.synchronize()  # 等待cuda所有核上的任务都完成
        model_time = time.time()
        output = model(images)

        labels_list.append(labels.view(-1, 1).to(cpu_device))
        output_list.append(output.to(cpu_device))
        # output_list.append(output.to(cpu_device))

        evaluator_time = time.time() - model_time
        metric_logger.update(evaluator_time=evaluator_time)

    y_val = torch.cat(labels_list).detach().numpy()
    y_pred = torch.cat(output_list).detach().numpy()
    result['accuracy'] = metrics.accuracy_score(y_val, y_pred.argmax(1))
    result['f1_score'] = metrics.f1_score(y_val, y_pred.argmax(1), average='weighted')
    result['y_pred'] = y_pred
    result['y_true'] = y_val

    if isinstance(classes,list) or isinstance(classes,tuple):
        for i in classes:
            y_true = y_val.copy()
            y_true[y_val != i] = 0
            y_true[y_val == i] = 1

            precision, recall, _thresholds = metrics.precision_recall_curve(y_true, y_pred[:,i])
            result['class_{}_pr_auc_score'.format(str(i))] =  metrics.auc(recall, precision)
            result['class_{}_roc_auc_score'.format(str(i))] = metrics.roc_auc_score(y_true, y_pred[:,i])


            print('class {kind:}: pr_auc={pr:.3f}, roc_auc_score={roc:.3f}, average_f1_score={f1:.3f}, acc={acc:.3f}'.format(
                kind=i,
                pr=  result['class_{}_pr_auc_score'.format(str(i))],
                roc= result['class_{}_roc_auc_score'.format(str(i))],
                f1= result['f1_score'],
                acc= result['accuracy']))

    return result

def get_filenames(file_list, with_ext=False):
    file_names = []
    base_path = os.path.dirname(file_list[0])
    for item in file_list:
        [_, name] = os.path.split(item)
        
        if with_ext:
            suffix = name[name.find('.'):]
            name = name[:name.find('.')] + suffix 
        else:
            name = name[:name.find('.')] 
        file_names.append(name)   
    return file_names, base_path