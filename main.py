import torch
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.models.vgg import VGG19_Weights
import logging
from datetime import datetime

import copy

current_time = datetime.now().strftime('%Y-%m-%d %H.%M.%S')
# 配置日志
logging.basicConfig(filename='logs/style_transfer_{}.log'.format(current_time), filemode="w", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 创建一个StreamHandler实例，名为console。StreamHandler是一个日志处理器，用于将日志输出到类似文件或系统控制台的流中。
# 在这个例子中，它被配置为将日志输出到控制台（即标准输出）。
console = logging.StreamHandler()

# 设置console日志处理器的日志级别为INFO。这意味着只有INFO级别及以上（WARNING, ERROR, CRITICAL）的日志才会被这个处理器处理。
console.setLevel(logging.INFO)

# 创建一个Formatter实例，用于定义日志的格式。这里定义的格式是：时间戳 - 日志级别 - 日志消息。
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # %(asctime)s 表示时间戳，%(levelname)s 表示日志级别，%(message)s 表示日志消息。

# 将上面定义的格式器formatter应用到console日志处理器上。
# 这样，所有通过console处理器处理的日志都将按照指定的格式输出。
console.setFormatter(formatter)

# 获取根日志记录器（即默认的日志记录器），并给它添加一个处理器（在这个例子中是console）。
# 这意味着所有通过根日志记录器（或未明确指定日志记录器的日志）都将被发送到console处理器，并按照设置的格式输出到控制台。
# 注意：这里的''（空字符串）是获取根日志记录器的标准方式。
logging.getLogger('').addHandler(console)

# 设置图片大小
img_size = 512

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                 std=[0.229, 0.224, 0.225])
# 加载图片
def load_img(img_path):
    # 使打开的图片通道为RGB格式,如果不使用.convert('RGB')进行转换的话，读出来的图像是RGBA四通道的，A通道为透明通道，该对深度学习模型训练来说暂时用不到，因此使用convert('RGB')进行通道转换。
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))  # 对图片进行裁剪，为512x512
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0)  # unsqueeze升维，使数据格式符合[batch_size, n_channels, hight, width],[1,3,512,512]
    return img

# 显示图片
def show_img(tensor, save_path=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)  # 图片第一维为batch_size，将其降维
    if save_path is not None:
        transform = transforms.ToPILImage()
        image_pil = transform(image)
        image_pil.save(save_path)
    return image

# 构建神经网络
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        # self.vgg = models.vgg19(pretrained=True).features  # .features用于提取卷积层
        self.vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)  # name为第几层的序列号，layer就是卷积层,,x为输入的图片。x = layer(x)的意思是，x经过layer层卷积后再赋值给x
            if name in self.select:
                features.append(x)

        return features

# for name, layer in models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features._modules.items():
#     print(name)
#     print(layer)

# 加载图片
content_img = load_img("images/content_img.jpg")
style_img = load_img("images/赛博风格.png")

# 图片输入到gpu，否则就会报错
content_img = content_img.to(device)
style_img = style_img.to(device)

target = content_img.clone().requires_grad_(True)  # clone()操作后的tensor requires_grad=True，clone操作在不共享数据内存的同时支持梯度梯度传递与叠加
optimizer = torch.optim.Adam([target], lr=0.003)  # 选择优化器
vgg = VGGNet().to(device).eval()
total_step = 3000  # 训练次数

content_weight = 1  # 给content_loss加上的权重
style_weight = 100  # 给style_loss加上的权重

# 设置tensorboard，用于可视化
writer = SummaryWriter("runs")

"""detach() 函数的作用是返回一个与当前计算图（graph）分离的新张量（tensor），
该张量不再需要计算梯度。这意呀着，当你对返回的张量进行任何操作时，这些操作不会影响到原始张量所在的计算图，
也不会被PyTorch的自动求导系统（autograd）所追踪。"""
content_features = [x.detach() for x in vgg(content_img)]
style_features = [x.detach() for x in vgg(style_img)]

# 开始训练
for step in range(1, total_step+1):
    target_features = vgg(target)

    style_loss = 0
    content_loss = 0
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        content_loss = torch.mean((f1 - f2) ** 2) + content_loss
        _, c, h, w = f1.size()  # 结果为torch.Size([1, 64, 512, 512])
        f1 = f1.view(c, h * w)  # 处理数据格式为后面gram计算
        f3 = f3.view(c, h * w)

        # 计算gram matrix
        f1 = torch.mm(f1, f1.t())  # torch.mm()两个矩阵相乘,.t()是矩阵倒置
        f3 = torch.mm(f3, f3.t())
        style_loss = torch.mean((f1 - f3) ** 2) / (c * h * w) + style_loss

    loss = content_weight * content_loss + style_weight * style_loss

    # 更新target
    optimizer.zero_grad()  # 每一次优化都要梯度清零
    loss.backward()  # 反向传播
    optimizer.step()
    writer.add_scalar("loss", loss, step)
    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    img = target.clone().squeeze()  # 升维-->降维
    img = denorm(img).clamp_(0, 1)
    if step % 500 == 0:   # 每500轮保存图片
        img = show_img(img, "result/step{}.png".format(step))
    else:
        img = show_img(img)
    writer.add_image("target", img, global_step=step)
    # print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}"
    #       .format(step, total_step, content_loss.item(), style_loss.item()))
    logging.info("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}, loss: {:.2f}"
                 .format(step, total_step, content_loss.item(), style_loss.item(), loss.item()))

    # if step % 500 == 0:
    #     current_model = copy.deepcopy(vgg.state_dict())
    #     torch.save(current_model, "save_model/model_{}.pth".format(step))
writer.close()
