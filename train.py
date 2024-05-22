import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import Conv2d, Linear, ReLU, Flatten, Dropout2d, Dropout
from torch.nn import MaxPool2d
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
import time

# 定义超参数
input_size = 28  # 图像的总尺寸28*28
num_classes = 10  # 标签的种类数
epochs = 10  # 训练的总循环周期
batch_size = 64  # —个批次的大小，64张图片
learning_rate = 0.01  # 学习率
epsilons = [0, .05, .1, .15, .2, .25, .3]
iteration = 10  # 迭代次数
decay = 1.0  # MI-FGSM参数

"""CNN模型"""


class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.model = nn.Sequential(
            Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0),
            Dropout2d(),
            MaxPool2d(2),
            ReLU(),
            Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0),
            Dropout2d(),
            MaxPool2d(2),
            ReLU(),
            Flatten(),
            Linear(320, 50),
            ReLU(),
            Dropout(0.25),
            Linear(50, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


"""数据加载"""


def dataloader():
    # 预处理：将两个步骤整合在一起
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), 0.3081)])  # 将灰度图片像素值（0~255
    # ）转为Tensor（0~1），方便后续处理

    # 加载数据集
    train_data = MNIST(root='../MNIST', train=True, transform=transform, download=True)  # 训练数据集
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = MNIST(root="../MNIST", train=False, transform=transform, download=True)  # 测试数据集
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    data = torch.Tensor(test_data.data[:10000].float()).unsqueeze(1)
    attack_data = TensorDataset((data - data.min()) / (data.max() - data.min()), test_data.targets[:10000])
    attack_loader = DataLoader(dataset=attack_data, batch_size=1, shuffle=False)

    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print("训练数据集的长度：{}".format(train_data_size))
    print("测试数据集的长度：{}".format(test_data_size))

    return train_loader, test_loader, attack_loader


"""模型初始化"""


def model_init():
    # 损失函数CrossentropyLoss
    model = MnistModel()  # 实例化
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，相当于Softmax+Log+NllLoss
    # 线性多分类模型Softmax,给出最终预测值对于10个类别出现的概率，Log:将乘法转换为加法，减少计算量，保证函数的单调性
    # NLLLoss:计算损失，此过程不需要手动one-hot编码，NLLLoss会自动完成

    # SGD，优化器，梯度下降算法e
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)  # lr:学习率
    return model, criterion, optimizer


"""模型训练"""


def train(train_loader, model, criterion, optimizer, train_losses):
    # index = 0
    model.train()
    for index, data in enumerate(train_loader):  # 获取训练数据以及对应标签
        # for data in train_loader:
        optimizer.zero_grad()  # 梯度清零
        input, target = data  # input为输入数据，target为标签
        y_predict = model(input)  # 模型预测
        loss = criterion(y_predict, target)
        loss.backward()  # loss值反向传播
        optimizer.step()  # 更新参数
        # index += 1
        if index % 100 == 0:  # 每一百次保存一次模型，打印损失
            print("训练次数为：{}，损失值为：{}".format(index, loss.item()))
    train_losses.append(loss.item())
    # train_counter.append((index * 64) + (i * len(train_loader.dataset)))
    torch.save(model.state_dict(), "新建文件夹/model.pth")  # 保存模型
    torch.save(optimizer.state_dict(), "新建文件夹/optimizer.pth")
    return train_losses


"""模型测试"""


def test(test_loader, model, criterion, test_losses, test_accuracy):
    model.eval()
    test_loss = 0
    correct = 0  # 正确预测的个数
    total = 0  # 总数
    with torch.no_grad():  # 测试不用计算梯度
        for data in test_loader:
            optimizer.zero_grad()
            input, target = data
            output = model(input)  # output输出10个预测取值，概率最大的为预测数
            probability, predict = torch.max(input=output.data, dim=1)  # 返回一个元祖，第一个为最大概率值，第二个为最大概率值的下标
            loss = criterion(output, target)
            test_loss += loss.item()
            total += target.size(0)  # target是形状为（batch_size,1)的矩阵，使用size（0）取出该批的大小
            correct += (predict == target).sum().item()  # predict 和target均为（batch_size,1)的矩阵，sum求出相等的个数
    accuracy = correct / total
    print("测试准确率为：%.6f" % accuracy)
    test_loss /= len(test_loader.dataset)
    test_losses.append(loss.item())
    test_accuracy.append(accuracy)
    return test_losses, test_accuracy


"""FGSM攻击"""


def fgsm_attack(model, image, label, epsilon, criterion):
    image = image.clone().detach()
    label = label.clone().detach()
    image.requires_grad = True
    # Classify the image with the trained model
    output = model(image)
    # Calculate the loss
    loss = criterion(output, label)
    # # Calculate gradients of model in backward pass
    model.zero_grad()
    # loss.backward()
    # # Collect datagrad
    # data_grad = image.grad.data
    data_grad = torch.autograd.grad(
        loss, image, retain_graph=False, create_graph=False
    )[0]
    # 通过调整输入图像的每个像素来创建扰动图像
    adv_image = image + epsilon * data_grad.sign()
    # 添加剪切以维持[0,1]范围
    adv_image = torch.clamp(adv_image, 0, 1)
    # 返回被扰动的图像
    return adv_image, loss.item()


def ifgsm_attack(model, image, label, epsilon, iteration=10):
    alpha = epsilon / iteration
    image = image.clone().detach()
    label = label.clone().detach()
    ori_image = image.clone().detach()
    iter_loss = 0
    for _ in range(iteration):
        image.requires_grad = True
        output = model(image)
        # model.zero_grad()
        # Calculate loss
        loss = criterion(output, label)
        iter_loss += loss.item()
        # Update adversarial images
        grad = torch.autograd.grad(
            loss, image, retain_graph=False, create_graph=False
        )[0]
        adv_image = image.detach() + alpha * grad.sign()
        a = torch.clamp(ori_image - epsilon, min=0)  # a = max{0, X-eps}
        b = (adv_image >= a).float() * adv_image + (adv_image < a).float() * a  # b = max{a, X'}
        c = (b > ori_image + epsilon).float() * (ori_image + epsilon) + (b <= ori_image + epsilon).float() * b  # c = min{X+eps, b}
        image = torch.clamp(c, max=1).detach()  # d = min{1, c}

    return image, iter_loss / iteration


# def ifgsm_attack(input,epsilon,data_grad):
#   iter = 10
#   alpha = epsilon/iter
#   pert_out = input
#   for i in range(iter-1):
#     pert_out = pert_out + alpha*data_grad.sign()
#     pert_out = torch.clamp(pert_out, 0, 1)
#     if torch.norm((pert_out-input),p=float('inf')) > epsilon:
#       break
#   return pert_out
# I-FGSM attack code
# def ifgsm_attack(model, image, label, epsilon, criterion, iteration=10):
#     alpha = epsilon / iteration
#     image = image.clone().detach()
#     adv_image = image.clone().detach()
#     for i in range(iteration):
#         # Set requires_grad attribute of tensor
#         adv_image.requires_grad = True
#         # Classify the image with the trained model
#         output = model(adv_image)
#         # Calculate the loss
#         cost = criterion(output, label)
#         # Calculate gradients of model in backward pass
#         model.zero_grad()
#         cost.backward()
#         # Collect datagrad
#         data_grad = adv_image.grad.data
#         # Create the perturbed image by adjusting each pixel of the input image with the element-wise sign of the
#         # data gradient
#         adv_image = adv_image.detach() + alpha * data_grad.sign()
#
#         # Adding clipping so that the image alteration doesn't exceed epsilon
#         eta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
#         # Adding clipping to maintain [0,1] range
#         adv_image = torch.clamp(image + eta, min=0, max=1).detach()
#     return adv_image


# def mifgsm_attack(model, image, label, epsilon, iteration=10, decay=0.2):
#     r"""
#     Overridden.
#     """
#     alpha = epsilon / iteration
#     image = image.clone().detach()
#     label = label.clone().detach()
#     momentum = torch.zeros_like(image).detach()
#     adv_image = image.clone().detach()
#     for i in range(iteration):
#         adv_image.requires_grad = True
#         output = model(adv_image)
#         # Calculate loss
#         cost = criterion(output, label)
#
#         # Update adversarial images
#         grad = torch.autograd.grad(cost, adv_image, retain_graph=False, create_graph=False)[0]
#
#         grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
#         grad = grad + momentum * decay
#         momentum = grad
#
#         adv_image = adv_image.detach() + alpha * grad.sign()
#         delta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
#         adv_image = torch.clamp(image + delta, min=0, max=1).detach()
#
#     return adv_image

# MI-FGSM attack code
def mifgsm_attack(model, image, label, epsilon, criterion, iteration=10, decay=0.2):
    alpha = epsilon / iteration
    # Initialize momentum
    momentum = torch.zeros_like(image).detach()
    image = image.clone().detach()
    adv_image = image.clone().detach()
    iter_loss = 0
    for i in range(iteration):
        # Set requires_grad attribute of tensor
        adv_image.requires_grad = True
        # Classify the image with the trained model
        output = model(adv_image)
        # Calculate the loss
        loss = criterion(output, label)
        iter_loss += loss.item()
        # Calculate gradients of model in backward pass
        # model.zero_grad()
        # loss.backward()
        # Collect datagrad
        data_grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
        grad = data_grad / torch.mean(torch.abs(data_grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay
        momentum = grad
        # Create the perturbed image by adjusting each pixel of the input image with the element-wise sign of the
        # data gradient
        adv_image = adv_image.detach() + alpha * grad.sign()
        # Adding clipping so that the image alteration doesn't exceed epsilon
        eta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
        # Adding clipping to maintain [0,1] range
        adv_image = torch.clamp(image + eta, min=0, max=1).detach()
        image = adv_image.clone().detach()
    return adv_image, iter_loss / iteration


# def mifgsm_attack(input, epsilon, data_grad):
#     iter = 20
#     decay_factor = 1.0
#     pert_out = input
#     alpha = epsilon / iter
#     g = 0
#     for i in range(iter - 1):
#         g = decay_factor * g + data_grad / torch.norm(data_grad, p=1)
#         pert_out = pert_out + alpha * torch.sign(g)
#         pert_out = torch.clamp(pert_out, 0, 1)
#         if torch.norm((pert_out - input), p=float('inf')) > epsilon:
#             break
#     return pert_out

# def mifgsm_attack(input, epsilon, data_grad):
#     iter = 10
#     decay_factor = 1.0
#     pert_out = input
#     alpha = epsilon / iter
#     g = 0
#     for i in range(iter - 1):
#         g = decay_factor * g + data_grad / torch.norm(data_grad, p=1)
#         pert_out = pert_out + alpha * torch.sign(g)
#         pert_out = torch.clamp(pert_out, 0, 1)
#         if torch.norm((pert_out - input), p=float('inf')) > epsilon:
#             break
#     return pert_out

# # restores the tensors to their original scale
# def denorm(batch, mean=[0.1307], std=[0.3081]):
#     """
#     Convert a batch of tensors to their original scale.
#
#     Args:
#         batch (torch.Tensor): Batch of normalized tensors.
#         mean (torch.Tensor or list): Mean used for normalization.
#         std (torch.Tensor or list): Standard deviation used for normalization.
#
#     Returns:
#         torch.Tensor: batch of tensors without normalization applied to them.
#     """
#     if isinstance(mean, list):
#         mean = torch.tensor(mean)
#     if isinstance(std, list):
#         std = torch.tensor(std)
#
#     return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

# def pgd_attack(model, image, target, epsilon, iteration=10, alpha=2):
#     r"""
#         Overridden.
#         """
#     alpha = epsilon / iteration
#     image = image.clone().detach()
#     target = target.clone().detach()
#     adv_image = image.clone().detach()
#     for i in range(iteration):
#         adv_image.requires_grad = True
#         outputs = model(adv_image)
#         # Calculate loss
#         cost = criterion(outputs, target)
#         # Update adversarial images
#         grad = torch.autograd.grad(
#             cost, adv_image, retain_graph=False, create_graph=False
#         )[0]
#         adv_image = adv_image.detach() + alpha * grad.sign()
#         delta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
#         adv_image = torch.clamp(image + delta, min=0, max=1).detach()
#
#     return adv_image

# PGD attack code
def pgd_attack(model, image, target, epsilon, criterion, iteration=10):
    image = image.clone().detach()
    target = target.clone().detach()
    adv_image = image.clone().detach()
    alpha = epsilon / iteration
    iter_loss = 0

    adv_image = adv_image + torch.empty_like(adv_image).uniform_(-epsilon, epsilon)
    adv_image = torch.clamp(adv_image, min=0, max=1).detach()

    for i in range(iteration):
        adv_image.requires_grad = True
        # model.zero_grad()
        output = model(adv_image)
        loss = criterion(output, target)
        iter_loss += loss.item()
        # loss.backward()
        data_grad = torch.autograd.grad(
                loss, adv_image, retain_graph=False, create_graph=False
            )[0]
        adv_image = adv_image.detach() + alpha * data_grad.sign()

        adv_image = torch.min(torch.max(adv_image, image - epsilon), image + epsilon)
        adv_image = torch.clamp(adv_image, 0, 1)
        # Projection
        # delta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
        # adv_image = torch.clamp(image + delta, min=0, max=1).detach()
    return adv_image, iter_loss / iteration


def attack(model, attack_loader, criterion, epsilon, attack):
    # 精度计数器
    correct = 0
    total = 0
    all_loss = 0
    adv_examples = []
    number = 0
    number_examples = []
    # 循环遍历测试集中的所有示例
    for data in attack_loader:

        # 设置张量的requires_grad属性，这对于攻击很关键
        input, target = data
        input.requires_grad = True
        # 通过模型前向传递数据
        output = model(input)
        probability, predict = torch.max(input=output.data, dim=1)  # 返回一个元祖，第一个为最大概率值，第二个为最大概率值的下标

        # 如果初始预测是错误的，不打断攻击，继续
        if predict.item() != target.item():
            continue
        # 计算损失
        # loss = criterion(output, target)
        # # 将所有现有的渐变归零
        # model.zero_grad()
        # # 计算后向传递模型的梯度
        # loss.backward()
        # # 收集datagrad
        # data_grad = input.grad.data
        # # Restore the data to its original scale
        # data_denorm = denorm(data)

        if attack == 'FGSM':
            attack_data, attack_loss = fgsm_attack(model, input, target, epsilon, criterion)  # 唤醒FGSM进行攻击
        elif attack == 'IFGSM':
            attack_data, attack_loss = ifgsm_attack(model, input, target, epsilon, iteration=iteration)  # 唤醒I-FGSM进行攻击
        elif attack == 'MIFGSM':
            attack_data, attack_loss = mifgsm_attack(model, input, target, epsilon, criterion, iteration=iteration,
                                                     decay=decay)  # 唤醒MI-FGSM进行攻击
        elif attack == 'PGD':
            attack_data, attack_loss = pgd_attack(model, input, target, epsilon, criterion,
                                                  iteration=iteration)  # 唤醒PGD进行攻击
        else:
            print('wrong')

        all_loss += attack_loss
        # # Reapply normalization
        # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(attack_data)
        # 重新分类受扰乱的图像
        attack_output = model(attack_data)
        # 检查是否成功
        attack_probability, attack_predict = torch.max(input=attack_output.data, dim=1)
        total = float(len(attack_loader))
        # final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        if number == 0:
            number = 1
            number_ex = attack_data.squeeze().detach().cpu().numpy()
            number_examples.append(number_ex)
        if attack_predict.item() == target.item():
            correct += 1
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = attack_data.squeeze().detach().cpu().numpy()
                # if epsilon == 0 or epsilon == 0.15 or epsilon == 0.3:
                adv_examples.append((target.item(), attack_predict.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = attack_data.squeeze().detach().cpu().numpy()
                # if epsilon == 0 or epsilon == 0.15 or epsilon == 0.3:
                adv_examples.append((target.item(), attack_predict.item(), adv_ex))

    final_acc = correct / total
    average_loss = all_loss / len(attack_loader.dataset)
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}\tLoss = {}".format(epsilon, correct, total, final_acc, average_loss))

    return final_acc, adv_examples, average_loss, number_examples


# def IFGSM(model, fgsm_loader, criterion, epsilon):
#     # 精度计数器
#     correct = 0
#     total = 0
#     adv_examples = []
#
#     # 循环遍历测试集中的所有示例
#     for data in fgsm_loader:
#         # 设置张量的requires_grad属性，这对于攻击很关键
#         input, target = data
#         input.requires_grad = True
#         # 通过模型前向传递数据
#         output = model(input)
#         probability, predict = torch.max(input=output.data, dim=1)  # 返回一个元祖，第一个为最大概率值，第二个为最大概率值的下标
#
#         # 如果初始预测是错误的，不打断攻击，继续
#         if predict.item() != target.item():
#             continue
#         # 计算损失
#         # loss = criterion(output, target)
#         # # 将所有现有的渐变归零
#         # model.zero_grad()
#         # # 计算后向传递模型的梯度
#         # loss.backward()
#         # # 收集datagrad
#         # data_grad = input.grad.data
#         # # Restore the data to its original scale
#         # data_denorm = denorm(data)
#         # 唤醒FGSM进行攻击
#         perturbed_data = ifgsm_attack(model, input, target, epsilon)
#         # # Reapply normalization
#         # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
#         # 重新分类受扰乱的图像
#         fgsm_output = model(perturbed_data)
#         # 检查是否成功
#         fgsm_probability, fgsm_predict = torch.max(input=fgsm_output.data, dim=1)
#         total += 1
#         # final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         if fgsm_predict.item() == target.item():
#             correct += 1
#             if (epsilon == 0) and (len(adv_examples) < 5):
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append((fgsm_predict.item(), fgsm_predict.item(), adv_ex))
#
#             if len(adv_examples) < 5:
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append((fgsm_predict.item(), fgsm_predict.item(), adv_ex))
#
#     final_acc = correct / total
#     print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))
#
#     return final_acc, adv_examples
#
#
# def MIFGSM(model, fgsm_loader, criterion, epsilon):
#     # 精度计数器
#     correct = 0
#     total = 0
#     adv_examples = []
#
#     # 循环遍历测试集中的所有示例
#     for data in fgsm_loader:
#         # 设置张量的requires_grad属性，这对于攻击很关键
#         input, target = data
#         input.requires_grad = True
#         # 通过模型前向传递数据
#         output = model(input)
#         probability, predict = torch.max(input=output.data, dim=1)  # 返回一个元祖，第一个为最大概率值，第二个为最大概率值的下标
#
#         # 如果初始预测是错误的，不打断攻击，继续
#         if predict.item() != target.item():
#             continue
#         # 计算损失
#         # loss = criterion(output, target)
#         # # 将所有现有的渐变归零
#         # model.zero_grad()
#         # # 计算后向传递模型的梯度
#         # loss.backward()
#         # # 收集datagrad
#         # data_grad = input.grad.data
#         # grad = torch.autograd.grad(loss, input, retain_graph=False, create_graph=False)[0]
#         # grad_norm = torch.norm(grad, p=1)
#         # data_grad = grad / grad_norm
#
#         # # Restore the data to its original scale
#         # data_denorm = denorm(data)
#         # 唤醒FGSM进行攻击
#         perturbed_data = mifgsm_attack(model, input, target, epsilon)
#         # # Reapply normalization
#         # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
#         # 重新分类受扰乱的图像
#         fgsm_output = model(perturbed_data)
#         # 检查是否成功
#         fgsm_probability, fgsm_predict = torch.max(input=fgsm_output.data, dim=1)
#         total += 1
#         # final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         if fgsm_predict.item() == target.item():
#             correct += 1
#             if (epsilon == 0) and (len(adv_examples) < 5):
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append((fgsm_predict.item(), fgsm_predict.item(), adv_ex))
#
#             if len(adv_examples) < 5:
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append((fgsm_predict.item(), fgsm_predict.item(), adv_ex))
#
#     final_acc = correct / total
#     print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))
#
#     return final_acc, adv_examples
#
#
# def PGD(model, fgsm_loader, criterion, epsilon):
#     # 精度计数器
#     correct = 0
#     total = 0
#     adv_examples = []
#
#     # 循环遍历测试集中的所有示例
#     for data in fgsm_loader:
#         # 设置张量的requires_grad属性，这对于攻击很关键
#         input, target = data
#         input.requires_grad = True
#         # 通过模型前向传递数据
#         output = model(input)
#         probability, predict = torch.max(input=output.data, dim=1)  # 返回一个元祖，第一个为最大概率值，第二个为最大概率值的下标
#
#         # 如果初始预测是错误的，不打断攻击，继续
#         if predict.item() != target.item():
#             continue
#         # 计算损失
#         # loss = criterion(output, target)
#         # # 将所有现有的渐变归零
#         # model.zero_grad()
#         # # 计算后向传递模型的梯度
#         # loss.backward()
#         # # 收集datagrad
#         # data_grad = input.grad.data
#         # # Restore the data to its original scale
#         # data_denorm = denorm(data)
#         # 唤醒FGSM进行攻击
#         perturbed_data = pgd_attack(model, input, target, epsilon)
#         # # Reapply normalization
#         # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
#         # 重新分类受扰乱的图像
#         fgsm_output = model(perturbed_data)
#         # 检查是否成功
#         fgsm_probability, fgsm_predict = torch.max(input=fgsm_output.data, dim=1)
#         total += 1
#         # final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
#
#         if fgsm_predict.item() == target.item():
#             correct += 1
#             if (epsilon == 0) and (len(adv_examples) < 5):
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append((fgsm_predict.item(), fgsm_predict.item(), adv_ex))
#
#             if len(adv_examples) < 5:
#                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
#                 adv_examples.append((fgsm_predict.item(), fgsm_predict.item(), adv_ex))
#
#     final_acc = correct / total
#     print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, total, final_acc))
#
#     return final_acc, adv_examples


# 测试识别函数
if __name__ == '__main__':
    # 加载数据
    train_loader, test_loader, attack_loader = dataloader()
    # 模型初始化
    model, criterion, optimizer = model_init()
    acc_test = [0.9366, 0.9589, 0.9663, 0.9693, 0.9711, 0.9754, 0.9766, 0.9780, 0.9789, 0.9798]
    # 加载模型

    if os.path.exists('新建文件夹/model.pth') and os.path.exists('新建文件夹/optimizer.pth'):
        model.load_state_dict(torch.load("新建文件夹/model.pth"))  # 加载保存模型的参数
        optimizer.load_state_dict(torch.load('新建文件夹/optimizer.pth'))
        print('已加载模型')
        model.eval()
    else:
        train_losses = []
        test_losses = []
        test_accuracy = []

        # 训练与测试
        for i in range(epochs):  # 训练和测试进行10轮
            print({"————————第{}轮测试开始——————".format(i + 1)})
            train_losses = train(train_loader, model, criterion, optimizer, train_losses)
            test_losses, test_accuracy = test(test_loader, model, criterion, test_losses, test_accuracy)
        plt.figure()
        plt.plot(np.arange(epochs), train_losses, label="Training loss")
        plt.plot(np.arange(epochs), test_losses, label="Validation loss")
        plt.title("Training and Validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.figure()
        plt.plot(np.arange(epochs), test_accuracy, label="Testing accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
    # 对抗攻击
    accuracies = []
    examples = []
    losses = []
    num_examples = []
    start_time = time.time()
    # for epsilon in epsilons:
    #     final_acc, adv_examples, attack_losses, number_examples = attack(model, attack_loader, criterion, epsilon, attack='FGSM')
    #     num_examples.append(number_examples)
    # for epsilon in epsilons:
    #     final_acc, adv_examples, attack_losses, number_examples = attack(model, attack_loader, criterion, epsilon, attack='IFGSM')
    #     num_examples.append(number_examples)
    # for epsilon in epsilons:
    #     final_acc, adv_examples, attack_losses, number_examples = attack(model, attack_loader, criterion, epsilon, attack='MIFGSM')
    #     num_examples.append(number_examples)
    for epsilon in epsilons:
        final_acc, adv_examples, attack_losses, number_examples = attack(model, attack_loader, criterion, epsilon, attack='PGD')
        num_examples.append(number_examples)
        accuracies.append(final_acc)
        examples.append(adv_examples)
        losses.append(attack_losses)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"程序运行时间：{int(elapsed_time//60)}m{int(elapsed_time%60)}s")
    plt.figure()
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.plot(epsilons, accuracies, "*-")
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()

    plt.figure()
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.plot(epsilons, losses, "*-")
    plt.title("Loss vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Loss")
    plt.show()

    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons), len(examples[0]), cnt)
            # plt.subplots_adjust(hspace=0.1, wspace=0.1)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
            orig, adv, ex = examples[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(ex, cmap="gray")
    plt.tight_layout()
    plt.show()


    # algorithm = ['FGSM', 'I-FGSM', 'MI-FGSM', 'PGD']
    # cnt = 0
    # plt.figure(figsize=(8, 10))
    # for i in range(len(epsilons)):
    #     for j in range(len(num_examples)//7):
    #         cnt += 1
    #         plt.subplot(len(epsilons), len(num_examples)//7, cnt)
    #         # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    #         plt.xticks([], [])
    #         plt.yticks([], [])
    #
    #         if j == 0:
    #             plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=12)
    #         ex = num_examples[cnt-1][0]
    #         if cnt <= 4:
    #             plt.title(f"{algorithm[cnt-1]}", fontsize=12)
    #         plt.imshow(ex, cmap="gray")
    #         plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # # plt.tight_layout()
    # plt.show()

    # algorithm = ['FGSM', 'I-FGSM', 'MI-FGSM', 'PGD']
    # cnt = 0
    # plt.figure(figsize=(8, 10))
    # for j in range(len(num_examples)//7):
    #     for i in range(len(epsilons)):
    #         cnt += 1
    #         plt.subplot(len(num_examples)//7, len(epsilons), cnt)
    #         # plt.subplots_adjust(hspace=0.1, wspace=0.1)
    #         plt.xticks([], [])
    #         plt.yticks([], [])
    #
    #         if i == 0:
    #             plt.ylabel(f"{algorithm[j]}", fontsize=12)
    #         ex = num_examples[cnt-1][0]
    #         if cnt <= 7:
    #             plt.title("Eps: {}".format(epsilons[cnt-1]), fontsize=12)
    #         plt.imshow(ex, cmap="gray")
    #         plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # # plt.tight_layout()
    # plt.show()