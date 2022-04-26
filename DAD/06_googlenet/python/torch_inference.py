import torch
from torch import nn
import torchvision
from PIL import Image
from torchvision import transforms
import sys
import os

cur_dir = os.path.split(os.path.realpath(__file__))[0]
print(cur_dir)

def main():
    print('cuda device count: ', torch.cuda.device_count())
    net = torchvision.models.googlenet(pretrained=True).cuda()
    net = net.eval()
    tmp = torch.ones(1, 3, 224, 224).to('cuda:0')
    out = net(tmp)

    print('output:', out)


def inference(img_name):
    # sample execution (requires torchvision)
    input_image = Image.open(img_name)
    print(input_image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # print(input_tensor)
    # input_tensor.save("pil_crop_img.png")
    # return
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    model = torchvision.models.googlenet(pretrained=True).cuda()
    model = model.eval()

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    # print(probabilities)

    # Read the categories
    label_file = os.path.join(os.path.dirname(cur_dir), "data", "imagenet_classes.txt")
    with open(label_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print(top5_catid)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        main()
    else:
        inference(sys.argv[1])
