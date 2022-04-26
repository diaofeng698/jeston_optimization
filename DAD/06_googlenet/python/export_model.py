import torch
import torchvision
import onnxsim
import onnx
import onnxruntime as ort
import numpy as np
import os

cur_dir = os.path.split(os.path.realpath(__file__))[0]
print(cur_dir)
model_save_path = os.path.realpath(
    os.path.join(os.path.dirname(cur_dir), "model"))
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

net_name = "googlenet"

onnx_name = "{0}/{1}.onnx".format(model_save_path, net_name)
onnx_simplify_name = "{0}/{1}_simplify.onnx".format(model_save_path, net_name)

dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
model = torchvision.models.googlenet(pretrained=True).cuda()
# model.save()

input_names = ["data"]
output_names = ["prob"]

torch.onnx.export(model, dummy_input, onnx_name, verbose=True,
                  input_names=input_names, output_names=output_names)

# simplify onnx
print("##############simplify onnx#####################")
onnx_model = onnx.load(onnx_name)  # load onnx model​
model_simp, check = onnxsim.simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, onnx_simplify_name)

# check onnx
print("##############check onnx#####################")
onnx_simplify = onnx.load(onnx_simplify_name)  # load onnx model​
onnx.checker.check_model(onnx_simplify)
# print(onnx.helper.printable_graph(onnx_simplify.graph))


print("##############inference onnx#####################")
ort_session = ort.InferenceSession(onnx_simplify_name)
outputs = ort_session.run(
    None,
    {"data": np.ones(shape=[1, 3, 224, 224]).astype(np.float32)},
)
print(outputs[0])
