
import os.path

import torch
import numpy as np
import pointnet_part_seg as MODEL
from pointnet_utils import  pc_normalize


point_num = 5000
num_class = 4
normal = False

model_dir = "./"
model = MODEL.get_model(num_class)
model = model.to("cuda")
model.eval()

with torch.no_grad():

    checkpoint = torch.load(model_dir + 'checkpoints/best_model.pth',  map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint['model_state_dict'])

    data = np.loadtxt("./0001_label_22.txt")[:, 0:3]
    data = pc_normalize(data)
    data = data.T
    data = np.expand_dims(data, axis=0).astype(float)
    #x = torch.rand(1, 3, point_num)

    x = torch.from_numpy(data)
    x = x.to(torch.float32).to(device="cuda")


    export_onnx_file = os.path.join(model_dir, "checkpoints/best_model_gpu.onnx")
    torch.onnx.export(model, x, export_onnx_file, opset_version=11)

