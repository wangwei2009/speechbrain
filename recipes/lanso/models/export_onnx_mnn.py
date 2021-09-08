# coding:utf-8
#

import os
import MNN
import torch
import argparse
import onnxruntime
import numpy as np
from torchsummary import summary

# from CRN_Att import crn_att
# from BIGRU import CustomModel
# from res8 import Res8
# from tdnn3 import TDNN


def export_onnx_mnn(pth_model, dummy_input, output_model='model.mnn'):

    pth_model.eval()

    pth_output = pth_model(dummy_input)
    print("pth_output:{}".format(pth_output[0]))
    pth_output = pth_output.detach().numpy()

    onnx_model = output_model.split('.')[0] + '.onnx'
    # Export the model
    torch.onnx.export(pth_model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      onnx_model,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True  # whether to execute constant folding for optimization
                      )
    #
    onnx_model_session = onnxruntime.InferenceSession(onnx_model)

    print("onnx input:{}".format(onnx_model_session.get_inputs()[0].shape))
    print("onnx output:{}".format(onnx_model_session.get_outputs()[0].shape))

    inputs = {onnx_model_session.get_inputs()[0].name: dummy_input.detach().numpy()}
    onnx_output = onnx_model_session.run(None, inputs)[0]
    print("difference between pth and onnx:{}".format(np.sum(np.exp(onnx_output) - np.exp(pth_output))))


    # MNN_output = os.path.join(os.path.dirname(output_model), 'tdnn3.mnn')
    MNN_output = output_model
    os.system("mnnconvert  -f ONNX --bizCode biz --modelFile {} --MNNModel {}".format(onnx_model, MNN_output))
    print("save mnn model to :{}".format(MNN_output))

    interpreter = MNN.Interpreter(MNN_output)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    input_shape = input_tensor.getShape()
    print("input_shape = {}".format(input_shape))

    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    print("output_shape = {}".format(output_shape))


    tmp_input = MNN.Tensor((input_shape[0], input_shape[1], input_shape[3], input_shape[2]), MNN.Halide_Type_Float,
                           dummy_input.detach().numpy(),
                           MNN.Tensor_DimensionType_Caffe)
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)

    # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    tmp_output = MNN.Tensor((output_shape[-2], output_shape[-1]), MNN.Halide_Type_Float, np.ones([output_shape[-2], output_shape[-1]]).astype(np.float32),
                            MNN.Tensor_DimensionType_Caffe)
    output_tensor.copyToHostTensor(tmp_output)
    mnn_output = tmp_output.getData()
    mnn_output = np.array(mnn_output).reshape([output_shape[-2], output_shape[-1]])
    print("mnn_output:{}".format(mnn_output[0]))
    print("difference between pth and mnn:{}".format(np.sum(np.exp(mnn_output) - np.exp(pth_output[0, :]))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='export onnx and mnn model')

    parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
    parser.add_argument('--half', action="store_true",
                        help='Use half precision. This is recommended when using mixed-precision at training time')
    parser.add_argument('--net-arch', default='TDNNet',
                        help="model architecture: TDNNet/TDNNetV2/TDNNetV3")
    parser.add_argument('--model-path', default='models/tdnn_final.pth',
                        help='Path to model file created by training')
    parser.add_argument('--mean-std-path', default='models/mean_std.npz',
                        help='Path to mean std file')

    args = parser.parse_args()
    print("***************************************")
    print("Load file:")
    for param, value in args.__dict__.items():
        print("{} = {}".format(param, value))
    print("***************************************")
    device = torch.device("cuda" if args.cuda else "cpu")
    print('device: {} '.format(device))
    # Arch_dict = {'TDNNet': TDNNet,
    #              'TDNNetV2': TDNNetV2,
    #              'TDNNetV3': TDNNetV3}
    # # mean std
    # mean_std = np.load(args.mean_std_path)
    # mean_array = mean_std["mean"]
    # std_array = mean_std["std"]
    # save_path = os.path.dirname(args.mean_std_path)
    # np.savetxt('{}/mean.txt'.format(save_path), mean_array)
    # np.savetxt('{}/std.txt'.format(save_path), std_array)
    #
    # pth_model = load_model(Arch_dict[args.net_arch], device, args.model_path, args.half)
    # pth_model = mobilenetv2_19()

    pth_model = crn_att(input_size=40, rnn_type='LSTM', layers=1)
    # pth_model = CustomModel(40, 4)
    # pth_model = Res8()
    pth_model = TDNN()
    pth_model.eval()

    #
    # try:
    #     lfr_num = pth_model.audio_conf["lfr_num"]
    # except:
    #     lfr_num = 1
    # try:
    #     feature_type = pth_model.audio_conf['feature_type']
    # except:
    #     feature_type = "spec"
    # if feature_type == "spec":
    #     input_channels = int(161 * lfr_num)
    # elif feature_type == "fbank":
    #     input_channels = 80 * lfr_num
    # elif feature_type == "mfcc":
    #     input_channels = 39 * lfr_num
    # else:
    #     raise ValueError("wrong type of feature type : [{}]".format(feature_type))
    # input_data = torch.randn(1, 1, 132, input_channels).to(device)  # 形状固定后，生成的mnn模型输入形状不可修改
    # input_data = torch.randn(1, 3, 224, 224).to(device)  # 形状固定后，生成的mnn模型输入形状不可修改

    # N, C, T, F = 1, 1, 151, 40
    input_data = torch.rand((1, 1, 151, 40))

    #
    pth_output = pth_model(input_data)
    pth_output = pth_output.detach().numpy()
    #
    # summary(pth_model, input_data[0, :])

    output_model = os.path.join(os.path.dirname(args.model_path), 'tdnn3.onnx')
    print("save onnx model to :{}".format(output_model))

    dummy_input = torch.randn(1,1,151,40)

    # Export the model
    torch.onnx.export(pth_model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      output_model,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True  # whether to execute constant folding for optimization
                      )
    #
    onnx_model_session = onnxruntime.InferenceSession(output_model)

    print("onnx input:{}".format(onnx_model_session.get_inputs()[0].shape))
    print("onnx output:{}".format(onnx_model_session.get_outputs()[0].shape))

    inputs = {onnx_model_session.get_inputs()[0].name: input_data.detach().numpy()}
    onnx_output = onnx_model_session.run(None, inputs)[0]
    print("difference between pth and onnx:{}".format(onnx_output - pth_output))

    
    MNN_output = os.path.join(os.path.dirname(output_model), 'tdnn3.mnn')
    os.system("mnnconvert  -f ONNX --bizCode biz --modelFile {} --MNNModel {}".format(output_model, MNN_output))
    print("save mnn model to :{}".format(MNN_output))

    #
    interpreter = MNN.Interpreter(MNN_output)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    input_shape = input_tensor.getShape()
    print("input_shape = {}".format(input_shape))

    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    print("output_shape = {}".format(output_shape))

    # tmp_input = MNN.Tensor((1, 1, 161, 132), MNN.Halide_Type_Float,
    #                        input_data.detach().numpy(),
    #                        MNN.Tensor_DimensionType_Caffe)
    # input_tensor.copyFrom(tmp_input)
    # interpreter.runSession(session)
    # output_tensor = interpreter.getSessionOutput(session)

    # # constuct a tmp tensor and copy/convert in case output_tensor is nc4hw4
    # tmp_output = MNN.Tensor((110, 327), MNN.Halide_Type_Float, np.ones([110, 327]).astype(np.float32),
    #                         MNN.Tensor_DimensionType_Caffe)
    # output_tensor.copyToHostTensor(tmp_output)
    # mnn_output = tmp_output.getData()
    # mnn_output = np.array(mnn_output).reshape([110, 327])
    # #
    # print("pth_output: \n")
    # print(pth_output)
    # print(pth_output.shape)
    # print("onnx_out: \n")
    # print(onnx_output)
    # print(onnx_output.shape)
    # print("mnn_out: \n")
    # print(mnn_output)
    # print(mnn_output.shape)
