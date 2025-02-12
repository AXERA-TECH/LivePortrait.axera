# LivePortrait.axera

> KwaiVGI LivePortrait DEMO on Axera

- 目前支持 `Python` 语言, `C++` 代码在开发中
- 预编译模型下载 [models](https://github.com/AXERA-TECH/LivePortrait.axera/releases/download/v1.0.0/download_models.tar), 如需自行转换请参考 [模型转换](/model_convert/README.md)
- 预编译模型如果下载失败, 请手动从 [releases 资源](https://github.com/AXERA-TECH/LivePortrait.axera/releases) 下载

## 支持平台

- [x] AX650N
- [ ] AX630C

## Git Clone

```bash
$ git clone git@github.com:AXERA-TECH/LivePortrait.axera.git
$ cd LivePortrait.axera/python
$ ln -s /path/your/hugging_face/models/LivePortrait pretrained_weights
```

⚠️注意, 这里使用软连接构造 `pretrained_weights` 目录, 也可以将 `huggingface` 中的模型直接复制到 `pretrained_weights` 目录中.

如果还没有 `huggingface liveportrait` 模型, 可以通过以下命令下载:

```bash
$ git clone https://huggingface.co/KwaiVGI/LivePortrait
```

## 模型转换

关于 `onnx` 和 `axmodel` 的导出、编译参见 [模型转换](./model_convert/README.md) 部分内容.

## 上板部署

- `AX650N` 的设备已预装 `Ubuntu 22.04`
- 以 `root` 权限登陆 `AX650N` 的板卡设备
- 接入互联网, 确保 `AX650N` 的设备能正常执行 `apt install`, `pip install` 等指令
- 已验证设备: `AX650N DEMO Board`、`爱芯派Pro(AX650N)`

### Python API 运行

#### Requirements

```bash
$ mkdir /opt/site-packages
$ cd python
$ pip3 install -r requirements.txt --prefix=/opt/site-packages
``` 

#### 添加环境变量

将以下两行添加到 `/root/.bashrc`(实际添加的路径需要自行检查)后, 重新连接终端或者执行 `source ~/.bashrc`

```bash
$ export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages  
$ export PATH=$PATH:/opt/site-packages/local/bin
``` 

#### 基于 ONNX-Runtime 运行

此 `Demo` 可在 `Axera 开发板` 或 `PC` 上运行 , 示例命令如下:
  
```bash
$ python3 python/infer_onnx.py --source ./assets/examples/source/s0.jpg --driving ./assets/examples/driving/d8.jpg --models python/onnx-models --output-dir onnx_infer
```
其中 `--models` 指定 `*.onnx` 模型的存储路径.

`onnx-infer` 输出结果如下:

![output_concat](assets/examples/result/s0--d8_concat.jpg)
![output](assets/examples/result/s0--d8.jpg)

#### 基于 AXEngine 运行

在 `Axera 开发板` 上运行以下命令:

```sh
$ cd LivePortrait.axera
$ python3 ./python/infer.py --source ./assets/examples/source/s0.jpg --driving ./assets/examples/driving/d8.jpg --models ./python/axmodels/ --output-dir ./axmodel_infer
```  

其中 `--models` 指定 `*.axmodel` 模型的存储路径.

`axmodel-infer` 输出结果如下:

![output_concat](assets/examples/result/s0--d8_concat_axmodel.jpg)
![output](assets/examples/result/s0--d8_axmodel.jpg)

## 技术讨论

- Github issues
- QQ 群: 139953715
