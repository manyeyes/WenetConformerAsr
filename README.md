# WenetConformerAsr
A C# library for decoding the Wenet ASR onnx model

这是一个用于解码 wenet asr onnx 模型的c#库，使用c#编写，基于.net6.0，支持在多平台（包括windows、linux、android、macos、ios等）编译、调用。

可以使用maui或uno迅速构建可在多平台运行的应用程序。
##### 项目中WenetConformerAsr和WenetConformerAsr2的区别:
1.共同之处：

功能一样，调用的方式一样，都支持streaming和non-streaming模型的解码。

2.不同之处：

| library  | streaming和non-streaming模型加载模块  |模型和扩展|
| ------------ | ------------ |------------|
| WenetConformerAsr  | 合二为一  |1.加载wenet官方导出的onnx模型，代码简洁|
| WenetConformerAsr2  |各自独立   |1.加载wenet官方导出的onnx模型。2.便于扩展，如果自己导出的streaming和non-streaming onnx模型配置参数不尽相同，可以在各自的模块上进行调整而互不影响|

如果没有二次开发的需求，想要直接使用wenet官导onnx模型，推荐使用WenetConformerAsr.

##### 模型的下载
待续

##### 模型的使用
引用
```csharp
using WenetConformerAsr;
```
调用(use non-streaming onnx model decoding)
```csharp
//load model
string modelName = "wenet_onnx_wenetspeech_u2pp_conformer_20220506_offline";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.quant.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.quant.onnx";
string ctcFilePath = applicationBase + "./" + modelName + "/ctc.quant.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/units.txt";
WenetConformerAsr.OfflineRecognizer offlineRecognizer = new WenetConformerAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath);
//这里省略音频文件到sample的转换，具体可以参考examples中的test_WenetConformerAsrOfflineRecognizer
WenetConformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
stream.AddSamples(sample);
WenetConformerAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
Console.WriteLine(result.Text);
```

调用(use streaming onnx model decoding)
```csharp
//load model
string modelName = "wenet_onnx_wenetspeech_u2pp_conformer_20220506_online";
string encoderFilePath = applicationBase + "./" + modelName + "/encoder.quant.onnx";
string decoderFilePath = applicationBase + "./" + modelName + "/decoder.quant.onnx";
string ctcFilePath = applicationBase + "./" + modelName + "/ctc.quant.onnx";
string tokensFilePath = applicationBase + "./" + modelName + "/units.txt";
WenetConformerAsr.OnlineRecognizer onlineRecognizer = new WenetConformerAsr.OnlineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath);
//这里省略音频文件到sample的转换，或者来自于麦克风，具体如何做，可以参考examples中的test_WenetConformerAsrOnlineRecognizer
WenetConformerAsr.OnlineStream stream = offlineRecognizer.CreateOfflineStream();
while (true)
{
    //这是一个简单的解码示意，如需了解更详细周密的流程，请参考examples
	//sample=来自音频文件或麦克风
    stream.AddSamples(sample);
    WenetConformerAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
    Console.WriteLine(result.Text);
}
```
