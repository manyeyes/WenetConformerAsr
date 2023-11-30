// See https://github.com/manyeyes for more information
// Copyright (c)  2023 by manyeyes
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WenetConformerAsr.Model;
using WenetConformerAsr.Utils;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using System.Text.RegularExpressions;
using System.IO;

namespace WenetConformerAsr
{
    /// <summary>
    /// offline recognizer package
    /// Copyright (c)  2023 by manyeyes
    /// </summary>
    public class OnlineRecognizer
    {
        private readonly ILogger<OnlineRecognizer> _logger;
        private string[] _tokens;
        private IAsrProj _asrProj;

        public OnlineRecognizer(string encoderFilePath, string decoderFilePath, string ctcFilePath, string tokensFilePath, string configFilePath = "", int threadsNum = 1)
        {
            AsrModel asrModel = new AsrModel(encoderFilePath, decoderFilePath, ctcFilePath, configFilePath: configFilePath, threadsNum: threadsNum);
            _tokens = File.ReadAllLines(tokensFilePath);
            _asrProj=new AsrProjOfConformer(asrModel);
            ILoggerFactory loggerFactory = new LoggerFactory();
            _logger = new Logger<OnlineRecognizer>(loggerFactory);
        }

        public OnlineStream CreateOnlineStream()
        {
            OnlineStream onlineStream = new OnlineStream(_asrProj);
            return onlineStream;
        }

        public List<OnlineRecognizerResultEntity> GetResults(List<OnlineStream> streams)
        {
            this._logger.LogInformation("get features begin");
            this.Forward(streams);
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = this.DecodeMulti(streams);

            return onlineRecognizerResultEntities;
        }

        private void Forward(List<OnlineStream> streams)
        {
            if (streams.Count == 0)
            {
                return;
            }
            List<OnlineStream> streamsWorking = new List<OnlineStream>();
            int contextSize = 2;//_onlineModel.CustomMetadata.Context_size;
            List<AsrInputEntity> modelInputs = new List<AsrInputEntity>();
            List<List<float[]>> statesList = new List<List<float[]>>();
            List<Int64[]> hypList = new List<Int64[]>();
            //List<Int64>[] tokens = new List<Int64>[batchSize];
            //Int64[] hyps = new Int64[_context_size * batchSize];
            List<List<Int64>> tokens = new List<List<Int64>>();
            int padFrameNum = _asrProj.ChunkLength;
            int shiftFrameNum = _asrProj.ShiftLength;
            List<OnlineStream> streamsTemp = new List<OnlineStream>();
            foreach (OnlineStream stream in streams)
            {
                AsrInputEntity asrInputEntity = new AsrInputEntity();

                asrInputEntity.Speech = stream.GetDecodeChunk(padFrameNum);
                if (asrInputEntity.Speech == null)
                {
                    streamsTemp.Add(stream);
                    continue;
                }
                asrInputEntity.SpeechLength = asrInputEntity.Speech.Length;
                modelInputs.Add(asrInputEntity);
                stream.RemoveChunk(shiftFrameNum);
                hypList.Add(stream.Hyp);
                statesList.Add(stream.States);
                tokens.Add(stream.Tokens);
                streamsWorking.Add(stream);
            }
            if (modelInputs.Count == 0)
            {
                return;
            }
            foreach (OnlineStream stream in streamsTemp)
            {
                streams.Remove(stream);
            }
            try
            {
                int batchSize = modelInputs.Count;
                int offset = streams[0].Offset;
                List<float[]> stackStatesList = new List<float[]>();
                stackStatesList = _asrProj.stack_states(statesList);
                EncoderOutputEntity encoderOutputEntity = _asrProj.EncoderProj(modelInputs, stackStatesList, offset);
                CtcOutputEntity ctcOutputEntity = _asrProj.CtcProj(encoderOutputEntity);
                //TODO
                //DecoderOutputEntity decoderOutputEntity = _asrProj.DecoderProj(encoderOutputEntity, ctcOutputEntity);
                List<List<float[]>> next_statesList = new List<List<float[]>>();
                next_statesList = _asrProj.unstack_states(encoderOutputEntity.StatesList);
                int streamIndex = 0;
                foreach (OnlineStream stream in streamsWorking)
                {
                    stream.Tokens.AddRange(ctcOutputEntity.Hyps[streamIndex].ToList());
                    stream.States = next_statesList[streamIndex];
                    stream.Offset = offset + encoderOutputEntity.Index;
                    streamIndex++;
                }
            }
            catch (Exception ex)
            {
                //
            }
        }

        private List<OnlineRecognizerResultEntity> DecodeMulti(List<OnlineStream> streams)
        {
            List<OnlineRecognizerResultEntity> onlineRecognizerResultEntities = new List<OnlineRecognizerResultEntity>();
#pragma warning disable CS8602 // 解引用可能出现空引用。
            foreach (OnlineStream stream in streams)
            {
                List<Int64> token_num = stream.Tokens;
                string text_result = "";
                foreach (Int64 token in token_num)
                {
                    if (token == 2)
                    {
                        break;
                    }
                    if (_tokens[token].Split(' ')[0] != "</s>" && _tokens[token].Split(' ')[0] != "<s>" && _tokens[token].Split(' ')[0] != "<sos/eos>" && _tokens[token].Split(' ')[0] != "<blank>" && _tokens[token].Split(' ')[0] != "<unk>")
                    {
                        if (IsChinese(_tokens[token].Split(' ')[0], true))
                        {
                            text_result += _tokens[token].Split(' ')[0];
                        }
                        else
                        {
                            text_result += "▁" + _tokens[token].Split(' ')[0] + "▁";
                        }
                    }
                }
                OnlineRecognizerResultEntity onlineRecognizerResultEntity = new OnlineRecognizerResultEntity();
                onlineRecognizerResultEntity.text = text_result.Replace("@@▁▁", "").Replace("@@▁", "").Replace("▁▁▁", " ").Replace("▁▁", " ").Replace("▁", "").ToLower();
                onlineRecognizerResultEntities.Add(onlineRecognizerResultEntity);
            }
#pragma warning restore CS8602 // 解引用可能出现空引用。

            return onlineRecognizerResultEntities;
        }

        /// <summary>
        /// Verify if the string is in Chinese.
        /// </summary>
        /// <param name="checkedStr">The string to be verified.</param>
        /// <param name="allMatch">Is it an exact match. When the value is true,all are in Chinese; 
        /// When the value is false, only Chinese is included.
        /// </param>
        /// <returns></returns>
        private bool IsChinese(string checkedStr, bool allMatch)
        {
            string pattern;
            if (allMatch)
                pattern = @"^[\u4e00-\u9fa5]+$";
            else
                pattern = @"[\u4e00-\u9fa5]";
            if (Regex.IsMatch(checkedStr, pattern))
                return true;
            else
                return false;
        }
    }
}