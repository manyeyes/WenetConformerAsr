using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using WenetConformerAsr.Model;
using WenetConformerAsr.Utils;
using System.Diagnostics;

namespace WenetConformerAsr
{
    internal class AsrProjOfConformer : IAsrProj
    {
        private InferenceSession _encoderSession;
        private InferenceSession _decoderSession;
        private InferenceSession _ctcSession;
        private CustomMetadata _customMetadata;
        private int _blank_id = 0;
        private int _sos_eos_id = 1;
        private int _unk_id = 2;

        private int _featureDim = 80;
        private int _sampleRate = 16000;
        private int _chunkLength = 0;
        private int _shiftLength = 0;
        private int _required_cache_size = 0;
        public AsrProjOfConformer(AsrModel asrModel)
        {
            _encoderSession = asrModel.EncoderSession;
            _decoderSession = asrModel.DecoderSession;
            _ctcSession = asrModel.CtcSession;
            _blank_id = asrModel.Blank_id;
            _sos_eos_id = asrModel.Sos_eos_id;
            _unk_id = asrModel.Unk_id;
            _featureDim = asrModel.FeatureDim;
            _sampleRate = asrModel.SampleRate;
            _customMetadata = asrModel.CustomMetadata;
            _chunkLength = asrModel.ChunkLength;
            _shiftLength = asrModel.ShiftLength;
            _required_cache_size = asrModel.Required_cache_size;
        }

        public InferenceSession EncoderSession { get => _encoderSession; set => _encoderSession = value; }
        public InferenceSession DecoderSession { get => _decoderSession; set => _decoderSession = value; }
        public InferenceSession CtcSession { get => _ctcSession; set => _ctcSession = value; }
        public CustomMetadata CustomMetadata { get => _customMetadata; set => _customMetadata = value; }
        public int Blank_id { get => _blank_id; set => _blank_id = value; }
        public int Sos_eos_id { get => _sos_eos_id; set => _sos_eos_id = value; }
        public int Unk_id { get => _unk_id; set => _unk_id = value; }
        public int ChunkLength { get => _chunkLength; set => _chunkLength = value; }
        public int ShiftLength { get => _shiftLength; set => _shiftLength = value; }
        public int FeatureDim { get => _featureDim; set => _featureDim = value; }
        public int SampleRate { get => _sampleRate; set => _sampleRate = value; }
        public int Required_cache_size { get => _required_cache_size; set => _required_cache_size = value; }

        public List<float[]> GetEncoderInitStates(int batchSize = 1)
        {
            List<float[]> statesList = new List<float[]>();
            //计算尺寸
            int required_cache_size = _required_cache_size < 0 ? 0 : _required_cache_size;
            float[] att_cache = new float[_customMetadata.Num_blocks * _customMetadata.Head * required_cache_size * (_customMetadata.Output_size / _customMetadata.Head * 2)];
            float[] cnn_cache = new float[_customMetadata.Num_blocks * 1 * _customMetadata.Output_size * (_customMetadata.Cnn_module_kernel - 1)];
            statesList.Add(att_cache);
            statesList.Add(cnn_cache);
            return statesList;
        }

        private float[] InitCacheFeats(int batchSize = 1)
        {

            int cached_feature_size = 0;//1 + _right_context - _subsampling_rate;//TODO temp test
            float[] cacheFeats = new float[batchSize * cached_feature_size * 80];
            return cacheFeats;
        }

        public List<float[]> stack_states(List<List<float[]>> statesList)
        {
            List<float[]> states = new List<float[]>();
            states = statesList[0];
            return states;
        }
        public List<List<float[]>> unstack_states(List<float[]> states)
        {
            List<List<float[]>> statesList = new List<List<float[]>>();
            Debug.Assert(states.Count % 2 == 0, "when stack_states, state_list[0] is 2x");
            statesList.Add(states);
            return statesList;
        }

        public EncoderOutputEntity EncoderProj(List<AsrInputEntity> modelInputs, List<float[]> statesList, int offset)
        {
            float[] padSequence = PadHelper.PadSequence(modelInputs);
            var inputMeta = _encoderSession.InputMetadata;
            EncoderOutputEntity encoderOutput = new EncoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputNames = new List<string>();
            var inputValues = new List<FixedBufferOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "chunk")
                {
                    int[] dim = new int[] { 1, padSequence.Length / _customMetadata.Feature_size, _customMetadata.Feature_size };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "offset")
                {
                    int[] dim = new int[] { 1 };
                    Int64[] offset_tensor = new Int64[] { offset };
                    var tensor = new DenseTensor<Int64>(offset_tensor, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
                if (name == "required_cache_size")
                {
                    int required_cache_size = _required_cache_size;
                    int[] dim = new int[] { 1 };
                    Int64[] required_cache_size_tensor = new Int64[] { required_cache_size };
                    var tensor = new DenseTensor<Int64>(required_cache_size_tensor, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor(name, tensor));
                }
                if (name == "att_cache")
                {
                    float[] att_cache = statesList[0];
                    int required_cache_size = att_cache.Length / _customMetadata.Num_blocks / _customMetadata.Head / (_customMetadata.Output_size / _customMetadata.Head * 2);
                    int[] dim = new int[] { _customMetadata.Num_blocks, _customMetadata.Head, required_cache_size, _customMetadata.Output_size / _customMetadata.Head * 2 };
                    var tensor = new DenseTensor<float>(att_cache, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "cnn_cache")
                {
                    float[] cnn_cache = statesList[1];
                    int[] dim = new int[] { _customMetadata.Num_blocks, 1, _customMetadata.Output_size, _customMetadata.Cnn_module_kernel - 1 };
                    var tensor = new DenseTensor<float>(cnn_cache, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
                if (name == "att_mask")
                {
                    int required_cache_size = _required_cache_size;
                    int[] dim = new int[] { required_cache_size + _customMetadata.Chunk_size, 1 };
                    bool[] att_mask = new bool[(required_cache_size + _customMetadata.Chunk_size) * 1];
                    att_mask = att_mask.Select(x => x = true).ToArray();
                    var tensor = new DenseTensor<bool>(att_mask, dim, false);
                    if (_customMetadata.Left_chunks > 0)
                    {
                        dim = new int[] { 1, 1, required_cache_size + _customMetadata.Chunk_size };
                        att_mask = new bool[1 * 1 * (required_cache_size + _customMetadata.Chunk_size)];
                        att_mask = att_mask.Select(x => x = true).ToArray();
                        int chunk_idx = offset / _customMetadata.Chunk_size - _customMetadata.Left_chunks;
                        if (chunk_idx < _customMetadata.Left_chunks)
                        {
                            for (int i = 0; i < (_customMetadata.Left_chunks - chunk_idx) * _customMetadata.Chunk_size; ++i)
                            {
                                att_mask[i] = false;
                            }
                        }
                        tensor = new DenseTensor<bool>(att_mask, dim, false);
                    }
                    container.Add(NamedOnnxValue.CreateFromTensor<bool>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> encoderResults = null;
                encoderResults = _encoderSession.Run(container);

                if (encoderResults != null)
                {
                    var encoderResultsArray = encoderResults.ToArray();
                    var outputTensor = encoderResultsArray[0].AsTensor<float>();
                    encoderOutput.Index = outputTensor.Dimensions[1];
                    encoderOutput.Output = outputTensor.ToArray();
                    var rAttCacheTensor = encoderResultsArray[1].AsTensor<float>();
                    encoderOutput.R_att_cache = rAttCacheTensor.ToArray();
                    var rCnnCache = encoderResultsArray[2].AsTensor<float>();
                    encoderOutput.R_cnn_cache = rCnnCache.ToArray();
                    List<float[]> next_statesList = new List<float[]>();
                    foreach (var item in encoderResultsArray)
                    {
                        if (item.Name.EndsWith("_cache"))
                        {
                            next_statesList.Add(item.AsEnumerable<float>().ToArray());
                        }
                    }
                    encoderOutput.StatesList = next_statesList;
                }
            }
            catch (Exception ex)
            {
                //
            }
            return encoderOutput;
        }

        public CtcOutputEntity CtcProj(EncoderOutputEntity encoderOutput)
        {
            CustomMetadata customMetadata = _customMetadata;
            float[] padSequence = encoderOutput.Output;
            var inputMeta = _ctcSession.InputMetadata;
            CtcOutputEntity ctcOutput = new CtcOutputEntity();
            var container = new List<NamedOnnxValue>();
            foreach (var name in inputMeta.Keys)
            {
                if (name == "hidden")
                {
                    int[] dim = new int[] { 1, padSequence.Length / customMetadata.Output_size, customMetadata.Output_size };
                    var tensor = new DenseTensor<float>(padSequence, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }
            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> ctcResults = null;
                ctcResults = _ctcSession.Run(container);

                if (ctcResults != null)
                {
                    var ctcResultsArray = ctcResults.ToArray();
                    var probsTensor = ctcResultsArray[0].AsTensor<float>();
                    ctcOutput.Probs = probsTensor.ToArray();
                    List<Int64[]> token_nums = new List<Int64[]> { };
                    List<Int64> token_len = new List<Int64>();

                    for (int i = 0; i < probsTensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[probsTensor.Dimensions[1]];
                        for (int j = 0; j < probsTensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < probsTensor.Dimensions[2]; k++)
                            {
                                token_num = probsTensor[i, j, token_num] > probsTensor[i, j, k] ? token_num : k;
                            }
                            item[j] = token_num;
                        }
                        token_nums.Add(item);
                        token_len.Add(item.Length);
                    }
                    ctcOutput.Hyps = token_nums;
                    ctcOutput.Hyps_lens = token_len;
                }
            }
            catch (Exception ex)
            {
                //
            }
            return ctcOutput;
        }

        public DecoderOutputEntity DecoderProj(EncoderOutputEntity encoderOutputEntity, CtcOutputEntity ctcOutputEntity, int batchSize = 1)
        {
            CustomMetadata customMetadata = _customMetadata;
            DecoderOutputEntity decoderOutputEntity = new DecoderOutputEntity();
            var container = new List<NamedOnnxValue>();
            var inputMeta = _decoderSession.InputMetadata;
            foreach (var name in inputMeta.Keys)
            {
                if (name == "hyps")
                {
                    int[] dim = new int[] { batchSize, ctcOutputEntity.Hyps[0].Length };
                    var tensor = new DenseTensor<Int64>(ctcOutputEntity.Hyps[0], dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
                if (name == "hyps_lens")
                {
                    int[] dim = new int[] { batchSize };
                    var tensor = new DenseTensor<Int64>(ctcOutputEntity.Hyps_lens.ToArray(), dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<Int64>(name, tensor));
                }
                if (name == "encoder_out")
                {
                    int[] dim = new int[] { batchSize, encoderOutputEntity.Output.Length / customMetadata.Output_size / batchSize, customMetadata.Output_size };
                    var tensor = new DenseTensor<float>(encoderOutputEntity.Output, dim, false);
                    container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                }
            }

            try
            {
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderResults = null;
                decoderResults = _decoderSession.Run(container);

                if (decoderResults != null)
                {
                    var decoderResultsArray = decoderResults.ToArray();
                    Tensor<float> logits_tensor = decoderResultsArray[0].AsTensor<float>();
                    List<Int64[]> token_nums = new List<Int64[]> { };

                    for (int i = 0; i < logits_tensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[logits_tensor.Dimensions[1]];
                        for (int j = 0; j < logits_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < logits_tensor.Dimensions[2]; k++)
                            {
                                token_num = logits_tensor[i, j, token_num] > logits_tensor[i, j, k] ? token_num : k;
                            }
                            item[j] = (int)token_num;
                        }
                        token_nums.Add(item);
                    }

                    Tensor<float> r_score_tensor = decoderResultsArray[1].AsTensor<float>();
                    List<Int64[]> r_token_nums = new List<Int64[]> { };

                    for (int i = 0; i < r_score_tensor.Dimensions[0]; i++)
                    {
                        Int64[] item = new Int64[r_score_tensor.Dimensions[1]];
                        for (int j = 0; j < r_score_tensor.Dimensions[1]; j++)
                        {
                            int token_num = 0;
                            for (int k = 1; k < r_score_tensor.Dimensions[2]; k++)
                            {
                                token_num = r_score_tensor[i, j, token_num] > r_score_tensor[i, j, k] ? token_num : k;
                            }
                            item[j] = (int)token_num;
                        }
                        r_token_nums.Add(item);
                    }

                    decoderOutputEntity.Logits = logits_tensor.ToArray();
                    decoderOutputEntity.Sample_ids = r_token_nums;
                }
            }
            catch (Exception ex)
            {
                //
            }
            return decoderOutputEntity;
        }
    }
}
