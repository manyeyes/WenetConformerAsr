using WenetConformerAsr.Examples.Utils;

namespace WenetConformerAsr.Examples
{
    internal static partial class Program
    {
        public static WenetConformerAsr.OfflineRecognizer initWenetConformerAsrOfflineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.quant.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.quant.onnx";
            string ctcFilePath = applicationBase + "./" + modelName + "/ctc.quant.onnx";
            //string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
            string tokensFilePath = applicationBase + "./" + modelName + "/units.txt";
            WenetConformerAsr.OfflineRecognizer offlineRecognizer = new WenetConformerAsr.OfflineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath, threadsNum: 1);
            return offlineRecognizer;
        }

        public static void test_WenetConformerAsrOfflineRecognizer(List<float[]>? samples = null)
        {
            //string modelName = "wenet_onnx_aishell_u2pp_conformer_20211025_offline";
            string modelName = "wenet_onnx_gigaspeech_u2pp_conformer_20210728_offline";
            //string modelName = "wenet_onnx_wenetspeech_u2pp_conformer_20220506_offline";
            //string modelName = "wenet_onnx_aishell_u2pp_conformer_20210601_offline";
            //string modelName = "wenet_onnx_aishell2_u2pp_conformer_20210618_offline";
            WenetConformerAsr.OfflineRecognizer offlineRecognizer = initWenetConformerAsrOfflineRecognizer(modelName);
            TimeSpan total_duration = new TimeSpan(0L);
            List<List<float[]>> samplesList = new List<List<float[]>>();
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int i = 0; i < 4; i++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", i.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    // method 1
                    //TimeSpan duration = TimeSpan.Zero;
                    //float[] sample = SpeechProcessing.AudioHelper.GetFileSample(wavFilePath, ref duration);
                    //samples.Add(sample);
                    //total_duration += duration;
                    //method 2
                    TimeSpan duration = TimeSpan.Zero;
                    samples = AudioHelper.GetFileChunkSamples(wavFilePath, ref duration);
                    samplesList.Add(samples);
                    total_duration += duration;
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            TimeSpan start_time = new TimeSpan(DateTime.Now.Ticks);
            List<WenetConformerAsr.OfflineStream> streams = new List<WenetConformerAsr.OfflineStream>();
            // method 1
            //foreach (var sample in samples)
            //{
            //    OfflineStream stream = offlineRecognizer.CreateOfflineStream();
            //    stream.AddSamples(sample);
            //    streams.Add(stream);
            //}
            // method 2
            foreach (List<float[]> samplesListItem in samplesList)
            {
                WenetConformerAsr.OfflineStream stream = offlineRecognizer.CreateOfflineStream();
                foreach (float[] sample in samplesListItem)
                {
                    stream.AddSamples(sample);
                }
                streams.Add(stream);
            }
            // decode,fit batch=1
            foreach (WenetConformerAsr.OfflineStream stream in streams)
            {
                WenetConformerAsr.Model.OfflineRecognizerResultEntity result = offlineRecognizer.GetResult(stream);
                Console.WriteLine(result.Text);
                Console.WriteLine("");
            }
            //fit batch>1,but all in one
            //List<WenetConformerAsr.Model.OfflineRecognizerResultEntity> results_batch = offlineRecognizer.GetResults(streams);
            //foreach (WenetConformerAsr.Model.OfflineRecognizerResultEntity result in results_batch)
            //{
            //    Console.WriteLine(result.Text);
            //    Console.WriteLine("");
            //}
            TimeSpan end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("Hello, World!");
        }

        public static WenetConformerAsr.OnlineRecognizer initWenetConformerAsrOnlineRecognizer(string modelName)
        {
            string encoderFilePath = applicationBase + "./" + modelName + "/encoder.quant.onnx";
            string decoderFilePath = applicationBase + "./" + modelName + "/decoder.quant.onnx";
            string ctcFilePath = applicationBase + "./" + modelName + "/ctc.quant.onnx";
            //string configFilePath = applicationBase + "./" + modelName + "/asr.yaml";
            string tokensFilePath = applicationBase + "./" + modelName + "/units.txt";
            WenetConformerAsr.OnlineRecognizer onlineRecognizer = new WenetConformerAsr.OnlineRecognizer(encoderFilePath, decoderFilePath, ctcFilePath, tokensFilePath);
            return onlineRecognizer;
        }

        public static void test_WenetConformerAsrOnlineRecognizer(List<float[]>? samples = null)
        {
            //string modelName = "wenet_onnx_aishell_u2pp_conformer_20211025_online";
            //string modelName = "wenet_onnx_gigaspeech_u2pp_conformer_20210728_online";
            string modelName = "wenet_onnx_wenetspeech_u2pp_conformer_20220506_online";
            //string modelName = "wenet_onnx_aishell_u2pp_conformer_20210601_online";
            //string modelName = "wenet_onnx_aishell2_u2pp_conformer_20210618_online";
            WenetConformerAsr.OnlineRecognizer onlineRecognizer = initWenetConformerAsrOnlineRecognizer(modelName);
            TimeSpan total_duration = TimeSpan.Zero;
            TimeSpan start_time = TimeSpan.Zero;
            TimeSpan end_time = TimeSpan.Zero;


            List<List<float[]>> samplesList = new List<List<float[]>>();
            int batchSize = 1;
            int startIndex = 2;
            if (samples == null)
            {
                samples = new List<float[]>();
                for (int n = startIndex; n < startIndex + batchSize; n++)
                {
                    string wavFilePath = string.Format(applicationBase + "./" + modelName + "/test_wavs/{0}.wav", n.ToString());
                    if (!File.Exists(wavFilePath))
                    {
                        continue;
                    }
                    // method 1
                    TimeSpan duration = TimeSpan.Zero;
                    samples = Utils.AudioHelper.GetFileChunkSamples(wavFilePath, ref duration, chunkSize: 160 * 6);
                    for (int j = 0; j < 30; j++)
                    {
                        samples.Add(new float[400]);
                    }
                    samplesList.Add(samples);
                    total_duration += duration;
                    // method 2
                    //List<TimeSpan> durations = new List<TimeSpan>();
                    //samples = SpeechProcessing.AudioHelper.GetMediaChunkSamples(wavFilePath, ref durations);
                    //samplesList.Add(samples);
                    //foreach(TimeSpan duration in durations)
                    //{
                    //    total_duration += duration;
                    //}
                }
            }
            else
            {
                samplesList.Add(samples);
            }
            start_time = new TimeSpan(DateTime.Now.Ticks);
            // one stream decode
            //for (int j = 0; j < samplesList.Count; j++)
            //{
            //    K2TransducerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();//new K2TransducerAsr.OnlineStream(16000,80);
            //    foreach (float[] samplesItem in samplesList[j])
            //    {
            //        stream.AddSamples(samplesItem);
            //    }
            //    // 1
            //    int w = 0;
            //    while (w < 17)
            //    {
            //        OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
            //        Console.WriteLine(result_on.text);
            //        w++;
            //    }
            //    // 2
            //    //OnlineRecognizerResultEntity result_on = onlineRecognizer.GetResult(stream);
            //    //Console.WriteLine(result_on.text);
            //}

            //multi streams decode
            List<WenetConformerAsr.OnlineStream> onlineStreams = new List<WenetConformerAsr.OnlineStream>();
            List<bool> isEndpoints = new List<bool>();
            List<bool> isEnds = new List<bool>();
            for (int num = 0; num < samplesList.Count; num++)
            {
                WenetConformerAsr.OnlineStream stream = onlineRecognizer.CreateOnlineStream();
                onlineStreams.Add(stream);
                isEndpoints.Add(false);
                isEnds.Add(false);
            }
            int i = 0;
            List<WenetConformerAsr.OnlineStream> streams = new List<WenetConformerAsr.OnlineStream>();

            while (true)
            {
                streams = new List<WenetConformerAsr.OnlineStream>();

                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (samplesList[j].Count > i && samplesList.Count > j)
                    {
                        onlineStreams[j].AddSamples(samplesList[j][i]);
                        streams.Add(onlineStreams[j]);
                        isEndpoints[0] = false;
                    }
                    else
                    {
                        streams.Add(onlineStreams[j]);
                        samplesList.Remove(samplesList[j]);
                        isEndpoints[0] = true;
                    }
                }
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (isEndpoints[j])
                    {
                        if (onlineStreams[j].IsFinished(isEndpoints[j]))
                        {
                            isEnds[j] = true;
                        }
                        else
                        {
                            streams.Add(onlineStreams[j]);
                        }
                    }
                }
                List<WenetConformerAsr.OnlineRecognizerResultEntity> results_batch = onlineRecognizer.GetResults(streams);
                foreach (WenetConformerAsr.OnlineRecognizerResultEntity result in results_batch)
                {
                    Console.WriteLine(result.text);
                    //Console.WriteLine("");
                }
                Console.WriteLine("");
                i++;
                bool isAllFinish = true;
                for (int j = 0; j < samplesList.Count; j++)
                {
                    if (!isEnds[j])
                    {
                        isAllFinish = false;
                        break;
                    }
                }
                if (isAllFinish)
                {
                    break;
                }
            }
            end_time = new TimeSpan(DateTime.Now.Ticks);
            double elapsed_milliseconds = end_time.TotalMilliseconds - start_time.TotalMilliseconds;
            double rtf = elapsed_milliseconds / total_duration.TotalMilliseconds;
            Console.WriteLine("elapsed_milliseconds:{0}", elapsed_milliseconds.ToString());
            Console.WriteLine("total_duration:{0}", total_duration.TotalMilliseconds.ToString());
            Console.WriteLine("rtf:{1}", "0".ToString(), rtf.ToString());
            Console.WriteLine("Hello, World!");
        }
    }
}
