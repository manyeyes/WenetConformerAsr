// ConsoleApp1, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null
// ConsoleApp1.Program
namespace WenetConformerAsr.Examples
{
    internal static partial class Program
    {
        public static string applicationBase = AppDomain.CurrentDomain.BaseDirectory;
        [STAThread]
        private static void Main()
        {
            test_WenetConformerAsrOfflineRecognizer();
            test_WenetConformerAsrOnlineRecognizer();
        }
    }
}