using System;
using Antmicro.Renode.Core;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Core.Structure.Registers;
using Antmicro.Renode.Peripherals.GPIOPort;

namespace Antmicro.Renode.Peripherals.Miscellaneous
{
    public class ULN2803A : BaseGPIOPort, IDoubleWordPeripheral
    {
        private const int InputPins = 8;  // 输入引脚数
        private const int TotalPins = 16; // 输入0-7 + 输出8-15

        public ULN2803A(IMachine machine) : base(machine, TotalPins)
        {
            // 初始化所有引脚为输入模式（根据需求调整）
            for (int i = 0; i < TotalPins; i++)
            {
                Connections[i].Set(false); // 默认低电平
            }
            Reset();
        }

        public uint ReadDoubleWord(long offset)
        {
            switch ((Registers)offset)
            {
                case Registers.InputData:
                    return ReadInputData();
                case Registers.OutputData:
                    return outputData;
                case Registers.Control:
                    return control;
                default:
                    this.LogUnhandledRead(offset);
                    return 0;
            }
        }

        public void WriteDoubleWord(long offset, uint value)
        {
            switch ((Registers)offset)
            {
                case Registers.OutputData:
                    outputData = value;
                    UpdateOutputs();
                    break;
                case Registers.Control:
                    control = value;
                    break;
                case Registers.InputData:
                    this.Log(LogLevel.Warning, "Write to InputData ignored: 0x{0:X}", value);
                    break;
                default:
                    this.LogUnhandledWrite(offset, value);
                    break;
            }
        }

        public override void Reset()
        {
            base.Reset();
            outputData = 0;
            control = 0;
            IRQ.Set(false);
            UpdateOutputs();
        }

        // 关键修改：将 protected 改为 public
        public override void OnGPIO(int pin, bool value)
        {
            if (pin < InputPins)
            {
                // 输入引脚变化可在此处理（例如触发中断）
                // this.Log(LogLevel.Info, $"Input pin {pin} changed to {value}");
            }
        }

        private uint ReadInputData()
        {
            uint data = 0;
            for (int i = 0; i < InputPins; i++)
            {
                data |= (Connections[i].IsSet ? 1u : 0u) << i;
            }
            return data;
        }

        private void UpdateOutputs()
        {
            for (int i = 0; i < InputPins; i++)
            {
                bool state = (outputData & (1u << i)) != 0;
                Connections[InputPins + i].Set(state); // 输出引脚从8开始
            }
            IRQ.Set(outputData != 0);
        }

        private uint outputData;
        private uint control;
        public GPIO IRQ { get; } = new GPIO();

        private enum Registers : long
        {
            InputData = 0x00,  // 只读，反映输入引脚0-7状态
            OutputData = 0x04, // 写入设置输出引脚8-15
            Control = 0x08
        }
    }
}