using System;
using System.Collections.Generic;
using Antmicro.Renode.Core;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Utilities;
using Antmicro.Renode.Exceptions;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Core.Structure.Registers;

namespace Antmicro.Renode.Peripherals.GPIOPort
{
    public class PC104 : BaseGPIOPort, IDoubleWordPeripheral
    {
        // 定义60个引脚
        private const int NumberOfPins = 60;
        private readonly uint[] mode = new uint[NumberOfPins];

        public PC104(IMachine machine, uint modeResetValue = 0) : base(machine, NumberOfPins)
        {
            for (int i = 0; i < NumberOfPins; i++)
            {
                mode[i] = modeResetValue;
            }

            registers = CreateRegisters();
            Reset();
        }

        public uint ReadDoubleWord(long offset)
        {
            return registers.Read(offset);
        }

        public void WriteDoubleWord(long offset, uint value)
        {
            registers.Write(offset, value);
        }

        // 重置寄存器和引脚状态
        public override void Reset()
        {
            base.Reset();
            registers.Reset();

            for (var i = 0; i < NumberOfPins; i++)
            {
                ChangeMode(i, (Mode)BitHelper.GetValue(0, 2 * i, 2));  // 默认模式
            }
        }

        // 设置引脚模式
        private void ChangeMode(int index, Mode mode)
        {
            this.mode[index] = (uint)mode;
        }

        // 创建寄存器
        private DoubleWordRegisterCollection CreateRegisters()
        {
            var registersMap = new Dictionary<long, DoubleWordRegister>
            {
                {(long)Registers.Mode0, new DoubleWordRegister(this)
                    .WithEnumFields<DoubleWordRegister, Mode>(0, 2, 16, name: "MODER0",
                        valueProviderCallback: (idx, _) => (Mode)mode[idx], // 显式转换 uint 为 Mode
                        writeCallback: (idx, _, val) => ChangeMode(idx, (Mode)val)) // 同样的显式转换
                },
                {(long)Registers.Mode1, new DoubleWordRegister(this)
                    .WithEnumFields<DoubleWordRegister, Mode>(0, 2, 16, name: "MODER1",
                        valueProviderCallback: (idx, _) => (Mode)mode[16 + idx],
                        writeCallback: (idx, _, val) => ChangeMode(16 + idx, (Mode)val))
                },
                {(long)Registers.Mode2, new DoubleWordRegister(this)
                    .WithEnumFields<DoubleWordRegister, Mode>(0, 2, 16, name: "MODER2",
                        valueProviderCallback: (idx, _) => (Mode)mode[32 + idx],
                        writeCallback: (idx, _, val) => ChangeMode(32 + idx, (Mode)val))
                },
                {(long)Registers.Mode3, new DoubleWordRegister(this)
                    .WithEnumFields<DoubleWordRegister, Mode>(0, 2, 12, name: "MODER3",
                        valueProviderCallback: (idx, _) => (Mode)mode[48 + idx],
                        writeCallback: (idx, _, val) => ChangeMode(48 + idx, (Mode)val))
                },
            };

            return new DoubleWordRegisterCollection(this, registersMap);
        }

        // 寄存器枚举
        private enum Registers
        {
            Mode0 = 0x00,
            Mode1 = 0x04,
            Mode2 = 0x08,
            Mode3 = 0x0C,
        }

        // 模式枚举
        public enum Mode
        {
            Input = 0x0,
            Output = 0x1,
            AlternateFunction = 0x2,
            Analog = 0x3,
        }

        // 寄存器集合
        private DoubleWordRegisterCollection registers;
    }
}
