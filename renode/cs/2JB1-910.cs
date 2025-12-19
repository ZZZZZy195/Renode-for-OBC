using System;
using System.Collections.Generic;
using Antmicro.Renode.Core;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Core.Structure.Registers;
using Antmicro.Renode.Peripherals.GPIOPort;

namespace Antmicro.Renode.Peripherals.GPIOPort
{
    [AllowedTranslations(AllowedTranslation.WordToDoubleWord)]
    public class Relay2JB1910 : BaseGPIOPort, IDoubleWordPeripheral
    {
        public Relay2JB1910(IMachine machine) : base(machine, NumberOfPins)
        {
            // 初始化寄存器系统
            registers = CreateRegisters();
            // 初始化引脚状态
            Reset();
        }

        public override void Reset()
        {
            base.Reset();
            registers.Reset();
            // 初始化所有继电器为关闭状态
            UpdateRelays(0);
        }

        public uint ReadDoubleWord(long offset)
        {
            return registers.Read(offset);
        }

        public void WriteDoubleWord(long offset, uint value)
        {
            registers.Write(offset, value);
        }

        private void UpdateRelays(uint state)
        {
            // 解析寄存器值
            var relayAState = (state & 0x1) != 0;
            var relayBState = (state & 0x2) != 0;

            // 设置GPIO输出（使用BaseGPIOPort的Connections数组）
            Connections[0].Set(relayAState);  // ON_A
            Connections[1].Set(!relayAState); // OFF_A (反向)
            Connections[2].Set(relayBState);  // ON_B
            Connections[3].Set(!relayBState); // OFF_B (反向)

            // 输出调试信息
            this.Log(LogLevel.Debug, "继电器状态更新：A={0}, B={1}", relayAState ? "开启" : "关闭", relayBState ? "开启" : "关闭");
        }

        private DoubleWordRegisterCollection CreateRegisters()
        {
            var registersMap = new Dictionary<long, DoubleWordRegister>
            {
                {(long)Registers.Control, new DoubleWordRegister(this)
                    // 控制寄存器（32位全字段）
                    .WithValueField(
                        position: 0,     // 起始位
                        width: 32,       // 位宽
                        name: "CONTROL",
                        writeCallback: (_, value) => UpdateRelays((uint)value), // ulong转uint
                        valueProviderCallback: _ => GetCurrentState()
                    )
                }
            };
            return new DoubleWordRegisterCollection(this, registersMap);
        }

        private uint GetCurrentState()
        {
            // 从GPIO状态反推寄存器值
            uint state = 0;
            if (Connections[0].IsSet) state |= 0x1;  // A开启状态
            if (Connections[2].IsSet) state |= 0x2;  // B开启状态
            return state;
        }

        // 寄存器集合
        private readonly DoubleWordRegisterCollection registers;

        // 引脚数量定义
        private const int NumberOfPins = 4; // ON_A, OFF_A, ON_B, OFF_B

        // 寄存器偏移量定义
        private enum Registers
        {
            Control = 0x00 // 控制寄存器基地址
        }
    }
}