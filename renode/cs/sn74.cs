using System;
using System.Collections.Generic;
using Antmicro.Renode.Core;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Peripherals.GPIOPort;
using Antmicro.Renode.Exceptions;
using Antmicro.Renode.Core.Structure.Registers;

namespace Antmicro.Renode.Peripherals.GPIOPort
{
    [AllowedTranslations(AllowedTranslation.WordToDoubleWord)]
    public class SN74AHC273 : BaseGPIOPort, IDoubleWordPeripheral, ILocalGPIOReceiver
    {
        private readonly bool[] inputState = new bool[8];  // D0-D7 输入状态
        private readonly bool[] outputState = new bool[8]; // Q0-Q7 输出状态
        private bool prevLatchEnable = false;
        private readonly DoubleWordRegisterCollection registers;
        private const int NumberOfPins = 16;

        public SN74AHC273(IMachine machine) : base(machine, NumberOfPins)
        {
            registers = CreateRegisters();
            Reset();
        }

        public override void Reset()
        {
            base.Reset();
            registers.Reset();
            Array.Clear(inputState, 0, inputState.Length);
            Array.Clear(outputState, 0, outputState.Length);
            prevLatchEnable = false;

            // 初始化 Q0-Q7 低电平
            for (int i = 8; i < 16; i++)
            {
                Connections[i].Set(false);
            }
            
            this.Log(LogLevel.Noisy, "[SN74AHC273] 设备复位，所有 Q0-Q7 输出清零。");
        }

        public uint ReadDoubleWord(long offset)
        {
            uint value = registers.Read(offset);
            this.Log(LogLevel.Noisy, $"[SN74AHC273] 读取 {((Registers)offset)} = 0x{value:X8}");
            return value;
        }

        public void WriteDoubleWord(long offset, uint value)
        {
            registers.Write(offset, value);
            this.Log(LogLevel.Noisy, $"[SN74AHC273] 写入 {((Registers)offset)} = 0x{value:X8}");
        }

        public override void OnGPIO(int number, bool value)
        {
            base.OnGPIO(number, value);

            if (number < 8) // D0-D7 输入
            {
                inputState[number] = value;
                this.Log(LogLevel.Noisy, $"[SN74AHC273] D{number} 输入更新: {value}");
            }
            else if (number == 15) // LE (Latch Enable)
            {
                if (value && !prevLatchEnable) // **严格检测 LE 上升沿**
                {
                    this.Log(LogLevel.Noisy, "[SN74AHC273] LE 上升沿触发，锁存 D0-D7 到 Q0-Q7...");
                    
                    uint debugOutput = 0;  // 用于调试 Q0-Q7 值
                    for (int i = 0; i < 8; i++)
                    {
                        outputState[i] = inputState[i]; // 锁存数据
                        Connections[8 + i].Set(outputState[i]); // 更新 Q0-Q7
                        this.Log(LogLevel.Noisy, $"[SN74AHC273] Q{i} (GPIO {8 + i}) -> {outputState[i]}");

                        if (outputState[i])
                            debugOutput |= (uint)(1 << i);
                    }
                    
                    this.Log(LogLevel.Noisy, $"[SN74AHC273] 锁存后 Q0-Q7: 0x{debugOutput:X2}");
                }
                prevLatchEnable = value; // 记录 LE 状态
            }
        }

        public IGPIOReceiver GetLocalReceiver(int pin)
        {
            if (pin < 0 || pin >= NumberOfPins)
            {
                throw new RecoverableException($"[SN74AHC273] 无效的 GPIO 端口号: {pin}");
            }
            return this;
        }

        private DoubleWordRegisterCollection CreateRegisters()
        {
            var regMap = new Dictionary<long, DoubleWordRegister>
            {
                {(long)Registers.InputData, new DoubleWordRegister(this)
                    .WithValueField(0, 8, FieldMode.Read, 
                        valueProviderCallback: _ =>
                        {
                            byte result = 0;
                            for (int i = 0; i < 8; i++)
                            {
                                if (inputState[i]) 
                                    result |= (byte)(1 << i);
                            }
                            this.Log(LogLevel.Noisy, $"[SN74AHC273] 读取 IDR: 0x{result:X2}");
                            return result;
                        },
                        name: "IDR")
                },
                {(long)Registers.OutputData, new DoubleWordRegister(this)
                    .WithValueField(0, 8, FieldMode.Read, 
                        valueProviderCallback: _ =>
                        {
                            byte result = 0;
                            for (int i = 0; i < 8; i++)
                            {
                                if (outputState[i]) 
                                    result |= (byte)(1 << i);
                            }
                            this.Log(LogLevel.Noisy, $"[SN74AHC273] 读取 ODR: 0x{result:X2}");
                            return result;
                        },
                        name: "ODR")
                },
                {(long)Registers.LatchEnable, new DoubleWordRegister(this)
                    .WithFlag(0, 
                        writeCallback: (_, val) =>
                        {
                            this.Log(LogLevel.Noisy, $"[SN74AHC273] LatchEnable 写入: {val}");
                            if (val && !prevLatchEnable) // 上升沿触发
                            {
                                this.Log(LogLevel.Noisy, "[SN74AHC273] LatchEnable 上升沿触发，数据锁存！");
                                for (int i = 0; i < 8; i++)
                                {
                                    outputState[i] = inputState[i];
                                    Connections[8 + i].Set(outputState[i]);
                                }
                            }
                            prevLatchEnable = val;
                        },
                        name: "LE")
                }
            };

            return new DoubleWordRegisterCollection(this, regMap);
        }

        private enum Registers
        {
            InputData = 0x00,  // D0-D7 读取
            OutputData = 0x04, // Q0-Q7 读取
            LatchEnable = 0x08 // 触发锁存
        }
    }
}
