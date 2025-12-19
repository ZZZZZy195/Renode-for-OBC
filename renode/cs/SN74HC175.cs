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
    public class SN74HC175 : BaseGPIOPort, IDoubleWordPeripheral, ILocalGPIOReceiver
    {
        private readonly bool[] inputState = new bool[4];
        private readonly bool[] outputState = new bool[4];
        private readonly DoubleWordRegisterCollection registers;
        
        private const int TotalPinCount = 8;  
        private const int DataPinsStart = 0;  
        private const int OutputPinsStart = 4;  

        public SN74HC175(IMachine machine) : base(machine, TotalPinCount)
        {
            registers = CreateRegisters();
            Reset();
        }

        public override void Reset()
        {
            base.Reset();
            registers.Reset();
            for(int i = 0; i < 4; i++)
            {
                inputState[i] = false;
                outputState[i] = false;
                Connections[OutputPinsStart + i].Set(false);
            }
            this.Log(LogLevel.Info, "[SN74HC175] Reset complete. All Q0-Q3 cleared.");
        }

        public uint ReadDoubleWord(long offset) => registers.Read(offset);

        public void WriteDoubleWord(long offset, uint value)
        {
            registers.Write(offset, value);

            if(offset == (long)Registers.InputData)
            {
                uint maskedValue = value & 0xF; // 只处理低4位
                for(int i = 0; i < 4; i++)
                {
                  inputState[i] = (maskedValue & (1u << i)) != 0;
                }
                this.Log(LogLevel.Info, "[SN74HC175] Write to InputData, updating Q0-Q3.");
                UpdateOutputs();

        // 忽略高位写入的警告
                if((value & 0xF0) != 0)
                {
                this.Log(LogLevel.Debug, "[SN74HC175] Ignoring upper bits [4-7] in InputData write: 0x{0:X}", value);
                }
            }
        }

        public override void OnGPIO(int number, bool value)
        {
            if(number >= DataPinsStart && number < DataPinsStart + 4)
            {
                inputState[number - DataPinsStart] = value;
                this.Log(LogLevel.Info, "[SN74HC175] GPIO {0} changed to {1}, updating Q0-Q3.", number, value);
                UpdateOutputs();
            }
        }

        private void UpdateOutputs()
        {
            for(int i = 0; i < 4; i++)
            {
                outputState[i] = inputState[i];
                Connections[OutputPinsStart + i].Set(outputState[i]);
            }
            this.Log(LogLevel.Info, "[SN74HC175] Updated Q0-Q3: {0}{1}{2}{3}", 
                outputState[3] ? 1 : 0, outputState[2] ? 1 : 0, 
                outputState[1] ? 1 : 0, outputState[0] ? 1 : 0);
        }

        public IGPIOReceiver GetLocalReceiver(int pin)
        {
            return this;
        }

        private DoubleWordRegisterCollection CreateRegisters()
        {
            var regMap = new Dictionary<long, DoubleWordRegister>
            {
                {(long)Registers.InputData, new DoubleWordRegister(this)
                    .WithValueField(0, 4, FieldMode.Read,
                        valueProviderCallback: _ =>
                        {
                            uint result = 0;
                            for(int i = 0; i < 4; i++)
                            {
                                if(inputState[i]) result |= (1u << i);
                            }
                            return result & 0xF;
                        },
                        name: "IDR")
                },

                {(long)Registers.OutputData, new DoubleWordRegister(this)
                    .WithValueField(0, 4, FieldMode.Read,
                        valueProviderCallback: _ =>
                        {
                            uint result = 0;
                            for(int i = 0; i < 4; i++)
                            {
                                if(outputState[i]) result |= (1u << i);
                            }
                            this.Log(LogLevel.Info, "[SN74HC175] Read OutputData = {0:X}", result);
                            return result;
                        },
                        name: "ODR")
                }
            };

            return new DoubleWordRegisterCollection(this, regMap);
        }

        private enum Registers
        {
            InputData = 0x00,
            OutputData = 0x04
        }
    }
}
