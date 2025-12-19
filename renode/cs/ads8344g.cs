using System;
using Antmicro.Renode.Core;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Core.Structure.Registers;
using Antmicro.Renode.Time;
using Antmicro.Renode.Peripherals;

namespace Antmicro.Renode.Peripherals.Analog
{
    public class ADS8344_ADC : BasicDoubleWordPeripheral, IKnownSize
    {
        public ADS8344_ADC(IMachine machine) : base(machine)
        {
            IRQ = new GPIO();
            CS = new GPIO();    // 定义用于连接CS的GPIO
            DOUT = new GPIO();  // 定义用于连接DOUT的GPIO
            DIN = new GPIO();   // 定义用于连接DIN的GPIO
            DCLK = new GPIO();  // 定义用于连接DCLK的GPIO
            BUSY = new GPIO();  // 定义用于连接BUSY的GPIO

            // 使用 GPIO 来处理模拟信号输入（AIn0 到 AIn7）
            AIn0 = new GPIO();  // CH0
            AIn1 = new GPIO();  // CH1
            AIn2 = new GPIO();  // CH2
            AIn3 = new GPIO();  // CH3
            AIn4 = new GPIO();  // CH4
            AIn5 = new GPIO();  // CH5
            AIn6 = new GPIO();  // CH6
            AIn7 = new GPIO();  // CH7

            this.machine = machine; // 记录 Renode 机器实例
            DefineRegisters();
        }

        public override void Reset()
        {
            base.Reset();
            IRQ.Unset();
            CS.Unset();   // 重置GPIO状态
            DOUT.Unset();
            DIN.Unset();
            DCLK.Unset();
            BUSY.Unset();
        }

        public GPIO IRQ { get; }
        public GPIO CS { get; }
        public GPIO DOUT { get; }
        public GPIO DIN { get; }
        public GPIO DCLK { get; }
        public GPIO BUSY { get; }

        // 输入通道（直接通过 GPIO 引脚处理）
        public GPIO AIn0 { get; }
        public GPIO AIn1 { get; }
        public GPIO AIn2 { get; }
        public GPIO AIn3 { get; }
        public GPIO AIn4 { get; }
        public GPIO AIn5 { get; }
        public GPIO AIn6 { get; }
        public GPIO AIn7 { get; }

        public long Size => 0x1000;

        // 定义寄存器
        public uint AIn0Value { get => aIn0; set => aIn0 = ValidateAIn(value); }
        public uint AIn1Value { get => aIn1; set => aIn1 = ValidateAIn(value); }
        public uint AIn2Value { get => aIn2; set => aIn2 = ValidateAIn(value); }
        public uint AIn3Value { get => aIn3; set => aIn3 = ValidateAIn(value); }
        public uint AIn4Value { get => aIn4; set => aIn4 = ValidateAIn(value); }
        public uint AIn5Value { get => aIn5; set => aIn5 = ValidateAIn(value); }
        public uint AIn6Value { get => aIn6; set => aIn6 = ValidateAIn(value); }
        public uint AIn7Value { get => aIn7; set => aIn7 = ValidateAIn(value); }

        private uint ValidateAIn(uint value)
        {
            return value & 0xFFF;
        }

        private void DefineRegisters()
        {
            Registers.Control.Define(this)
                .WithFlag(0, FieldMode.WriteOneToClear, name: "CTRL.start",
                    writeCallback: (_, value) =>
                    {
                        if (value)
                        {
                            StartConversion();
                        }
                    });

            Registers.OutputData.Define(this)
                .WithValueField(0, 12, name: "DATA.data",
                    valueProviderCallback: _ => GetValueFromActiveChannel());

            Registers.InterruptControl.Define(this)
                .WithFlag(0, out interruptDoneEnabled, name: "INTR.done_ie")         
                .WithFlag(1, out interruptReferenceReadyEnabled, name: "INTR.ref_ie") 
                .WithReservedBits(2, 26)
                .WithFlag(28, out interruptDonePending, FieldMode.WriteOneToClear | FieldMode.Read, name: "INTR.done_if") 
                .WithFlag(29, out interruptReferenceReadyPending, FieldMode.WriteOneToClear | FieldMode.Read, name: "INTR.ref_ready_if")
                .WithFlag(30, out interruptAnyPending, FieldMode.Read, name: "INTR.pending");
        }

        private uint GetValueFromActiveChannel()
        {
            return currentChannel.Value switch
            {
                Channels.AIn0 => AIn0Value,
                Channels.AIn1 => AIn1Value,
                Channels.AIn2 => AIn2Value,
                Channels.AIn3 => AIn3Value,
                Channels.AIn4 => AIn4Value,
                Channels.AIn5 => AIn5Value,
                Channels.AIn6 => AIn6Value,
                Channels.AIn7 => AIn7Value,
                _ => 0
            };
        }

        private void StartConversion()
        {
            this.Log(LogLevel.Debug, "Starting ADC conversion");

            // 使用 Renode 事件调度机制，模拟 10ms 后 ADC 采样完成
            machine.ScheduleAction(TimeInterval.FromMilliseconds(10), _ => OnConversionFinished());
        }

        private void OnConversionFinished()
        {
            this.Log(LogLevel.Debug, "ADC Conversion Finished");
            interruptDonePending.Value = true;
            IRQ.Set(true);
        }

        private const uint ADCDataMask = 0xFFF; 

        private IEnumRegisterField<Channels> currentChannel;
        private IFlagRegisterField interruptDoneEnabled;
        private IFlagRegisterField interruptReferenceReadyEnabled;
        private IFlagRegisterField interruptDonePending;
        private IFlagRegisterField interruptReferenceReadyPending;
        private IFlagRegisterField interruptAnyPending;

        private uint aIn0;
        private uint aIn1;
        private uint aIn2;
        private uint aIn3;
        private uint aIn4;
        private uint aIn5;
        private uint aIn6;
        private uint aIn7;

        private readonly IMachine machine;  // 用于 Renode 事件调度

        private enum Channels
        {
            AIn0 = 0,
            AIn1,
            AIn2,
            AIn3,
            AIn4,
            AIn5,
            AIn6,
            AIn7
        }

        private enum Registers : long
        {
            Control = 0x00,
            Status = 0x04,
            OutputData = 0x08,
            InterruptControl = 0x0C,
        }
    }
}
