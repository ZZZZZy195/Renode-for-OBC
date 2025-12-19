using Antmicro.Renode.Core;
using Antmicro.Renode.Core.Structure.Registers;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Peripherals.Bus;

namespace Antmicro.Renode.Peripherals.UART
{
    [AllowedTranslations(AllowedTranslation.DoubleWordToByte)]
    public class MAX488ESA : IDoubleWordPeripheral, IProvidesRegisterCollection<DoubleWordRegisterCollection>, IKnownSize
    {
        public MAX488ESA(IMachine machine)
        {
            RegistersCollection = new DoubleWordRegisterCollection(this);
            DefineRegisters();

            // 初始化RS-485差分信号引脚
            A = new GPIO();
            B = new GPIO();
            Y = new GPIO();
            Z = new GPIO();
            
            // 初始化控制引脚
            DE = new GPIO();
            RE = new GPIO();
        }

        public uint ReadDoubleWord(long offset)
        {
            return RegistersCollection.Read(offset);
        }

        public void WriteDoubleWord(long offset, uint value)
        {
            RegistersCollection.Write(offset, value);
        }

        public void Reset()
        {
            RegistersCollection.Reset();
            A.Unset();
            B.Unset();
            Y.Unset();
            Z.Unset();
        }

        public long Size => 0x100;

        public DoubleWordRegisterCollection RegistersCollection { get; }

        // RS-485差分信号接口
        public GPIO A { get; }
        public GPIO B { get; }
        public GPIO Y { get; }
        public GPIO Z { get; }

        // 控制信号
        public GPIO DE { get; } // Driver Enable
        public GPIO RE { get; } // Receiver Enable

        private void DefineRegisters()
        {
            Registers.Control.Define(this)
                .WithFlag(0, out driverEnabled, name: "DE", 
                    writeCallback: (_, val) => DE.Set(val))
                .WithFlag(1, out receiverEnabled, name: "RE",
                    writeCallback: (_, val) => RE.Set(!val)) // RE低电平有效
                .WithReservedBits(2, 30);

            Registers.Status.Define(this)
                .WithFlag(0, FieldMode.Read, valueProviderCallback: _ => A.IsSet, name: "LINE_A")
                .WithFlag(1, FieldMode.Read, valueProviderCallback: _ => B.IsSet, name: "LINE_B")
                .WithTaggedFlag("BUS_FAULT", 2)
                .WithReservedBits(3, 29);

            Registers.Interrupt.Define(this)
                .WithFlag(0, out interruptEnabled, name: "INT_EN")
                .WithFlag(1, FieldMode.Read | FieldMode.WriteOneToClear, name: "INT_FLAG")
                .WithReservedBits(2, 30);
        }

        // 寄存器字段定义
        private IFlagRegisterField driverEnabled;
        private IFlagRegisterField receiverEnabled;
        private IFlagRegisterField interruptEnabled;

        private enum Registers
        {
            Control = 0x00,
            Status = 0x04,
            Interrupt = 0x08
        }
    }
}