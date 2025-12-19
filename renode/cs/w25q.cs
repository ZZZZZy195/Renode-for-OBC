using System;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Core.Structure;
using Antmicro.Renode.Core;
using Antmicro.Renode.Core.Structure.Registers;
using Antmicro.Renode.Utilities.Collections;

namespace Antmicro.Renode.Peripherals.SPI
{
    public sealed class W25Q512_SPI : NullRegistrationPointPeripheralContainer<ISPIPeripheral>, IWordPeripheral, IDoubleWordPeripheral, IBytePeripheral, IKnownSize
    {
        public W25Q512_SPI(IMachine machine, int bufferCapacity = DefaultBufferCapacity) : base(machine)
        {
            receiveBuffer = new CircularBuffer<byte>(bufferCapacity);
            
            // GPIO初始化
            CS = new GPIO();   // Chip Select (CS)
            VCC = new GPIO();  // 电源（VCC）
            SO = new GPIO();   // 数据输出（MISO），也可作为复位（SO/RST）
            WP = new GPIO();   // 写保护（WP）
            CLK = new GPIO();  // 时钟（CLK）
            GND = new GPIO();  // 地（GND）
            SI = new GPIO();   // 数据输入（MOSI）
            IRQ = new GPIO();  // 中断请求（IRQ）
            DMARecieve = new GPIO(); // DMA 接收引脚

            registers = new DoubleWordRegisterCollection(this);
            SetupRegisters();
            Reset();
        }

        // GPIO接口
        public GPIO CS { get; }
        public GPIO VCC { get; }
        public GPIO SO { get; }
        public GPIO WP { get; }
        public GPIO CLK { get; }
        public GPIO GND { get; }
        public GPIO SI { get; }
        public GPIO IRQ { get; }
        public GPIO DMARecieve { get; }

        #region 总线接口
        public byte ReadByte(long offset)
        {
            if(offset % 4 == 0)
            {
                return (byte)ReadDoubleWord(offset);
            }
            this.LogUnhandledRead(offset);
            return 0;
        }

        public void WriteByte(long offset, byte value)
        {
            if(offset % 4 == 0)
            {
                WriteDoubleWord(offset, (uint)value);
            }
            else
            {
                this.LogUnhandledWrite(offset, value);
            }
        }

        public ushort ReadWord(long offset) => (ushort)ReadDoubleWord(offset);
        public void WriteWord(long offset, ushort value) => WriteDoubleWord(offset, value);

        public uint ReadDoubleWord(long offset) => registers.Read(offset);
        public void WriteDoubleWord(long offset, uint value) => registers.Write(offset, value);
        #endregion

        #region 核心功能
        public override void Reset()
        {
            // 初始化GPIO状态
            CS.Set(false);
            SO.Set(false);  // 默认为低电平，表示MISO为空
            WP.Set(false);
            CLK.Set(false);
            VCC.Set(true);  // 设置电源为高电平
            GND.Set(false);
            SI.Set(false);  // 默认为低电平，表示MOSI为空
            IRQ.Unset();
            DMARecieve.Unset();
            
            lock(receiveBuffer)
            {
                receiveBuffer.Clear();
            }
            registers.Reset();
        }

        public long Size => 0x400;
        #endregion

        #region SPI传输逻辑
        private uint HandleDataRead()
        {
            if(!VCC.IsSet)
            {
                this.Log(LogLevel.Error, "Device not powered");
                return 0xFF;
            }

            lock(receiveBuffer)
            {
                if(receiveBuffer.TryDequeue(out var value))
                {
                    UpdateInterrupts();
                    return value;
                }
                return 0;
            }
        }

        private void HandleDataWrite(uint value)
        {
            if(!CheckTransferConditions())
                return;

            var byteValue = (byte)(value & 0xFF);
            var response = ProcessSPITransfer(byteValue);
            
            lock(receiveBuffer)
            {
                receiveBuffer.Enqueue(response);
                if(rxDmaEnable.Value)
                {
                    DMARecieve.Blink();
                }
                this.NoisyLog("Tx: 0x{0:X2} Rx: 0x{1:X2}", byteValue, response);
            }
            UpdateInterrupts();
        }

        private byte ProcessSPITransfer(byte value)
        {
            CLK.Set(true);
            var response = RegisteredPeripheral?.Transmit(value) ?? 0xFF;
            CLK.Set(false);
            return response;
        }

        private bool CheckTransferConditions()
        {
            if(!VCC.IsSet)
            {
                this.Log(LogLevel.Error, "VCC not powered");
                return false;
            }
            if(CS.IsSet)
            {
                this.Log(LogLevel.Warning, "CS not active");
                return false;
            }
            if(WP.IsSet)
            {
                this.Log(LogLevel.Warning, "Write protected");
                return false;
            }
            return true;
        }
        #endregion

        #region 寄存器配置
        private void SetupRegisters()
        {
            // Control Register 1
            Registers.Control1.Define(registers)
                .WithFlag(0, name: "CPHA")
                .WithFlag(1, name: "CPOL")
                .WithFlag(2, writeCallback: (previous, value) => 
                {
                    if(!value) this.Log(LogLevel.Warning, "Slave mode not supported");
                }, name: "MSTR")
                .WithValueField(3, 3, name: "BaudRate")
                .WithFlag(6, changeCallback: (_, newValue) => 
                {
                    if(!newValue) IRQ.Unset();
                }, name: "SPI_Enable")
                .WithFlag(7, name: "LSB_First");

            // Status Register
            Registers.Status.Define(registers)
                .WithFlag(0, FieldMode.Read, valueProviderCallback: _ => receiveBuffer.Count > 0, name: "RXNE")
                .WithFlag(1, FieldMode.Read, valueProviderCallback: _ => true, name: "TXE");

            // Data Register
            Registers.Data.Define(registers)
                .WithValueField(0, 32, 
                    valueProviderCallback: _ => HandleDataRead(),
                    writeCallback: (_, value) => HandleDataWrite((uint)value),
                    name: "DR");

            // Control Register 2
            Registers.Control2.Define(registers)
                .WithFlag(0, out rxDmaEnable, name: "RX_DMA_Enable")
                .WithFlag(6, out rxBufferNotEmptyInterruptEnable, name: "RX_Int_Enable")
                .WithFlag(7, out txBufferEmptyInterruptEnable, name: "TX_Int_Enable");
        }

        private void UpdateInterrupts()
        {
            var irqState = txBufferEmptyInterruptEnable.Value || 
                         (rxBufferNotEmptyInterruptEnable.Value && receiveBuffer.Count > 0);
            IRQ.Set(irqState);
        }
        #endregion

        #region 状态管理
        private DoubleWordRegisterCollection registers;
        private IFlagRegisterField txBufferEmptyInterruptEnable, 
                                  rxBufferNotEmptyInterruptEnable, 
                                  rxDmaEnable;

        private readonly CircularBuffer<byte> receiveBuffer;
        private const int DefaultBufferCapacity = 8;

        private enum Registers
        {
            Control1 = 0x00,
            Control2 = 0x04,
            Status = 0x08,
            Data = 0x0C
        }
        #endregion
    }
}
