using System;
using Antmicro.Renode.Core;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Peripherals.Bus;
using Antmicro.Renode.Core.CAN;

namespace Antmicro.Renode.Peripherals.CAN
{
    public class TJA1050 : IDoubleWordPeripheral, ICAN
    {
        private readonly IMachine machine;

        public TJA1050(IMachine machine)
        {
            this.machine = machine;
            TXD = new GPIO();
            RXD = new GPIO();
            IRQ = new GPIO();
            Reset();
        }

        public void Reset()
        {
            // Reset GPIO pins
            TXD.Unset();
            RXD.Unset();
            IRQ.Unset();

            // Reset internal states
            interruptPending = false;
            transmitting = false;
            receiving = false;
        }

        public uint ReadDoubleWord(long offset)
        {
            // Here we would implement reading from TJA1050 registers
            // For simplicity, just return 0 as we are focusing on basic functionality
            return 0;
        }

        public void WriteDoubleWord(long offset, uint value)
        {
            // Handling the write operation, mostly controlling TXD
            if (offset == TX_OFFSET)
            {
                // Start transmission when data is written to the TX register
                Transmit(value);
            }
        }

        public void OnFrameReceived(CANMessageFrame rxMessage)
        {
            // Simulating RXD pin receiving CAN message
            receiving = true;
            LogMessageReceived(rxMessage);
            UpdateInterrupts();
        }

        private void Transmit(uint data)
        {
            // Simulate transmitting data via TXD pin
            transmitting = true;
            this.Log(LogLevel.Info, "Transmitting data: 0x{0:X}", data);
            TXD.Set(); // Set TXD pin high to indicate transmission
            machine.LocalTimeSource.ExecuteInNearestSyncedState(_ =>
            {
                // After some time, complete the transmission and clear the TXD pin
                TXD.Unset();
                transmitting = false;
                FrameSent?.Invoke(new CANMessageFrame(0, new byte[] { (byte)data })); // Trigger FrameSent event
                UpdateInterrupts();
            });
        }

        private void UpdateInterrupts()
        {
            // Trigger IRQ for transmitting or receiving events
            interruptPending = transmitting || receiving;
            IRQ.Set(interruptPending);
        }

        private void LogMessageReceived(CANMessageFrame message)
        {
            // Log the received message (basic logging)
            this.Log(LogLevel.Info, "Received CAN message: ID={0:X}, Data={1}",
                message.Id, BitConverter.ToString(message.Data));
        }

        public GPIO TXD { get; }
        public GPIO RXD { get; }
        public GPIO IRQ { get; }

        private bool interruptPending;
        private bool transmitting;
        private bool receiving;

        private const long TX_OFFSET = 0x10; // Assume TX register is at 0x10

        // Implementing the FrameSent event from ICAN interface
        public event Action<CANMessageFrame> FrameSent;
    }
}
