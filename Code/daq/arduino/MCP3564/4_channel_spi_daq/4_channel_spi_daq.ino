#include <SPI.h>

#define CS_PIN 10  // Chip Select

void setup() {
  Serial.begin(115200);
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);
  delay(100);

  // RESET the ADC
  writeCommand(0x61);  // Software reset
  delay(10);

  // --- DIAGNOSTIC CHECK ---
  byte deviceID = readRegister(0x07); // DEVICE_ID register
  if (deviceID != 0xB0) {
    Serial.print("ERROR: MCP3564 not found. DEVICE_ID = 0x");
    Serial.println(deviceID, HEX);
    while (1); // Halt
  } else {
    Serial.println("MCP3564 detected successfully.");
  }

  // CONFIG0: VREF_SEL=1 (internal), BOOST=1, GAIN=100 (64x)
  writeRegister(0x01, 0b10001010);

  // CONFIG1: RES=11 (24-bit), OSR=1001 (3072)
  writeRegister(0x02, 0b11001001);

  // CONFIG2: REF_BUF_EN=1, MODE=000 (continuous conversion)
  writeRegister(0x03, 0b10000000);

  // CONFIG3: IRQ_MODE=1 (push-pull), DR_EN=0, FORMAT=000 (24-bit output)
  writeRegister(0x05, 0b10000000);  // ⚠️ Prevents modulator from stalling

  // MUX: CH0 (IN+) to AGND (IN−)
  writeRegister(0x04, 0b00000000);

  // Start continuous conversions
  writeCommand(0x58);
}

void loop() {
  long adcValue = readADC();

  Serial.print("ADC Code: ");
  Serial.print(adcValue);

  // Convert to millivolts
  float voltage_mV = (adcValue / 16777215.0) * (2.048 / 64.0) * 1000.0;
  Serial.print("  |  Voltage: ");
  Serial.print(voltage_mV, 5);
  Serial.println(" mV");

  delay(1); // ~1 ms between samples
}

// ===== SPI Command Helpers =====

void writeRegister(byte reg, byte value) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x00 | (reg & 0x3F));  // WRITE command (CMD=00)
  SPI.transfer(value);
  digitalWrite(CS_PIN, HIGH);
}

byte readRegister(byte reg) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x40 | (reg & 0x3F));  // READ command (CMD=01)
  byte value = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  return value;
}

void writeCommand(byte command) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(command);
  digitalWrite(CS_PIN, HIGH);
}

long readADC() {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x40);  // READ ADCDATA (register 0x00)
  byte b1 = SPI.transfer(0x00);
  byte b2 = SPI.transfer(0x00);
  byte b3 = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);

  long val = ((long)b1 << 16) | ((long)b2 << 8) | b3;

  // Sign-extend 24-bit to 32-bit
  if (val & 0x800000)
    val |= 0xFF000000;

  return val;
}
