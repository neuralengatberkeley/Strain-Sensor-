#include <SPI.h>
#define CS_PIN 10

void setup() {
  Serial.begin(115200);
  SPI.begin();
  pinMode(CS_PIN, OUTPUT);
  digitalWrite(CS_PIN, HIGH);

  delay(500);
  Serial.println("MCP3564 SPI Test");

  // Reset the chip
  sendCommand(0x61);  // Fast command: RESET
  delay(10);

  // Read DEVICE_ID (optional)
  uint8_t id = readRegister(0x07);
  Serial.print("DEVICE_ID = 0x");
  Serial.println(id, HEX);

  // === Configuration for:
  // Internal oscillator
  // OSR = 256
  // PGA = 128
  // External VREF+ = 3.3 V
  // Single-ended AIN0 vs AGND
  writeRegister(0x01, 0b11000000);  // CONFIG0: Internal clock
  writeRegister(0x02, 0b10001000);  // CONFIG1: OSR = 256
  writeRegister(0x03, 0b10001111);  // CONFIG2: Ext VREF, PGA = 128
  writeRegister(0x04, 0b00000000);  // MUX: AIN0 vs AGND (SE0)

  sendCommand(0x58);  // Start continuous conversion
  Serial.println("✅ Configuration done. Reading ADC...");
}

void loop() {
  long adc = readADC();

  // Convert to voltage using VREF = 3.3V and PGA = 128
  float voltage = (adc / 8388607.0) * (3.3 / 128.0);

  Serial.print("ADC Code: ");
  Serial.print(adc);
  Serial.print(" | Voltage: ");
  Serial.println(voltage, 6);

  delay(1200);  // ~1 sample/sec
}

// === SPI Helper Functions ===
void sendCommand(uint8_t command) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(command);
  digitalWrite(CS_PIN, HIGH);
}

void writeRegister(uint8_t reg, uint8_t val) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x40 | (reg << 1));  // WRITE command
  SPI.transfer(val);
  digitalWrite(CS_PIN, HIGH);
}

uint8_t readRegister(uint8_t reg) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x01 | (reg << 1));  // READ command
  uint8_t value = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);
  return value;
}

long readADC() {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x01);  // READ ADCDATA
  uint8_t b1 = SPI.transfer(0x00);
  uint8_t b2 = SPI.transfer(0x00);
  uint8_t b3 = SPI.transfer(0x00);
  digitalWrite(CS_PIN, HIGH);

  long result = ((long)b1 << 16) | (b2 << 8) | b3;
  if (result & 0x800000) result |= 0xFF000000;  // Sign extend 24-bit to 32-bit
  return result;
}
