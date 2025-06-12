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
  sendCommand(0x61);
  delay(10);

  // Check DEVICE_ID
  uint8_t id = readRegister(0x07);
  Serial.print("DEVICE_ID = 0x");
  Serial.println(id, HEX);
  if (id != 0xB0) {
    Serial.println("❌ ERROR: MCP3564 not found!");
    while (1);
  }

  // Config registers
  writeRegister(0x01, 0b10000000); // CONFIG0
  writeRegister(0x02, 0b11000110); // CONFIG1
  writeRegister(0x03, 0b10000000); // CONFIG2
  writeRegister(0x04, 0b00000000); // MUX CH0 to AGND

  sendCommand(0x58); // Start continuous conversion
  Serial.println("✅ Configuration done. Reading ADC...");
}

void loop() {
  long adc = readADC();
  Serial.print("ADC Code: ");
  Serial.println(adc);
  delay(500);
}

void sendCommand(uint8_t command) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(command);
  digitalWrite(CS_PIN, HIGH);
}

void writeRegister(uint8_t reg, uint8_t val) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x40 | (reg << 1));
  SPI.transfer(val);
  digitalWrite(CS_PIN, HIGH);
}

uint8_t readRegister(uint8_t reg) {
  digitalWrite(CS_PIN, LOW);
  SPI.transfer(0x01 | (reg << 1));
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
  if (result & 0x800000) result |= 0xFF000000;
  return result;
}
