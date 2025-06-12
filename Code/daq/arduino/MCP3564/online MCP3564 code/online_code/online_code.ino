#include <SPI.h>

#define SPI_CS_PIN 10  // Chip select for MCP3564

#define VREF 2.048      // External voltage reference connected to VREF+
#define OSR_SETTING 0x02  // OSR_128

// MCP3564 SPI command constants
#define ADDR           0b01000000
#define RESET          0b00111000
#define CONVERT        0b00101000
#define CONFIG0_ADDR   0x04
#define CONFIG1_ADDR   0x08
#define CONFIG2_ADDR   0x0C
#define CONFIG3_ADDR   0x10
#define MUX_ADDR       0x18
#define ADC_READ       0x00
#define READ_FLAG      0x01

// MCP3564 register values
#define CONFIG0  (0b11010010 | (1 << 5))  // Internal clock enabled
#define CONFIG1  (OSR_SETTING << 2)       // OSR_128
#define CONFIG2  0b10000011  // Bit 7 = 1: external VREF, PGA = 1, REFEXT+ only
#define CONFIG3  0b11110000  // Continuous conversion mode
#define MUX      0b00000000  // SE_0 (AIN0 with respect to GND)

// === Setup ===
void setup() {
  Serial.begin(115200);
  pinMode(SPI_CS_PIN, OUTPUT);
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.begin();

  delay(100);  // Allow MCP3564 to boot
  resetADC();
  configureADC();
  startConversion();

  Serial.println("MCP3564 initialized using internal clock and external VREF (2.048 V).");
}

// === Main Loop ===
void loop() {
  int32_t adc_code = readADC();
  float voltage = (adc_code / 8388607.0) * VREF;

  Serial.print("ADC Code: ");
  Serial.print(adc_code);
  Serial.print(" | Voltage: ");
  Serial.println(voltage, 6);

  delay(100);  // 10 Hz read rate
}

// === Helper Functions ===
void resetADC() {
  digitalWrite(SPI_CS_PIN, LOW);
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE0));
  SPI.transfer(ADDR | RESET);  // Fast command: RESET
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.endTransaction();
  delay(5);
}

void configureADC() {
  digitalWrite(SPI_CS_PIN, LOW);
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE0));
  SPI.transfer(ADDR | CONFIG0_ADDR);  // Start writing at CONFIG0
  SPI.transfer(CONFIG0);
  SPI.transfer(CONFIG1);
  SPI.transfer(CONFIG2);  // Use external reference
  SPI.transfer(CONFIG3);
  SPI.transfer(MUX);
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.endTransaction();
  delay(10);
}

void startConversion() {
  digitalWrite(SPI_CS_PIN, LOW);
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE0));
  SPI.transfer(ADDR | CONVERT);  // Begin continuous conversion
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.endTransaction();
}

int32_t readADC() {
  uint32_t raw = 0;
  digitalWrite(SPI_CS_PIN, LOW);
  SPI.beginTransaction(SPISettings(2000000, MSBFIRST, SPI_MODE0));
  SPI.transfer(ADDR | ADC_READ | READ_FLAG);  // Read ADC register
  for (int i = 0; i < 4; i++) {
    raw <<= 8;
    raw |= SPI.transfer(0x00);
  }
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.endTransaction();

  // Extract signed 24-bit data (bits 0–25), discard tag
  int32_t value = ((int32_t)(raw & 0x01FFFFFF) << 7) >> 7;
  return value;
}
