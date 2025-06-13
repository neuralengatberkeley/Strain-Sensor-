// Arduino Uno-Compatible MCP3564 Control Code with SPI Communication Check
#include <Arduino.h>
#include <SPI.h>

const float Current_Command_Array[] = {0};
#define Time_Scale 5
#define IGBT_PWM_PIN 6        // Use PWM-capable pin
#define ADC_INT_PIN 2         // Must support external interrupt
#define SPI_CS_PIN 10

#define Shunt_R 0.0005
#define Ramp_Rate 2000
#define Current_Limit 200
#define Battery_Voltage_Limit 2.90
#define IGBT_Temp_Limit 100
#define Battery_Temp_Limit 60

#define P 0.5
#define I 0.1

enum OSR : uint8_t {
  OSR_128 = 0x02,
};
enum Channel : uint8_t {
  SE_0 = 0x00,
  SE_1 = 0x01,
  SE_2 = 0x02,
  SE_3 = 0x03,
};

constexpr double Current_Sense_Gain = 1 / (Shunt_R * 20);
constexpr double Ramp_Rate_Per_Millisecond = (double)Ramp_Rate / 1000;

// SPI Constants
constexpr uint32_t SPI_Speed = 2000000;
constexpr uint8_t ADDR = 0b01000000;
constexpr uint8_t FAST_CMD = 0b00000000;
constexpr uint8_t CONV = 0b00101000;
constexpr uint8_t STBY = 0b00101100;
constexpr uint8_t RESET = 0b00111000;
constexpr uint8_t INC_WRITE = 0b00000010;
constexpr uint8_t CONFIG0_ADDR = 0x04;
constexpr uint8_t CONFIG1_ADDR = 0x08;
constexpr uint8_t CONFIG2_ADDR = 0x0C;
constexpr uint8_t CONFIG3_ADDR = 0x10;
constexpr uint8_t IRQ_ADDR = 0x14;
constexpr uint8_t MUX_ADDR = 0x18;
constexpr uint8_t ADC_REG = 0x00;
constexpr uint8_t READ = 0x01;
constexpr uint8_t DEVICE_ID_ADDR = 0x0D;

// Config Register Values
constexpr uint8_t CONFIG0 = 0b11010010;
constexpr uint8_t CONFIG1 = (OSR_128 << 2);
constexpr uint8_t CONFIG2 = 0b10001011;
constexpr uint8_t CONFIG3 = 0b11110000;
constexpr uint8_t IRQ = 0b01110110;
constexpr uint8_t MUX = 0x00; // SE_0 selected

// Globals
volatile double Current_Command = 0;
volatile double Integral = 0;
volatile double Battery_Current = 0;
volatile double Battery_Voltage = 0;
volatile double IGBT_Temp = 0;
volatile double Battery_Temp = 0;
volatile uint32_t Time = 0;
volatile double Current_Array[4] = {0};
volatile uint32_t Current_Array_Index = 0;
volatile double Voltage_Array[2] = {0};
volatile uint32_t Voltage_Array_Index = 0;
bool Running = false;
uint32_t Last_Time = 0;
uint32_t i = 0;
double Current_Command_Temp = 0;
uint32_t Time_Delta_OLD = 0;

// Helper: SPI transfer 32-bit
uint32_t SPI_transfer32(uint32_t data) {
  uint32_t value = 0;
  value |= (uint32_t)SPI.transfer((data >> 24) & 0xFF) << 24;
  value |= (uint32_t)SPI.transfer((data >> 16) & 0xFF) << 16;
  value |= (uint32_t)SPI.transfer((data >> 8) & 0xFF) << 8;
  value |= (uint32_t)SPI.transfer(data & 0xFF);
  return value;
}

// SPI Communication Check
void checkMCP3564Connection() {
  SPI.beginTransaction(SPISettings(SPI_Speed, MSBFIRST, SPI_MODE0));
  digitalWrite(SPI_CS_PIN, LOW);
  SPI.transfer(ADDR | DEVICE_ID_ADDR | READ);
  uint8_t id1 = SPI.transfer(0x00);
  uint8_t id2 = SPI.transfer(0x00);
  uint8_t id3 = SPI.transfer(0x00);
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.endTransaction();

  Serial.println("Raw SDO response from MCP3564:");
  Serial.print("0x"); Serial.print(id1, HEX); Serial.print(" 0x"); Serial.print(id2, HEX); Serial.print(" 0x"); Serial.println(id3, HEX);
}

void ADC_Init() {
  SPI.begin();
  pinMode(SPI_CS_PIN, OUTPUT);
  digitalWrite(SPI_CS_PIN, HIGH);
  delay(100);

  checkMCP3564Connection();

  SPI.beginTransaction(SPISettings(SPI_Speed, MSBFIRST, SPI_MODE0));
  digitalWrite(SPI_CS_PIN, LOW);
  SPI.transfer(ADDR | RESET | FAST_CMD);
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.endTransaction();
  delay(5);

  SPI.beginTransaction(SPISettings(SPI_Speed, MSBFIRST, SPI_MODE0));
  digitalWrite(SPI_CS_PIN, LOW);
  SPI.transfer(ADDR | CONFIG0_ADDR | INC_WRITE);
  SPI.transfer(CONFIG0);
  SPI.transfer(CONFIG1);
  SPI.transfer(CONFIG2);
  SPI.transfer(CONFIG3);
  SPI.transfer(IRQ);
  SPI.transfer(MUX);
  digitalWrite(SPI_CS_PIN, HIGH);
  SPI.endTransaction();
}

void setup() {
  Serial.begin(115200);
  pinMode(IGBT_PWM_PIN, OUTPUT);
  analogWrite(IGBT_PWM_PIN, 0);
  pinMode(ADC_INT_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(ADC_INT_PIN), [] {}, FALLING); // dummy ISR
  ADC_Init();
  Last_Time = micros();
}

void loop() {
  // Placeholder loop
}
