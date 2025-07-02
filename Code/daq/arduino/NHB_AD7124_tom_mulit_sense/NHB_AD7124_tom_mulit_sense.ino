#include <NHB_AD7124.h>

#define CH_COUNT 4  // 4 differential pairs (AIN0–1, AIN2–3, AIN4–5, AIN6–7)

// Buffer for ADC readings
double buf[CH_COUNT];
const uint8_t csPin = 5;
double exRefV = 2.048;
AD7124_BurnoutCurrents burnout;
Ad7124 adc(csPin, 4000000);  // SPI @ 4 MHz

#define CONVMODE AD7124_OpMode_Continuous  // Continuous conversion mode
int filterSelBits = 100;  // Adjust this based on noise/speed tradeoff

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;  // Wait for Serial to be ready
  }

  Serial.println("AD7124 - 4 Differential Channel Read Example");

  // Initialize ADC
  adc.begin();

  // Set mode and power
  adc.setAdcControl(CONVMODE, AD7124_FullPower, true);  // true = enable reference

  // Configure Setup 4:
  // - External reference (e.g., 3.3V)
  // - Gain = 32 → ±103 mV input range
  // - Bipolar mode (true)
  adc.setup[0].setConfig(AD7124_Ref_ExtRef1, AD7124_Gain_32, true, burnout, exRefV);
  adc.setup[0].setFilter(AD7124_Filter_SINC3, filterSelBits);
  adc.setup[1].setConfig(AD7124_Ref_ExtRef1, AD7124_Gain_32, true, burnout, exRefV);
  adc.setup[1].setFilter(AD7124_Filter_SINC3, filterSelBits);
  adc.setup[2].setConfig(AD7124_Ref_ExtRef1, AD7124_Gain_32, true, burnout, exRefV);
  adc.setup[2].setFilter(AD7124_Filter_SINC3, filterSelBits);
  adc.setup[3].setConfig(AD7124_Ref_ExtRef1, AD7124_Gain_32, true, burnout, exRefV);
  adc.setup[3].setFilter(AD7124_Filter_SINC3, filterSelBits);

  // Enable 4 fully differential input channels using Setup 4
  adc.setChannel(0, 0, AD7124_Input_AIN4, AD7124_Input_AVSS, true);  // CH0
  adc.setChannel(1, 1, AD7124_Input_AIN1, AD7124_Input_AVSS, true);  // CH1
  adc.setChannel(2, 2, AD7124_Input_AIN2, AD7124_Input_AVSS, true);  // CH2
  adc.setChannel(3, 3, AD7124_Input_AIN3, AD7124_Input_AVSS, true);  // CH3
}

void loop() {
  const int activeChannels[CH_COUNT] = {0, 1, 2, 3};  // Channel indices in ADC

  for (int i = 0; i < CH_COUNT; i++) {
    buf[i] = adc.readVolts(activeChannels[i]);  // Read differential voltage
    Serial.print(buf[i], 6);
    Serial.print("\t");
  }

  Serial.println();
  delay(6);  // Adjust for sampling rate / readability
}
