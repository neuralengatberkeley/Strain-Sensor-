/* This arduino Code is used for the auto-bender, VOLTAGE DIVIDER circuit, DYNAMIC TEST.

The following actions are taking by the auto bender:

(1) rotate a servo motor 90 deg repeatedly and  (2) acquire data from the following sensors:  

 [A] Strain Gauge voltage data ouput from :  (1) Voltage Divider Circuit Adafruit_MCP3421 (https://www.adafruit.com/product/5870) 

*/

#include "Adafruit_MCP3421.h"


//Adafruit_NAU7802 nau;  // 24 bit ADC object created for wheatstone bridge circuit
Adafruit_MCP3421 mcp;  // 14 bit ADC object created for voltage divider circuit
int32_t adcValue = 0;

void setup() {
  Serial.begin(115200);
   
 // Begin can take an optional address and Wire interface
  if (!mcp.begin(0x68, &Wire)) { 
    Serial.println("Failed to find MCP3421 chip");
    while (1) {
      delay(10); // Avoid a busy-wait loop
    }
  }

   // Options: GAIN_1X, GAIN_2X, GAIN_4X, GAIN_8X
  mcp.setGain(GAIN_4X);
  Serial.print("Gain set to: ");
  switch (mcp.getGain()) {
    case GAIN_1X: Serial.println("1X"); break;
    case GAIN_2X: Serial.println("2X"); break;
    case GAIN_4X: Serial.println("4X"); break;
    case GAIN_8X: Serial.println("8X"); break;
  }

  // The resolution affects the sample rate (samples per second, SPS)
  // Other options: RESOLUTION_14_BIT (60 SPS), RESOLUTION_16_BIT (15 SPS), RESOLUTION_18_BIT (3.75 SPS)
  mcp.setResolution(RESOLUTION_16_BIT); // 240 SPS (12-bit)
  Serial.print("Resolution set to: ");
  switch (mcp.getResolution()) {
    case RESOLUTION_12_BIT: Serial.println("12 bits"); break;
    case RESOLUTION_14_BIT: Serial.println("14 bits"); break;
    case RESOLUTION_16_BIT: Serial.println("16 bits"); break;
    case RESOLUTION_18_BIT: Serial.println("18 bits"); break;
  }

  // Test setting and getting Mode
  mcp.setMode(MODE_CONTINUOUS); // Options: MODE_CONTINUOUS, MODE_ONE_SHOT
  Serial.print("Mode set to: ");
  switch (mcp.getMode()) {
    case MODE_CONTINUOUS: Serial.println("Continuous"); break;
    case MODE_ONE_SHOT: Serial.println("One-shot"); break;
  }
  
  delay(2000);
}

void loop() {
   
     adcValue = mcp.readADC(); // Read ADC value
        //Serial.print("ADC reading: ");
        Serial.print(adcValue);
      
        Serial.print("\n");
        //sampleCount++; // Increment the sample count
    //}
    delay(50); 
        
}
