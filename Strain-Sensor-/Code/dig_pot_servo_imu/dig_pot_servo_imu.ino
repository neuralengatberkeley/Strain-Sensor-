/* This arduino Code is used for the auto-bender.

The following actions are taking by the auto bender and WHEATSTONE BRIDGE CIRCUIT:

(1) rotate a servo motor and  (2) acquire data from the following sensors:  

 [A] Strain Gauge voltage data ouput from either:  (1) Voltage Divider Circuit Adafruit_MCP3421 (https://www.adafruit.com/product/5870) 
or (2) Wheatstone Bridge Circuit (https://www.adafruit.com/product/4538)
 [B] Adafruit BNO055 (https://www.adafruit.com/product/2472)
 [C] 320 degree Rotary Hall Effect Encoder (https://p3america.com/erc-series/?srsltid=AfmBOoqSsWeZSXtsMrmoHyB0DRfSCzyFlUfcnEff0gpcv0NyMvrGhVz7)

 */

// The following libraries are or were used 

#include <Adafruit_DS3502.h>  // Digital Potentiometer Library no longer used
#include <Servo.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <Adafruit_INA219.h>  // Current Sensor Library no longer used
#include "Adafruit_MCP3421.h"
#include <Adafruit_NAU7802.h>

Adafruit_NAU7802 nau;  // 24 bit ADC object created for wheatstone bridge circuit
Adafruit_MCP3421 mcp;  // 14 bit ADC object created for voltage divider circuit

//IMU object with address 0x29....we may need multiple imu's in future and so will need to specify address 
//of each imu...could also just use a multiplexer
Adafruit_BNO055 bno = Adafruit_BNO055(55);  //19, 0x29

sensors_event_t event; 

Servo myservo;  // create servo object to control a servo
int i ; //actual angle setting for servo
String d; //incoming string data from python
int state; //d converted to integer
int analogPin = A0; // potentiometer wiper (middle terminal) connected to analog pin 3
                    // outside leads to ground and +5V
int val = 0;  // variable to store the value read

void setup() {
  Serial.begin(115200);
   
   myservo.write(0);
   myservo.attach(9);  // attaches the servo on Arduino digital pin 9 to the servo object

    /* Initialise the sensor */
  if(!bno.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }
  delay(50);  
  bno.setExtCrystalUse(true);
  //set servo position to zero

  Serial.println("NAU7802");
  if (! nau.begin()) {
    Serial.println("Failed to find NAU7802");
  }
  Serial.println("Found NAU7802");


/* 

Here we have the option of selecting the excitation voltage to the weatstone bridge.  To my understanding, the larger the excitation voltage, 
the larger potential for increased SNR.

*/
  nau.setLDO(NAU7802_2V7);
  Serial.print("LDO voltage set to ");
  switch (nau.getLDO()) {
    case NAU7802_4V5:  Serial.println("4.5V"); break;
    case NAU7802_4V2:  Serial.println("4.2V"); break;
    case NAU7802_3V9:  Serial.println("3.9V"); break;
    case NAU7802_3V6:  Serial.println("3.6V"); break;
    case NAU7802_3V3:  Serial.println("3.3V"); break;
    case NAU7802_3V0:  Serial.println("3.0V"); break;
    case NAU7802_2V7:  Serial.println("2.7V"); break;
    case NAU7802_2V4:  Serial.println("2.4V"); break;
    case NAU7802_EXTERNAL:  Serial.println("External"); break;
  }

/* 

Here we have the option of selecting the Vout gain of the weatstone bridge.  To my understanding, the larger the gain, 
the larger potential for increased SNR.  However, to high of a gain may saturate readout of the ADC (we have 24 bit so max of 2^24)

*/

  nau.setGain(NAU7802_GAIN_64);
  Serial.print("Gain set to ");
  switch (nau.getGain()) {
    case NAU7802_GAIN_1:  Serial.println("1x"); break;
    case NAU7802_GAIN_2:  Serial.println("2x"); break;
    case NAU7802_GAIN_4:  Serial.println("4x"); break;
    case NAU7802_GAIN_8:  Serial.println("8x"); break;
    case NAU7802_GAIN_16:  Serial.println("16x"); break;
    case NAU7802_GAIN_32:  Serial.println("32x"); break;
    case NAU7802_GAIN_64:  Serial.println("64x"); break;
    case NAU7802_GAIN_128:  Serial.println("128x"); break;
  }

  nau.setRate(NAU7802_RATE_10SPS);
  Serial.print("Conversion rate set to ");
  switch (nau.getRate()) {
    case NAU7802_RATE_10SPS:  Serial.println("10 SPS"); break;
    case NAU7802_RATE_20SPS:  Serial.println("20 SPS"); break;
    case NAU7802_RATE_40SPS:  Serial.println("40 SPS"); break;
    case NAU7802_RATE_80SPS:  Serial.println("80 SPS"); break;
    case NAU7802_RATE_320SPS:  Serial.println("320 SPS"); break;
  }

  // Take 10 readings to flush out readings
  for (uint8_t i=0; i<10; i++) {
    while (! nau.available()) delay(1);
    nau.read();
  }

  while (! nau.calibrate(NAU7802_CALMOD_INTERNAL)) {
    Serial.println("Failed to calibrate internal offset, retrying!");
    delay(1000);
  }
  Serial.println("Calibrated internal offset");

  while (! nau.calibrate(NAU7802_CALMOD_OFFSET)) {
    Serial.println("Failed to calibrate system offset, retrying!");
    delay(1000);
  }
  Serial.println("Calibrated system offset");

/*

The following greyed out code is for the 14 bit adc used for the voltage divider circuit. 

*/

 // Begin can take an optional address and Wire interface
  //if (!mcp.begin(0x68, &Wire)) { 
    //Serial.println("Failed to find MCP3421 chip");
    //while (1) {
      //delay(10); // Avoid a busy-wait loop
    //}
  //}

   // Options: GAIN_1X, GAIN_2X, GAIN_4X, GAIN_8X
  //mcp.setGain(GAIN_2X);
  //Serial.print("Gain set to: ");
  //switch (mcp.getGain()) {
    //case GAIN_1X: Serial.println("1X"); break;
    //case GAIN_2X: Serial.println("2X"); break;
    //case GAIN_4X: Serial.println("4X"); break;
    //case GAIN_8X: Serial.println("8X"); break;
  //}

  // The resolution affects the sample rate (samples per second, SPS)
  // Other options: RESOLUTION_14_BIT (60 SPS), RESOLUTION_16_BIT (15 SPS), RESOLUTION_18_BIT (3.75 SPS)
  //mcp.setResolution(RESOLUTION_14_BIT); // 240 SPS (12-bit)
  //Serial.print("Resolution set to: ");
  //switch (mcp.getResolution()) {
    //case RESOLUTION_12_BIT: Serial.println("12 bits"); break;
    //case RESOLUTION_14_BIT: Serial.println("14 bits"); break;
    //case RESOLUTION_16_BIT: Serial.println("16 bits"); break;
    //case RESOLUTION_18_BIT: Serial.println("18 bits"); break;
  //}

  // Test setting and getting Mode
  //mcp.setMode(MODE_CONTINUOUS); // Options: MODE_CONTINUOUS, MODE_ONE_SHOT
  //Serial.print("Mode set to: ");
  //switch (mcp.getMode()) {
    //case MODE_CONTINUOUS: Serial.println("Continuous"); break;
    //case MODE_ONE_SHOT: Serial.println("One-shot"); break;
  //}
  
  delay(2000);
}

void loop() {
      
      // read serial port 
      if (Serial.available() > 0) {
        //d stores string data read from serial port
        d = Serial.readString();
        //d converted to integer
        state = d.toInt();
        //following if statement used to ensure data sent from python to arduino is held constant and not reverted back to 0
        //otherwise servo goes to desired angle and immidiately reverts back to 0 degrees
        if  ((state <= 10) ) {  //&& (state > 0)
          // if python sends a 1, angle is 0 degrees.  2 becomes 10 degrees, etc
          // 7 produduces an IMU angle of about 0 degrees!!
          i  = (-10*state + 70);  
          myservo.write(i); 
          delay(1000); 
         }     
        }

       
      
    // printing the following into serial port:
    // theoretical angle (deg)   imu angle (deg)   current sensor (mA)
    //Serial.print(i);
    //Serial.print("\t");
    bno.getEvent(&event);
    Serial.print(i);
    Serial.print("\t");
    /* Display the floating point data */
    Serial.print(event.orientation.y, 2);
    Serial.print("\t");
    delay(50); 

      // Check if MCP3421 has completed a conversion in continuous mode
    //if (mcp.isReady()) {
        int32_t sg = nau.read();
        Serial.print(sg);
        //int32_t adcValue = mcp.readADC(); // Read ADC value
        //Serial.print("ADC reading: ");
        //Serial.print(adcValue);
        Serial.print("\t");
        //sampleCount++; // Increment the sample count
    //}
    delay(50); 


    val = analogRead(analogPin);  // read the input pin
    Serial.print(val);          // debug value
    Serial.print("\n");


   delay(50); 
    
      
        
}
