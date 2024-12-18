/* This arduino Code is used for the auto-bender, VOLTAGE DIVIDER circuit, DYNAMIC TEST.

The following actions are taking by the auto bender:

(1) rotate a servo motor 90 deg repeatedly and  (2) acquire data from the following sensors:  

 [A] Strain Gauge voltage data ouput from :  (1) Voltage Divider Circuit Adafruit_MCP3421 (https://www.adafruit.com/product/5870) 
[B] Adafruit BNO055 (https://www.adafruit.com/product/2472)
 [C] 320 degree Rotary Hall Effect Encoder (https://p3america.com/erc-series/?srsltid=AfmBOoqSsWeZSXtsMrmoHyB0DRfSCzyFlUfcnEff0gpcv0NyMvrGhVz7)

 Servo rotates max of 0.18 sec / 60 deg

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

//Adafruit_NAU7802 nau;  // 24 bit ADC object created for wheatstone bridge circuit
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
int32_t adcValue = 0;

void setup() {
  Serial.begin(115200);
   
   myservo.write(0); // bend plate down to 90 first before starting test
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



/* 

Here we have the option of selecting the excitation voltage to the weatstone bridge.  To my understanding, the larger the excitation voltage, 
the larger potential for increased SNR.

*/



/*

The following greyed out code is for the 14 bit adc used for the voltage divider circuit. 

*/

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
          delay(300); 
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
        //int32_t sg = nau.read();
        //Serial.print(sg);
        adcValue = mcp.readADC(); // Read ADC value
        //Serial.print("ADC reading: ");
        Serial.print(adcValue);
      
        Serial.print("\t");
        //sampleCount++; // Increment the sample count
    //}
    delay(50); 


    val = analogRead(analogPin);  // read the input pin
    Serial.print(val);          // debug value
    Serial.print("\n");


   delay(50); 
    
      
        
}
