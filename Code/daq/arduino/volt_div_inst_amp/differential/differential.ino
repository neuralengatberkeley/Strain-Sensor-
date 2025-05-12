#include <Adafruit_ADS1X15.h>
//#include <Adafruit_DS3502.h>  // Digital Potentiometer Library no longer used
#include <Servo.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>


Adafruit_ADS1115 ads;  /* Use this for the 16-bit version */
//Adafruit_ADS1015 ads;     /* Use this for the 12-bit version */

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

void setup(void)
{
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




  // The ADC input range (or gain) can be changed via the following
  // functions, but be careful never to exceed VDD +0.3V max, or to
  // exceed the upper and lower limits if you adjust the input range!
  // Setting these values incorrectly may destroy your ADC!
  //                                                                ADS1015  ADS1115
  //                                                                -------  -------
  // ads.setGain(GAIN_TWOTHIRDS);  // 2/3x gain +/- 6.144V  1 bit = 3mV      0.1875mV (default)
  // ads.setGain(GAIN_ONE);        // 1x gain   +/- 4.096V  1 bit = 2mV      0.125mV
  ads.setGain(GAIN_TWO);        // 2x gain   +/- 2.048V  1 bit = 1mV      0.0625mV
  //ads.setGain(GAIN_FOUR);       // 4x gain   +/- 1.024V  1 bit = 0.5mV    0.03125mV
   //ads.setGain(GAIN_EIGHT);      // 8x gain   +/- 0.512V  1 bit = 0.25mV   0.015625mV
  //ads.setGain(GAIN_SIXTEEN);    // 16x gain  +/- 0.256V  1 bit = 0.125mV  0.0078125mV

  if (!ads.begin()) {
    Serial.println("Failed to initialize ADS.");
    while (1);
  }}

void loop(void)
{
  int16_t results;

  /* Be sure to update this value based on the IC and the gain settings! */
  //float   multiplier = 3.0F;    /* ADS1015 @ +/- 6.144V gain (12-bit results) */
  float multiplier = 0.1875F; /* ADS1115  @ +/- 6.144V gain (16-bit results) */

 

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


    results = ads.readADC_Differential_0_1();
    Serial.print(results); 
      
        Serial.print("\t");
        //sampleCount++; // Increment the sample count
    //}
    delay(50); 


    val = analogRead(analogPin);  // read the input pin
    Serial.print(val);          // debug value
    Serial.print("\n");


  delay(50);
}
