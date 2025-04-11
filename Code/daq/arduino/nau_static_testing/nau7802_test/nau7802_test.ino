#include <Adafruit_NAU7802.h>
#include <Servo.h>
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>


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

Adafruit_NAU7802 nau;

void setup() {
  Serial.begin(115200);
  Serial.println("NAU7802");
  if (! nau.begin()) {
    Serial.println("Failed to find NAU7802");
    while (1) delay(10);  // Don't proceed.
  }
  Serial.println("Found NAU7802");

  nau.setLDO(NAU7802_EXTERNAL);
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

  nau.setGain(NAU7802_GAIN_1);
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

  nau.setRate(NAU7802_RATE_320SPS);
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


}

void loop() {
  while (! nau.available()) {
    delay(1);
  }

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
          int32_t adcval = nau.read();
          Serial.print(adcval);
      
        Serial.print("\t");
        //sampleCount++; // Increment the sample count
    //}
    delay(50); 


    val = analogRead(analogPin);  // read the input pin
    Serial.print(val);          // debug value
    Serial.print("\n");


   delay(50); 
    
      
        
}


