# Strain-Sensor-

# ABS Filmanent LM Channels

3D printed ABS channels to be dissolved in acetone:  

![abs filaments](https://github.com/user-attachments/assets/2eabfff1-c124-45fa-aace-cf96298053ce)

CAD file for 3d prints shown in image above:  ...CAD Files/abs_lofted_insert/basic strain gauge 0p02.ipt

#  Auto Bender

All Structural parts (e.g. plates, shaft collars, fasteners) were purchased from goBILDA:  https://www.gobilda.com/

![auto bender parts](https://github.com/user-attachments/assets/b4aeed2e-ea4c-4d33-b1a1-e9932a8000fc)


# Strain Sensor Fabrication Procedure:

![LMSG fab process](https://github.com/user-attachments/assets/2a36e296-50d9-46c9-a4cc-2bd47d94766a)


# AutoBender Components:



# AutoBender Sensor Circuit:

![autobender sensor circuit](https://github.com/user-attachments/assets/f643499c-9904-4688-a8c6-9883a8adff69)

# AutoBender PCB (which inlcudes sensor circuit and servo circuit):

AutoBender PCB design can be found here in ...Fritzing Files/Voltage Divider PCB/bender PCB.fzz

# Interpreting CSV Data File Columns:

Column A:  Disregard.  It was initially meant to provide the theoretical bend angle, but it is wrong and not used in any calculation.

Column B:  Adafruit BNO055 IMU Pitch Euler Angle (degrees)

Column C:  ADC Value as read from Adafruit Sensor MCP3421 18-Bit ADC (The ADC sensor is in parrallel with the Liquid Metal Strain Gauge (LMSG).  The LMSG is in series 			with a 100 ohm resistor).  Since the code is set to read 14-bits, the possible range of values is from 0 to 2^14 - 1.

Column D:  ADC Value as read from Hall Effect Rotary Encoder (https://p3america.com/erc-series/?srsltid=AfmBOoqSsWeZSXtsMrmoHyB0DRfSCzyFlUfcnEff0gpcv0NyMvrGhVz7).  The 		rotary encoder can rotate from 0 to 320 degrees.  There is a mechanical stop at 0 and 320 degrees.  Since the rotary encoder is connected directly to 			Arduino Uno, the adc output from the Arduino can have values from 0 to 2^10 - 1.  

Note about converting rotary encoder adc value to angle in degrees:  

![rot encoder](https://github.com/user-attachments/assets/7d459cf2-2cbf-4c0e-9d99-0efad03670a0)

As a result of this calculation, the rotatry encoder angle should always start at 0 degrees.  The initial IMU angle in degrees typically starts around ~ 3 degrees.  


# Derivation of Change in Resistance Formula as a Function of Axial Strain:

![formula part 1](https://github.com/user-attachments/assets/0d268b8c-dba1-49ee-b6d9-f33db160800e)

If an axial load is applied to a solid in the x-direction, the solid's lateral dimensions in the y- and z-directions will typically contract, depending on the material properties. The ratio of lateral contraction (negative strain) to axial extension (positive strain) is called Poisson's ratio. The definition includes a negative sign, ensuring that Poisson's ratio is a positive value for materials that contract laterally when stretched axially. 

![Poisson's Ratio](https://github.com/user-attachments/assets/228fb441-45f0-472e-9315-fb1b07e9e67f)

![formula part 2](https://github.com/user-attachments/assets/ca11f255-4441-4958-ac1b-f0a5a4bc7124)



References:  

1. Soft Robotics Toolkit. (n.d.). Casting soft robots. Retrieved January 14, 2025, 
	from https://softroboticstoolkit.com/book/export/html/206021
2. Craig Jr., R. R. (2011). Mechanics of materials (3rd ed.). Wiley.
3. Hambley, A.R. (2011). Electrical Engineering Principles and Applications (5th ed.).  Prentice Hall.
4. Callister Jr., W. D., & Rethwisch, D. G.  Materials Science and Engineering: An Introduction (10th ed.).  Wiley.
5. Chen, J., Zhang, J., Luo, Z., Zhang, J., Li, L., Su, Y., Gao, X., Li, Y., Tang, W., Cao, C., Liu, Q., Wang, L., & Li, H. (2020). 
	Superelastic, sensitive, and low hysteresis flexible strain sensor based on wave-patterned liquid metal for 
	human activity monitoring. ACS Applied Materials & Interfaces, 12(19), 22200–22211. 
	https://doi.org/10.1021/acsami.0c04709

# Explanation of Model Developed for Axial Strain as a Function of Bend Angle:

The finger joints are modeled as a hinged joint where the leaf rotation axis is concentric with the knuckle axis.  Additionally, the interial surface of the leaf is in contact with the exterior surface of the knuckle.  The change in length of the sample is calculated from the arc length forumla. This model is presented in the image below:

![Human Knuckle Model](https://github.com/user-attachments/assets/11474a39-39c7-4457-9c3a-eb9dbaa327fb)


However, in our autobender, the interior rotating surface of the leaf is offset from the exterior surface of the knuckle by an amount delta.  The adjusted AutoBender model is presented in the image below:  

![Auto Bender Model](https://github.com/user-attachments/assets/c1693872-65de-482c-af7a-460ffc5fbb39)

# Expected Trend for Normalized Resistance vs Axial Strain:

The plot below shows theoretical curves of normalized resistance versus axial strain for samples with varying lengths L, but identical liquid metal channel length (Lo) and cross-sectional area (A). The following trends are observed:

1.  As sample length, L, increases (assuming same liquid metal channel lengths and cross-sectional area), the slope of the curve decreases.
2.  A longer sample will experience less axial strain at a 90 degree bend angle
3.  For axial strains below approximately 40% (e.g., in human knuckle models), the relationship can be approximated as linear.
4.  As bend radius (R in this case, not r as indicated in human knuckle model image :/) increases, axial strain also increases for the same bend angle. For example, a strain sensor on a larger knuckle would experience greater axial strain compared to one on a smaller knuckle.  

![res vs strain trends](https://github.com/user-attachments/assets/f88a247b-96f5-4efe-886f-15be79454da8)

# "Gluing" LMSG on to fingers:

"paint" Derma-Tac (https://www.smooth-on.com/products/derma-tac/) AND Skin-Tite (https://www.smooth-on.com/products/skin-tite/) onto finger, place sensor on finger for a couple minutes until cross-linking occurs





