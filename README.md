# Strain-Sensor-

# Derivation of Change in Resistance Formula as a Function of Axial Strain:

![formula part 1](https://github.com/user-attachments/assets/0d268b8c-dba1-49ee-b6d9-f33db160800e)


![Poisson's Ratio](https://github.com/user-attachments/assets/21063726-7ad0-4903-adce-4f93755d61b2)


![formula part 2](https://github.com/user-attachments/assets/61505956-694a-480d-be09-1cd1001e062b)


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





