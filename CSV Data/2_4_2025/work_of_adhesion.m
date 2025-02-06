E = 68947.6; % Pa
t = (0.0625 - 0.02)/(2 * 0.0254); % channel cover thickenss
nu = 0.5;
gamma = 20*10^-3 + 530 * 10^-3; %j/m^2
h = 0.02; 
a = 0.02;

Dbottom = E .* t.^3 ./ (12.*(1-nu.^2));
Dtop = Dbottom; 
D = Dtop

g_channel = gamma .* (2.*D).*a^4 ./ (2.* (D.^2) .* (h.^2))

