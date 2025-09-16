%% Parameters (consistent units)
rho_SI = 2.9e-7;                 % resistivity [Ohm·m] (adjust if different material)
rho    = rho_SI * 39.37007874;   % convert to [Ohm·inch] so L,w,h may be in inches

L  = 2.00;     % gauge length [in]
h  = 0.02;     % thickness [in]
w  = 0.02;     % width [in]
ro = 0.40;     % nominal/base resistance R0 [Ohm]

he    = 0.125; % neutral-axis offset [in]
delta = 0.250; % additional offset [in]

% Choose a few knuckle radii to compare
R_list = [0.25, 0.30, 0.40, 0.50];   % [in]

% Bend angle sweep (degrees -> radians)
theta_deg = linspace(0, 90, 361);    % 0:0.25:90 for smooth curves
theta_rad = deg2rad(theta_deg);      % radians for arc/curvature relations

%% Plot ΔR/R0 vs ε for each R
figure; hold on; grid on; box on;
set(gcf, 'Color', 'w');

for i = 1:numel(R_list)
    R = R_list(i);

    % Strain model (your expression): eps = (R + he + delta) * theta / L
    % NOTE: theta must be in radians for geometric relationships.
    eps = (R + he + delta) .* theta_rad ./ L;

    % Avoid the singularity at eps = 2 in the denominator
    valid = eps < 2;   % keep eps strictly below 2
    eps_v = eps(valid);

    % ΔR/R0 model (vectorized) — fixed variable name and element-wise ops
    % dR_R0 = (rho * L * eps * (8 - eps)) / (ro * w * h * (2 - eps)^2)
    dR_R0 = (rho .* L .* eps_v .* (8 - eps_v)) ./ (ro .* w .* h .* (2 - eps_v).^2);

    plot(eps_v, dR_R0, 'LineWidth', 1.6, ...
        'DisplayName', sprintf('R = %.2f in', R));
end

xlabel('\epsilon (strain, dimensionless)');
ylabel('\DeltaR / R_0');
title('\DeltaR/R_0 vs \epsilon for different knuckle radii R');
legend('Location','best'); hold off;



