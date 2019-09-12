% This is to generating non-linear AR data.

function F = generate_AR(N)
% N is number of the time points (samples)
% randn('seed', 1);
%% Inital parameters
a1 = 0.05;
a2 = -0.05;
b1 = 0.3; % Connectivity between XX1 and XX2
b2 = 0.4; % Connectivity between XX3 and XX2
c1 = 0.8; % Connectivity between XX1 and XX3
c2 = 0.6;
c3 = 0.2; % Connectivity between XX2 and XX3
d1 = 0.2; % Connectivity between XX1 and XX4
d2 = 0.05;
XX1 = zeros(1, N);
XX2 = zeros(1, N);
XX3 = zeros(1, N);
XX4 = zeros(1, N);
LP = 200;

%% Generating Synthetic AR Data
w = randn(4, N);
for j = 3:N
    XX1(j) = a1 * XX1(j-1) - a2 * XX1(j-2) + w(1, j-2);
    XX2(j) = b1 * XX1(j-1) + b2 * XX3(j-1) + w(2, j-2);
    XX3(j) = c1 * XX1(j-1) .^2 + c2 * XX3(j-1) + c3 * XX2(j-1) + w(3, j-2);
    XX4(j) = d2 * XX4(j-1) + d1 * XX1(j-1) + w(4, j-2);
end
XX1 = XX1(LP + 1 : end);
XX2 = XX2(LP + 1 : end);
XX3 = XX3(LP + 1 : end);
XX4 = XX4(LP + 1 : end);
F(1,:) = XX1;
F(2,:) = XX2;
F(3,:) = XX3;
F(4,:) = XX4;
end