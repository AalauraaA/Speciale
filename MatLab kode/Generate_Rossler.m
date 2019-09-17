function [x] = Generate_Rossler(N,conf)
% It is recommended to run this code several times and average the results to show that results are not 
% dependent on additional random noise.
%   conf: is number of configuration (see rosslerpaper function)
%   N:    is the number of samples used
tspan = linspace(0,50,N); 
% tspan: 50 seconds and sample rate is 60 Hz
Initial = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]; 
% Initial: Is initial condition for solving Rossler oscillator equation
c = random('norm',3,1,6,1); 
% c: additional noise which is selected from random normal distibution 
% random(distribution, A, B, M, N), where A and B is parameter values and
% M and N is the size.
a = find(c<0);
% find: this function finds the value in c that are less than 0 and collect
% them in the variable a
if ~isempty(a) 
    % if not empty meaning if there is c<0 then run random again
    % ~: mean 'not' and isempty
    c = random('norm',3,1,6,1);
end

[t,y] = ode45(@(t,y) rosslerpaper(t,y,conf,c), tspan, Initial); % solving rossler model with s1 configuration (for more detail see rosslerpaper function)
% ode45: Solve nonstiff differential equations, a medium order method.
% @(t,y) defines the axises.
% rosslerpaper(axis-t, axis-y, conf is between 1-6 after what kind of network connection, 
% c is additional noise). t is a scalar and y is a vector
% tspan: integrates the system of differential equations y' = f(t,y) from time 0 to N
% Initial/y0: initial conditions
% Each row in the solution array y corresponds to a time returned in the column vector t

x = y(:,[1,4,7,10,13,16]);  
%x: extracting X segments, here it is 6 segments. We look at all x/rows
% and extract 6 indices of y/columns
x = x(61:end,:);
% x: we start at index 61 to end of the rows and take all the associated
% columns
end

% Plot of the found X segments
%plot(linspace(0,50,size(x,1)),x(:,1));
%plot(linspace(0,50,size(x,1)),x(:,2));
%plot(linspace(0,50,size(x,1)),x(:,3));
%plot(linspace(0,50,size(x,1)),x(:,4));
%plot(linspace(0,50,size(x,1)),x(:,5));
%plot(linspace(0,50,size(x,1)),x(:,6));

