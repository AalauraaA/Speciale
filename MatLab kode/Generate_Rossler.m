function [x] = Generate_Rossler(N,conf)
% it is recommended to run this code several times and average the results
% to show that results are not dependent on additional random noise
% conf is number of configuration (see rosslerpaper function)
% N number of samples used

tspan = linspace(0,50,N); % 50 second and sample rate is 60 Hz
Initial = [1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0]; % initial condition for solving Rossler oscillator equation
c = random('norm',3,1,6,1); % additional noise which is selected from random normal distibution % Random(normal, mean, diviation, x, y)
a = find(c<0);
if ~isempty(a)
    c = random('norm',3,1,6,1);
end

%ode45 Solve nonstiff differential equations — medium order method
% @(t,y) difinnere akserne, lige bagefter er dif. ligningen givet ud fra rosslerpaper
% rosslerpaper(akse-t,akse-y,-conf kan være 1-6 efter hvad netværkstype, c er ekstra støj)
%tspan er hvor mange punkter i garfen der skal plottes.
% y0 er initerende værdi
[t,y] = ode45(@(t,y) rosslerpaper(t,y,conf,c), tspan, Initial); % solving rossler model with s1 configuration (for more detail see rosslerpaper function)
x = y(:,[1,4,7,10,13,16]);  % extracting X segments
x = x(61:end,:);
end
plot(linspace(0,50,size(x,1)),x(:,1));
plot(linspace(0,50,size(x,1)),x(:,2));
plot(linspace(0,50,size(x,1)),x(:,3));
plot(linspace(0,50,size(x,1)),x(:,4));
plot(linspace(0,50,size(x,1)),x(:,5));
plot(linspace(0,50,size(x,1)),x(:,6));

